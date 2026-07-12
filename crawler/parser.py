"""Article extraction and normalized HTML generation."""

from __future__ import annotations

import re
from html import escape

from crawler.models import ArticleMetadata, DiscoveredArticle, ParsedArticle, RenderedPage
from crawler.utils import parse_datetime, strip_html, to_iso_z


def parse_rendered_article(page: RenderedPage, discovered: DiscoveredArticle) -> ParsedArticle:
    """Extract clean article text and metadata from rendered HTML.

    Args:
        page: Rendered article page returned by PlaywrightFetcher.
        discovered: Candidate metadata collected during source discovery.

    Returns:
        ParsedArticle with metadata and body paragraphs.
    """

    newspaper_data = _parse_with_newspaper(page)
    fallback_data = _parse_with_beautifulsoup(page.html)

    title = _first_non_empty(
        newspaper_data.get("title"),
        fallback_data.get("title"),
        discovered.title,
    )
    url = _first_non_empty(
        newspaper_data.get("url"),
        fallback_data.get("url"),
        page.final_url,
        discovered.url,
    )
    author = _first_non_empty(
        newspaper_data.get("author"),
        fallback_data.get("author"),
    )
    published_at = (
        parse_datetime(newspaper_data.get("published_date"))
        or parse_datetime(fallback_data.get("published_date"))
        or discovered.published_at
    )

    paragraphs = _select_paragraphs(
        newspaper_data.get("paragraphs", ()),
        fallback_data.get("paragraphs", ()),
    )

    metadata = ArticleMetadata(
        source=discovered.source_name,
        source_key=discovered.source_key,
        title=title or "Untitled Article",
        url=url,
        published_date=to_iso_z(published_at),
        author=author,
        category=discovered.category or "Other",
    )
    return ParsedArticle(metadata=metadata, paragraphs=tuple(paragraphs))


def build_normalized_html(article: ParsedArticle) -> str:
    """Build ingestion-ready HTML with normalized metadata and article body.

    Args:
        article: Parsed article to serialize.

    Returns:
        UTF-8 HTML string containing title/meta tags and an article element.
    """

    metadata = article.metadata
    meta_tags = {
        "url": metadata.url,
        "source": metadata.source,
        "published_date": metadata.published_date,
        "date": metadata.published_date,
        "author": metadata.author,
        "category": metadata.category,
    }
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        f"  <title>{escape(metadata.title)}</title>",
    ]
    for name, value in meta_tags.items():
        lines.append(f'  <meta name="{escape(name)}" content="{escape(value or "")}">')
    lines.extend(
        [
            f'  <meta property="og:url" content="{escape(metadata.url or "")}">',
            "</head>",
            "<body>",
            "  <article>",
            f"    <h1>{escape(metadata.title)}</h1>",
        ]
    )
    for paragraph in article.paragraphs:
        lines.append(f"    <p>{escape(paragraph)}</p>")
    lines.extend(["  </article>", "</body>", "</html>", ""])
    return "\n".join(lines)


def _parse_with_newspaper(page: RenderedPage) -> dict:
    """Parse article data with newspaper3k when available."""

    try:
        from newspaper import Article
    except ImportError:
        return {}

    try:
        article = Article(url=page.final_url or page.requested_url)
        article.set_html(page.html)
        article.parse()
    except Exception:
        return {}

    paragraphs = _paragraphs_from_text(article.text or "")
    publish_date = str(article.publish_date) if article.publish_date else ""
    author = ", ".join(article.authors) if article.authors else ""
    return {
        "title": article.title or "",
        "url": _metadata_url(article.meta_data) or page.final_url,
        "author": author,
        "published_date": publish_date,
        "paragraphs": paragraphs,
    }


def _parse_with_beautifulsoup(html: str) -> dict:
    """Parse article data with BeautifulSoup or return an empty fallback."""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {"paragraphs": _paragraphs_from_text(strip_html(html))}

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "form"]):
        tag.decompose()

    article_node = soup.find("article") or soup.body or soup
    paragraph_nodes = article_node.find_all("p") if article_node else []
    paragraphs = [_normalize_text(node.get_text(" ", strip=True)) for node in paragraph_nodes]
    paragraphs = [paragraph for paragraph in paragraphs if _is_content_paragraph(paragraph)]

    return {
        "title": _extract_title(soup),
        "url": _extract_meta(soup, ("og:url", "twitter:url", "url")),
        "author": _extract_meta(soup, ("author", "article:author", "parsely-author")),
        "published_date": _extract_meta(
            soup,
            (
                "article:published_time",
                "datePublished",
                "date",
                "pubdate",
                "publication_date",
                "parsely-pub-date",
            ),
        ),
        "paragraphs": tuple(paragraphs),
    }


def _metadata_url(metadata: dict) -> str:
    """Extract URL from newspaper3k metadata."""

    if not metadata:
        return ""
    og_data = metadata.get("og")
    if isinstance(og_data, dict):
        return str(og_data.get("url") or "")
    return ""


def _extract_title(soup) -> str:
    """Extract a title from common HTML metadata and headings."""

    meta_title = _extract_meta(soup, ("og:title", "twitter:title", "title"))
    if meta_title:
        return meta_title
    if soup.title and soup.title.string:
        return _normalize_text(soup.title.string)
    heading = soup.find("h1")
    return _normalize_text(heading.get_text(" ", strip=True)) if heading else ""


def _extract_meta(soup, names: tuple[str, ...]) -> str:
    """Extract one metadata value by name, property, or itemprop."""

    for name in names:
        for attrs in ({"name": name}, {"property": name}, {"itemprop": name}):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                return _normalize_text(tag["content"])
    return ""


def _select_paragraphs(*candidates: tuple[str, ...] | list[str]) -> list[str]:
    """Pick the longest useful paragraph extraction result."""

    useful_sets: list[list[str]] = []
    for candidate in candidates:
        paragraphs = [_normalize_text(paragraph) for paragraph in candidate]
        paragraphs = [paragraph for paragraph in paragraphs if _is_content_paragraph(paragraph)]
        if paragraphs:
            useful_sets.append(paragraphs)
    if not useful_sets:
        return []
    return max(useful_sets, key=lambda paragraphs: sum(len(item) for item in paragraphs))


def _paragraphs_from_text(text: str) -> tuple[str, ...]:
    """Split article text into normalized paragraphs."""

    parts = re.split(r"\n{2,}|\r\n{2,}", text or "")
    paragraphs = [_normalize_text(part) for part in parts]
    return tuple(paragraph for paragraph in paragraphs if _is_content_paragraph(paragraph))


def _is_content_paragraph(value: str) -> bool:
    """Check whether text is long enough to be article content."""

    return len(value.split()) >= 8 and len(value) >= 40


def _normalize_text(value: str) -> str:
    """Collapse whitespace in a text fragment."""

    return re.sub(r"\s+", " ", strip_html(value)).strip()


def _first_non_empty(*values: str | None) -> str:
    """Return the first non-empty string from a list of candidates."""

    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""

