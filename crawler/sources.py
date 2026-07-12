"""Source registry and article discovery helpers."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.parse import urljoin

from crawler.filters import (
    deduplicate_articles,
    is_allowed_domain,
    is_probable_article_url,
    should_skip_article,
)
from crawler.models import DiscoveredArticle, FeedConfig, IndexPageConfig, SourceConfig
from crawler.utils import canonicalize_url, parse_datetime, strip_html

LOGGER = logging.getLogger(__name__)

DEFAULT_SOURCE_ORDER = ("reuters", "bbc", "npr", "guardian", "ap")

SOURCE_REGISTRY: dict[str, SourceConfig] = {
    "reuters": SourceConfig(
        key="reuters",
        name="Reuters",
        allowed_domains=("reuters.com", "www.reuters.com"),
        index_pages=(
            IndexPageConfig("https://www.reuters.com/world/", "World"),
            IndexPageConfig("https://www.reuters.com/business/", "Business"),
            IndexPageConfig("https://www.reuters.com/technology/", "Technology"),
            IndexPageConfig("https://www.reuters.com/science/", "Science"),
            IndexPageConfig("https://www.reuters.com/business/healthcare-pharmaceuticals/", "Health"),
        ),
    ),
    "bbc": SourceConfig(
        key="bbc",
        name="BBC",
        allowed_domains=("bbc.com", "bbc.co.uk", "bbci.co.uk", "feeds.bbci.co.uk"),
        feeds=(
            FeedConfig("https://feeds.bbci.co.uk/news/world/rss.xml", "World"),
            FeedConfig("https://feeds.bbci.co.uk/news/business/rss.xml", "Business"),
            FeedConfig("https://feeds.bbci.co.uk/news/technology/rss.xml", "Technology"),
            FeedConfig("https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", "Science"),
            FeedConfig("https://feeds.bbci.co.uk/news/health/rss.xml", "Health"),
        ),
    ),
    "npr": SourceConfig(
        key="npr",
        name="NPR",
        allowed_domains=("npr.org", "www.npr.org", "feeds.npr.org"),
        feeds=(
            FeedConfig("https://feeds.npr.org/1001/rss.xml", "World"),
            FeedConfig("https://feeds.npr.org/1004/rss.xml", "World"),
            FeedConfig("https://feeds.npr.org/1006/rss.xml", "Business"),
            FeedConfig("https://feeds.npr.org/1019/rss.xml", "Technology"),
            FeedConfig("https://feeds.npr.org/1007/rss.xml", "Science"),
        ),
    ),
    "guardian": SourceConfig(
        key="guardian",
        name="The Guardian",
        allowed_domains=("theguardian.com", "www.theguardian.com"),
        feeds=(
            FeedConfig("https://www.theguardian.com/world/rss", "World"),
            FeedConfig("https://www.theguardian.com/us-news/rss", "World"),
            FeedConfig("https://www.theguardian.com/business/rss", "Business"),
            FeedConfig("https://www.theguardian.com/technology/rss", "Technology"),
            FeedConfig("https://www.theguardian.com/science/rss", "Science"),
            FeedConfig("https://www.theguardian.com/environment/rss", "Science"),
        ),
    ),
    "ap": SourceConfig(
        key="ap",
        name="AP News",
        allowed_domains=("apnews.com", "www.apnews.com"),
        index_pages=(
            IndexPageConfig("https://apnews.com/world-news", "World"),
            IndexPageConfig("https://apnews.com/business", "Business"),
            IndexPageConfig("https://apnews.com/technology", "Technology"),
            IndexPageConfig("https://apnews.com/science", "Science"),
            IndexPageConfig("https://apnews.com/health", "Health"),
        ),
    ),
}


def parse_source_keys(value: str | None) -> list[str]:
    """Parse a comma-separated source key list.

    Args:
        value: Comma-separated source keys, or None for defaults.

    Returns:
        Ordered source keys to use for discovery.

    Raises:
        ValueError: If an unknown source key is requested.
    """

    keys = list(DEFAULT_SOURCE_ORDER) if not value else [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in SOURCE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown source(s): {', '.join(unknown)}")
    return keys


def parse_categories(value: str | None) -> set[str] | None:
    """Parse a comma-separated category filter.

    Args:
        value: Comma-separated category names, or None for all categories.

    Returns:
        Set of category names, or None when no filter is requested.
    """

    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def get_source_configs(keys: list[str]) -> list[SourceConfig]:
    """Return source configs in the requested order.

    Args:
        keys: Source keys validated by parse_source_keys.

    Returns:
        Ordered SourceConfig objects.
    """

    return [SOURCE_REGISTRY[key] for key in keys]


def discover_source_articles(
    source: SourceConfig,
    categories: set[str] | None,
    since_hours: int | None,
    per_source_limit: int,
    include_vietnam: bool,
    render_html: Callable[[str], str] | None = None,
) -> list[DiscoveredArticle]:
    """Discover recent article candidates for one source.

    Args:
        source: Source configuration to query.
        categories: Optional set of project categories to include.
        since_hours: Optional maximum article age based on feed metadata.
        per_source_limit: Maximum candidates returned for this source.
        include_vietnam: Whether to allow Vietnam-related candidates.
        render_html: Callback used to render index pages for non-feed sources.

    Returns:
        Newest-first article candidates after domain, topic, and duplicate filtering.
    """

    candidates: list[DiscoveredArticle] = []
    if source.feeds:
        candidates.extend(_discover_feed_articles(source, categories, since_hours))
    if source.index_pages and render_html is not None:
        candidates.extend(_discover_index_articles(source, categories, render_html, per_source_limit))

    filtered: list[DiscoveredArticle] = []
    for candidate in sort_newest_first(deduplicate_articles(candidates)):
        should_skip, reason = should_skip_article(candidate, include_vietnam=include_vietnam)
        if should_skip:
            LOGGER.info("Skipping %s: %s", candidate.url, reason)
            continue
        filtered.append(candidate)
        if len(filtered) >= per_source_limit:
            break
    return filtered


def sort_newest_first(articles: list[DiscoveredArticle]) -> list[DiscoveredArticle]:
    """Sort candidates by publication date when available.

    Args:
        articles: Candidate articles to sort.

    Returns:
        Articles with dated entries first and newest entries before older entries.
    """

    minimum = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        articles,
        key=lambda article: article.published_at or minimum,
        reverse=True,
    )


def _discover_feed_articles(
    source: SourceConfig,
    categories: set[str] | None,
    since_hours: int | None,
) -> list[DiscoveredArticle]:
    """Discover articles from RSS or Atom feeds."""

    try:
        import feedparser
    except ImportError as exc:
        raise RuntimeError("feedparser is required for RSS discovery. Install crawler/requirements-crawler.txt.") from exc

    cutoff = None
    if since_hours is not None and since_hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    articles: list[DiscoveredArticle] = []
    for feed in source.feeds:
        if categories and feed.category not in categories:
            continue
        parsed_feed = feedparser.parse(feed.url)
        if getattr(parsed_feed, "bozo", False):
            LOGGER.warning("Feed parse warning for %s: %s", feed.url, getattr(parsed_feed, "bozo_exception", "unknown"))
        for entry in parsed_feed.entries:
            link = canonicalize_url(entry.get("link", ""))
            if not link or not is_allowed_domain(link, source.allowed_domains):
                continue
            if not is_probable_article_url(link):
                continue
            published_at = parse_datetime(entry.get("published") or entry.get("updated"))
            if cutoff and published_at and published_at < cutoff:
                continue
            articles.append(
                DiscoveredArticle(
                    source_key=source.key,
                    source_name=source.name,
                    url=link,
                    title=strip_html(entry.get("title", "")),
                    summary=strip_html(entry.get("summary", "")),
                    published_at=published_at,
                    category=feed.category,
                )
            )
    return articles


def _discover_index_articles(
    source: SourceConfig,
    categories: set[str] | None,
    render_html: Callable[[str], str],
    per_source_limit: int,
) -> list[DiscoveredArticle]:
    """Discover articles from rendered source index pages."""

    articles: list[DiscoveredArticle] = []
    for page in source.index_pages:
        if categories and page.category not in categories:
            continue
        try:
            html = render_html(page.url)
        except Exception as exc:
            LOGGER.warning("Failed to render index page %s: %s", page.url, exc)
            continue
        articles.extend(extract_index_articles(source, page, html, per_page_limit=per_source_limit))
        if len(articles) >= per_source_limit:
            break
    return articles


def extract_index_articles(
    source: SourceConfig,
    page: IndexPageConfig,
    html: str,
    per_page_limit: int,
) -> list[DiscoveredArticle]:
    """Extract article candidates from one rendered index page.

    Args:
        source: Source configuration owning the index page.
        page: Index page metadata and category.
        html: Rendered HTML for the index page.
        per_page_limit: Maximum article candidates to return.

    Returns:
        Article candidates in the order they appear on the page.
    """

    anchors = _extract_anchors(html)
    articles: list[DiscoveredArticle] = []
    seen: set[str] = set()
    for href, text in anchors:
        url = canonicalize_url(urljoin(page.url, href))
        if url in seen:
            continue
        if not is_allowed_domain(url, source.allowed_domains):
            continue
        if not is_probable_article_url(url):
            continue
        seen.add(url)
        articles.append(
            DiscoveredArticle(
                source_key=source.key,
                source_name=source.name,
                url=url,
                title=strip_html(text),
                category=page.category,
            )
        )
        if len(articles) >= per_page_limit:
            break
    return articles


def _extract_anchors(html: str) -> list[tuple[str, str]]:
    """Extract anchor href/text pairs using BeautifulSoup or a regex fallback."""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return [
            (match.group("href"), strip_html(match.group("text")))
            for match in re.finditer(
                r"<a\s+[^>]*href=[\"'](?P<href>[^\"']+)[\"'][^>]*>(?P<text>.*?)</a>",
                html or "",
                flags=re.IGNORECASE | re.DOTALL,
            )
        ]

    soup = BeautifulSoup(html or "", "html.parser")
    anchors: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        anchors.append((anchor.get("href", ""), anchor.get_text(" ", strip=True)))
    return anchors
