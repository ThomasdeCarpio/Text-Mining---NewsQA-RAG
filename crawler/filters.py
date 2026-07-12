"""Filtering rules for allowed sources, duplicates, and sensitive topics."""

from __future__ import annotations

from urllib.parse import urlsplit

from crawler.models import DiscoveredArticle
from crawler.utils import canonicalize_url

EXCLUDED_PATH_PARTS = (
    "/audio/",
    "/authors/",
    "/cartoons/",
    "/help/",
    "/live/",
    "/live-news/",
    "/live-updates/",
    "/newsletter",
    "/newsletters",
    "/opinion/",
    "/podcast",
    "/podcasts",
    "/pictures/",
    "/video/",
    "/videos/",
)

VIETNAM_TERMS = (
    "vietnam",
    "viet nam",
    "viet-nam",
    "hanoi",
    "ha noi",
    "ho chi minh",
    "saigon",
    "communist party of vietnam",
    "south china sea",
)


def is_allowed_domain(url: str, allowed_domains: tuple[str, ...]) -> bool:
    """Check whether a URL belongs to an allowed source domain.

    Args:
        url: Absolute URL to validate.
        allowed_domains: Allowed hostnames for the selected source.

    Returns:
        True when the hostname matches one of the allowed domains.
    """

    hostname = (urlsplit(url).hostname or "").lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains)


def is_excluded_url(url: str) -> bool:
    """Check whether a URL points to a non-article or low-value page.

    Args:
        url: Absolute URL to evaluate.

    Returns:
        True when the URL path looks like opinion, live, video, or utility content.
    """

    path = urlsplit(url).path.lower()
    return any(part in path for part in EXCLUDED_PATH_PARTS)


def is_probable_article_url(url: str) -> bool:
    """Check whether a URL path is specific enough to be an article.

    Args:
        url: Absolute URL to evaluate.

    Returns:
        True when the path has enough detail and is not an obvious section page.
    """

    path = urlsplit(url).path.strip("/")
    if not path or is_excluded_url(url):
        return False
    segments = [segment for segment in path.split("/") if segment]
    return len(segments) >= 2 and len(segments[-1]) >= 12


def is_vietnam_related(article: DiscoveredArticle) -> bool:
    """Detect Vietnam-related candidates that should be skipped by default.

    Args:
        article: Candidate article with URL, title, and summary.

    Returns:
        True when the candidate mentions Vietnam-related terms.
    """

    text = f"{article.url} {article.title} {article.summary}"
    return text_contains_vietnam_terms(text)


def text_contains_vietnam_terms(text: str) -> bool:
    """Detect Vietnam-related terms in arbitrary article text.

    Args:
        text: Text to scan, usually title, URL, summary, or parsed body.

    Returns:
        True when the text contains Vietnam-related terms.
    """

    normalized = (text or "").lower()
    return any(term in normalized for term in VIETNAM_TERMS)


def should_skip_article(article: DiscoveredArticle, include_vietnam: bool = False) -> tuple[bool, str]:
    """Decide whether a discovered article should be skipped.

    Args:
        article: Candidate article to evaluate.
        include_vietnam: Whether to allow Vietnam-related content.

    Returns:
        Tuple of (should_skip, reason). Reason is empty when not skipped.
    """

    if is_excluded_url(article.url):
        return True, "excluded_url_type"
    if not include_vietnam and is_vietnam_related(article):
        return True, "vietnam_related"
    return False, ""


def deduplicate_articles(articles: list[DiscoveredArticle]) -> list[DiscoveredArticle]:
    """Remove duplicate article candidates while preserving order.

    Args:
        articles: Candidate articles, usually already sorted newest-first.

    Returns:
        List with only the first occurrence of each canonical URL.
    """

    seen: set[str] = set()
    unique: list[DiscoveredArticle] = []
    for article in articles:
        key = canonicalize_url(article.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return unique
