"""Shared data models for the news crawler."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class FeedConfig:
    """RSS feed configuration for one source/category pair.

    Args:
        url: Absolute RSS or Atom feed URL.
        category: Project category assigned to articles from this feed.
    """

    url: str
    category: str


@dataclass(frozen=True)
class IndexPageConfig:
    """Index page configuration used to discover article links.

    Args:
        url: Section URL that lists recent articles.
        category: Project category assigned to articles discovered from the page.
    """

    url: str
    category: str


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for a supported news source.

    Args:
        key: Stable CLI key, used in filenames and source selection.
        name: Canonical source name stored in article metadata.
        allowed_domains: Domains that are valid for this source.
        feeds: RSS/Atom feeds for newest-first discovery.
        index_pages: Web index pages for sources without reliable public feeds.
    """

    key: str
    name: str
    allowed_domains: tuple[str, ...]
    feeds: tuple[FeedConfig, ...] = ()
    index_pages: tuple[IndexPageConfig, ...] = ()


@dataclass(frozen=True)
class DiscoveredArticle:
    """Article candidate discovered before fetching the full article page.

    Args:
        source_key: Stable source key from SourceConfig.
        source_name: Canonical source name stored in metadata.
        url: Absolute article URL.
        title: Candidate title from feed/index page.
        summary: Candidate summary or description from feed/index page.
        published_at: Publication datetime if available from discovery.
        category: Project category assigned to the article.
    """

    source_key: str
    source_name: str
    url: str
    title: str = ""
    summary: str = ""
    published_at: datetime | None = None
    category: str = "Other"


@dataclass(frozen=True)
class RenderedPage:
    """Rendered page returned by the browser fetcher.

    Args:
        requested_url: Original URL sent to the browser.
        final_url: URL after redirects.
        html: Rendered HTML after JavaScript execution and waits.
        status_code: HTTP response status code when available.
    """

    requested_url: str
    final_url: str
    html: str
    status_code: int | None = None


@dataclass(frozen=True)
class ArticleMetadata:
    """Metadata stored in the normalized article HTML.

    Args:
        source: Canonical news outlet name.
        source_key: Stable source key used in filenames.
        title: Article title.
        url: Canonical or final article URL.
        published_date: ISO 8601 UTC timestamp, or an empty string.
        author: Comma-separated author names, or an empty string.
        category: Project category.
    """

    source: str
    source_key: str
    title: str
    url: str
    published_date: str
    author: str
    category: str


@dataclass(frozen=True)
class ParsedArticle:
    """Clean article content extracted from rendered HTML.

    Args:
        metadata: Metadata to embed in the normalized HTML.
        paragraphs: Ordered article body paragraphs.
    """

    metadata: ArticleMetadata
    paragraphs: tuple[str, ...]

    @property
    def word_count(self) -> int:
        """Return the approximate word count across all body paragraphs."""

        return sum(len(paragraph.split()) for paragraph in self.paragraphs)


@dataclass(frozen=True)
class CrawlOutput:
    """Filesystem outputs created for a crawled article.

    Args:
        clean_path: Path to normalized ingestion-ready HTML.
        raw_path: Path to raw rendered HTML.
    """

    clean_path: Path
    raw_path: Path


@dataclass(frozen=True)
class StorageLocations:
    """Locations created by an article storage backend.

    Args:
        clean_location: Filesystem path or remote URI for normalized HTML.
        raw_location: Filesystem path or remote URI for rendered raw HTML.
        metadata_location: Optional filesystem path or remote URI for metadata.
    """

    clean_location: str
    raw_location: str
    metadata_location: str = ""
