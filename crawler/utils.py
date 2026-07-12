"""General utilities for crawler paths, dates, URLs, and manifests."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from crawler.models import CrawlOutput, ParsedArticle

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_NAMES = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object for the directory.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def canonicalize_url(url: str) -> str:
    """Remove fragments and common tracking query parameters from a URL.

    Args:
        url: URL to normalize.

    Returns:
        URL string with stable scheme/host casing, no fragment, and no tracking query.
    """

    parts = urlsplit(url.strip())
    query_pairs = []
    for name, value in parse_qsl(parts.query, keep_blank_values=True):
        lower_name = name.lower()
        if lower_name in TRACKING_QUERY_NAMES:
            continue
        if any(lower_name.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        query_pairs.append((name, value))

    scheme = parts.scheme.lower() or "https"
    hostname = parts.hostname.lower() if parts.hostname else parts.netloc.lower()
    netloc = hostname
    if parts.port:
        netloc = f"{netloc}:{parts.port}"

    return urlunsplit((scheme, netloc, parts.path.rstrip("/") or "/", urlencode(query_pairs), ""))


def parse_datetime(value: str | None) -> datetime | None:
    """Parse common feed and article datetime formats.

    Args:
        value: Datetime string from RSS, metadata, or article parser.

    Returns:
        Timezone-aware UTC datetime, or None when parsing fails.
    """

    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    parsers = (
        _parse_email_datetime,
        _parse_iso_datetime,
        _parse_dateutil_datetime,
    )
    for parser in parsers:
        parsed = parser(text)
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
    return None


def _parse_email_datetime(value: str) -> datetime | None:
    """Parse RFC 2822 dates commonly used by RSS feeds."""

    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _parse_iso_datetime(value: str) -> datetime | None:
    """Parse ISO-like datetimes with optional trailing Z."""

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_dateutil_datetime(value: str) -> datetime | None:
    """Parse broad datetime formats when python-dateutil is installed."""

    try:
        from dateutil import parser as date_parser
    except ImportError:
        return None

    try:
        return date_parser.parse(value)
    except (TypeError, ValueError, OverflowError):
        return None


def to_iso_z(value: datetime | None) -> str:
    """Format a datetime as ISO 8601 UTC for metadata.

    Args:
        value: Datetime to format.

    Returns:
        Timestamp formatted as YYYY-MM-DDTHH:MM:SSZ, or an empty string.
    """

    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def date_label(value: str | datetime | None) -> str:
    """Return a compact date label for filenames.

    Args:
        value: ISO string, datetime, or None.

    Returns:
        YYYY-MM-DD when available, otherwise "undated".
    """

    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = parse_datetime(value)

    if parsed is None:
        return "undated"
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d")


def safe_slug(value: str, max_length: int = 80) -> str:
    """Convert free text into a stable ASCII filename slug.

    Args:
        value: Text to convert, usually an article title.
        max_length: Maximum slug length before trimming trailing separators.

    Returns:
        Lowercase ASCII slug containing only letters, numbers, and hyphens.
    """

    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    if not slug:
        return "article"
    return slug[:max_length].strip("-") or "article"


def url_hash(url: str, length: int = 12) -> str:
    """Create a short deterministic hash for a URL.

    Args:
        url: URL to hash.
        length: Number of hexadecimal characters to return.

    Returns:
        Stable lowercase hexadecimal hash prefix.
    """

    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def build_output_paths(
    article: ParsedArticle,
    output_dir: str | Path,
    raw_output_dir: str | Path,
) -> CrawlOutput:
    """Build clean and raw HTML output paths for a parsed article.

    Args:
        article: Parsed article with source, title, URL, and date metadata.
        output_dir: Directory for normalized ingestion-ready HTML.
        raw_output_dir: Directory for rendered raw HTML.

    Returns:
        CrawlOutput containing clean and raw output paths.
    """

    metadata = article.metadata
    source_slug = safe_slug(metadata.source_key or metadata.source, max_length=32)
    label = date_label(metadata.published_date)
    stem = f"{label}_{safe_slug(metadata.title)}_{url_hash(metadata.url)}"
    clean_path = Path(output_dir) / f"{source_slug}_{stem}.html"
    raw_path = Path(raw_output_dir) / source_slug / f"{stem}.html"
    return CrawlOutput(clean_path=clean_path, raw_path=raw_path)


def find_existing_by_url_hash(output_dir: str | Path, url: str) -> Path | None:
    """Find an existing normalized HTML file for a URL hash.

    Args:
        output_dir: Directory containing normalized article HTML files.
        url: Article URL whose hash should be matched.

    Returns:
        Matching Path when an article was already crawled, otherwise None.
    """

    directory = Path(output_dir)
    if not directory.exists():
        return None
    suffix = f"_{url_hash(url)}.html"
    for path in directory.glob(f"*{suffix}"):
        if path.is_file():
            return path
    return None


def write_text_file(path: str | Path, content: str, overwrite: bool = False) -> bool:
    """Write text content to disk.

    Args:
        path: Destination file path.
        content: Text content to write as UTF-8.
        overwrite: Whether to replace an existing file.

    Returns:
        True when the file was written, False when it already existed.

    """

    target = Path(path)
    ensure_directory(target.parent)
    if target.exists() and not overwrite:
        return False
    target.write_text(content, encoding="utf-8")
    return True


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one JSON object to a JSON Lines manifest.

    Args:
        path: Manifest JSONL file path.
        record: JSON-serializable record to append.

    Returns:
        None.
    """

    target = Path(path)
    ensure_directory(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def strip_html(value: str) -> str:
    """Remove simple HTML tags from a short text field.

    Args:
        value: HTML or plain text string.

    Returns:
        Unescaped plain text with collapsed whitespace.
    """

    without_tags = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()
