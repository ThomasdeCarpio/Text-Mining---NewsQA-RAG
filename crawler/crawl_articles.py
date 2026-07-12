"""CLI for crawling recent English news articles into project-ready HTML.

Usage:
    python crawler/crawl_articles.py --max-articles 50
    python crawler/crawl_articles.py --sources reuters,bbc --categories World,Business
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crawler.fetcher import FetchConfig, PlaywrightFetcher
from crawler.filters import text_contains_vietnam_terms
from crawler.models import DiscoveredArticle, ParsedArticle
from crawler.parser import build_normalized_html, parse_rendered_article
from crawler.sources import (
    discover_source_articles,
    get_source_configs,
    parse_categories,
    parse_source_keys,
)
from crawler.storage import (
    ArticleStorage,
    CompositeStorage,
    FilesystemStorage,
    HuggingFaceBucketStorage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "articles"
DEFAULT_RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_articles"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "crawl_manifest.jsonl"

LOGGER = logging.getLogger("crawler")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured ArgumentParser for the crawler CLI.
    """

    parser = argparse.ArgumentParser(description="Crawl recent trusted English news articles into normalized HTML.")
    parser.add_argument("--sources", default=None, help="Comma-separated source keys. Default: reuters,bbc,npr,guardian,ap")
    parser.add_argument("--categories", default=None, help="Comma-separated categories. Example: World,Business,Technology")
    parser.add_argument("--since-hours", type=int, default=72, help="Maximum feed article age in hours. Default: 72")
    parser.add_argument("--max-articles", type=int, default=50, help="Total article crawl limit. Default: 50")
    parser.add_argument("--per-source-limit", type=int, default=15, help="Candidate limit per source. Default: 15")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for normalized HTML.")
    parser.add_argument("--raw-output-dir", default=str(DEFAULT_RAW_OUTPUT_DIR), help="Directory for rendered raw HTML.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="JSONL manifest output path.")
    parser.add_argument(
        "--storage",
        choices=("filesystem", "hf-bucket", "both"),
        default=os.getenv("CRAWLER_STORAGE", "hf-bucket"),
        help="Output backend. Default: CRAWLER_STORAGE or hf-bucket.",
    )
    parser.add_argument(
        "--hf-bucket-id",
        default=os.getenv("HF_BUCKET_ID", ""),
        help="Hugging Face Bucket ID. Defaults to HF_BUCKET_ID.",
    )
    parser.add_argument(
        "--hf-bucket-prefix",
        default=os.getenv("HF_BUCKET_PREFIX", ""),
        help="Optional object-key prefix. Defaults to HF_BUCKET_PREFIX.",
    )
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Playwright navigation timeout in milliseconds.")
    parser.add_argument("--wait-ms", type=int, default=1000, help="Extra wait after page load in milliseconds.")
    parser.add_argument("--wait-selector", default=None, help="Optional CSS selector to wait for before reading HTML.")
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Delay between article fetches. Default: 1.0")
    parser.add_argument("--min-words", type=int, default=120, help="Minimum article word count to save. Default: 120")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing article HTML files.")
    parser.add_argument("--include-vietnam", action="store_true", help="Allow Vietnam-related articles.")
    parser.add_argument("--headed", action="store_true", help="Run Chromium with a visible browser window.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level. Default: INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the crawler command-line workflow.

    Args:
        argv: Optional CLI argument list. Uses sys.argv when omitted.

    Returns:
        Process exit code.
    """

    _load_environment()
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)

    source_keys = parse_source_keys(args.sources)
    categories = parse_categories(args.categories)
    fetch_config = FetchConfig(
        timeout_ms=args.timeout_ms,
        wait_ms=args.wait_ms,
        wait_selector=args.wait_selector,
        headless=not args.headed,
    )

    stats = {"success": 0, "failed": 0, "skipped": 0}
    LOGGER.info("Starting crawler for sources: %s", ", ".join(source_keys))

    try:
        with _build_storage(args) as storage:
            with PlaywrightFetcher(fetch_config) as fetcher:
                for source in get_source_configs(source_keys):
                    if stats["success"] >= args.max_articles:
                        break

                    LOGGER.info("Discovering %s articles", source.name)
                    candidates = discover_source_articles(
                        source=source,
                        categories=categories,
                        since_hours=args.since_hours,
                        per_source_limit=args.per_source_limit,
                        include_vietnam=args.include_vietnam,
                        render_html=lambda url: fetcher.fetch(url).html,
                    )
                    LOGGER.info("Discovered %d candidate(s) for %s", len(candidates), source.name)

                    for article in candidates:
                        if stats["success"] >= args.max_articles:
                            break
                        result = crawl_one_article(
                            article=article,
                            fetcher=fetcher,
                            storage=storage,
                            min_words=args.min_words,
                            overwrite=args.overwrite,
                            include_vietnam=args.include_vietnam,
                        )
                        stats[result] += 1
                        if args.delay_seconds > 0:
                            time.sleep(args.delay_seconds)
    except Exception as exc:
        LOGGER.error("Crawler stopped: %s", exc)
        return 1

    LOGGER.info(
        "Crawler finished: %d saved, %d skipped, %d failed",
        stats["success"],
        stats["skipped"],
        stats["failed"],
    )
    return 0 if stats["success"] > 0 or stats["failed"] == 0 else 1


def _build_storage(args: argparse.Namespace) -> ArticleStorage:
    """Build the storage backend selected by CLI and environment settings.

    Args:
        args: Parsed crawler CLI arguments.

    Returns:
        Configured local, Hugging Face Bucket, or composite storage backend.
    """

    backends: list[ArticleStorage] = []
    if args.storage in {"filesystem", "both"}:
        backends.append(
            FilesystemStorage(
                output_dir=Path(args.output_dir),
                raw_output_dir=Path(args.raw_output_dir),
                manifest_path=Path(args.manifest),
            )
        )

    if args.storage in {"hf-bucket", "both"}:
        backends.append(
            HuggingFaceBucketStorage(
                bucket_id=args.hf_bucket_id,
                token=os.getenv("HF_TOKEN", ""),
                private=_parse_boolean(os.getenv("HF_BUCKET_PRIVATE", "true")),
                prefix=args.hf_bucket_prefix,
            )
        )

    if len(backends) == 1:
        return backends[0]
    return CompositeStorage(backends)


def _load_environment() -> None:
    """Load project-root environment variables from an optional .env file."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _parse_boolean(value: str) -> bool:
    """Parse a conventional environment boolean value.

    Args:
        value: Environment value such as true, false, 1, or 0.

    Returns:
        Parsed boolean value.

    Raises:
        ValueError: If the value is not a recognized boolean spelling.
    """

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def crawl_one_article(
    article: DiscoveredArticle,
    fetcher: PlaywrightFetcher,
    storage: ArticleStorage,
    min_words: int,
    overwrite: bool,
    include_vietnam: bool,
) -> str:
    """Fetch, parse, and store one article.

    Args:
        article: Candidate article discovered from a source.
        fetcher: Started PlaywrightFetcher for page rendering.
        storage: Backend receiving article outputs and crawl manifests.
        min_words: Minimum parsed article word count required for saving.
        overwrite: Whether existing stored articles should be replaced.
        include_vietnam: Whether to allow Vietnam-related parsed content.

    Returns:
        One of "success", "skipped", or "failed" for summary accounting.
    """

    try:
        if storage.article_exists(article) and not overwrite:
            LOGGER.info("Skipping existing article: %s", article.url)
            _write_manifest(storage, article, "skipped", reason="exists")
            return "skipped"

        LOGGER.info("Fetching article: %s", article.url)
        rendered = fetcher.fetch(article.url)
        parsed = parse_rendered_article(rendered, article)
        parsed_text = f"{parsed.metadata.url} {parsed.metadata.title} {' '.join(parsed.paragraphs)}"
        if not include_vietnam and text_contains_vietnam_terms(parsed_text):
            LOGGER.info("Skipping Vietnam-related parsed article: %s", article.url)
            _write_manifest(storage, article, "skipped", parsed=parsed, reason="vietnam_related")
            return "skipped"

        if parsed.word_count < min_words:
            reason = f"too_short:{parsed.word_count}"
            LOGGER.info("Skipping short article (%s words): %s", parsed.word_count, article.url)
            _write_manifest(storage, article, "skipped", parsed=parsed, reason=reason)
            return "skipped"

        locations = storage.save_article(
            article=parsed,
            raw_html=rendered.html,
            clean_html=build_normalized_html(parsed),
            status_code=rendered.status_code,
            overwrite=overwrite,
        )
        _write_manifest(
            storage,
            article,
            "success",
            parsed=parsed,
            clean_location=locations.clean_location,
            raw_location=locations.raw_location,
            metadata_location=locations.metadata_location,
            status_code=rendered.status_code,
        )
        LOGGER.info("Saved article: %s", locations.clean_location)
        return "success"
    except FileExistsError:
        LOGGER.info("Skipping existing article: %s", article.url)
        _write_manifest(storage, article, "skipped", reason="exists")
        return "skipped"
    except Exception as exc:
        LOGGER.warning("Failed article %s: %s", article.url, exc)
        _write_manifest(storage, article, "failed", reason=str(exc))
        return "failed"


def configure_logging(level_name: str) -> None:
    """Configure process-level logging.

    Args:
        level_name: Logging level name such as INFO or DEBUG.

    Returns:
        None.
    """

    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _write_manifest(
    storage: ArticleStorage,
    article: DiscoveredArticle,
    status: str,
    parsed: ParsedArticle | None = None,
    clean_location: str = "",
    raw_location: str = "",
    metadata_location: str = "",
    status_code: int | None = None,
    reason: str = "",
) -> None:
    """Record one crawl status with article metadata and storage locations.

    Args:
        storage: Backend receiving the crawl manifest record.
        article: Candidate article associated with this outcome.
        status: Crawl outcome such as success, skipped, or failed.
        parsed: Parsed article metadata when extraction reached that stage.
        clean_location: Local path or remote URI for normalized HTML.
        raw_location: Local path or remote URI for raw rendered HTML.
        metadata_location: Optional local path or remote URI for metadata.
        status_code: HTTP response status when available.
        reason: Human-readable skip or failure reason.

    Returns:
        None.
    """

    metadata = parsed.metadata if parsed else None
    storage.record_manifest(
        {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
            "reason": reason,
            "source": article.source_name,
            "source_key": article.source_key,
            "category": article.category,
            "url": article.url,
            "title": metadata.title if metadata else article.title,
            "published_date": metadata.published_date if metadata else "",
            "author": metadata.author if metadata else "",
            "word_count": parsed.word_count if parsed else 0,
            "status_code": status_code,
            "clean_location": clean_location,
            "raw_location": raw_location,
            "metadata_location": metadata_location,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
