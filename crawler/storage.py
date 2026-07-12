"""Storage backends for crawled article HTML and crawl manifests."""

from __future__ import annotations

import gzip
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from crawler.models import DiscoveredArticle, ParsedArticle, StorageLocations
from crawler.utils import (
    append_jsonl,
    build_output_paths,
    canonicalize_url,
    find_existing_by_url_hash,
    safe_slug,
    url_hash,
    write_text_file,
)


class ArticleStorage(ABC):
    """Interface implemented by crawler storage backends."""

    def __enter__(self) -> "ArticleStorage":
        """Return this storage backend for context-manager use."""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Flush pending storage work when leaving a context manager."""

        self.close()

    @abstractmethod
    def article_exists(self, article: DiscoveredArticle) -> bool:
        """Return whether an article URL is already stored."""

    @abstractmethod
    def save_article(
        self,
        article: ParsedArticle,
        raw_html: str,
        clean_html: str,
        status_code: int | None,
        overwrite: bool,
    ) -> StorageLocations:
        """Store one parsed article and return its storage locations."""

    @abstractmethod
    def record_manifest(self, record: dict) -> None:
        """Record one crawl outcome for the current run."""

    @abstractmethod
    def close(self) -> None:
        """Flush buffered records and release storage resources."""


class FilesystemStorage(ArticleStorage):
    """Store crawler outputs in project-local files.

    Args:
        output_dir: Directory for normalized ingestion-ready HTML.
        raw_output_dir: Directory for rendered raw HTML.
        manifest_path: JSONL file receiving crawl outcome records.
    """

    def __init__(self, output_dir: Path, raw_output_dir: Path, manifest_path: Path) -> None:
        self.output_dir = output_dir
        self.raw_output_dir = raw_output_dir
        self.manifest_path = manifest_path

    def article_exists(self, article: DiscoveredArticle) -> bool:
        """Return whether normalized HTML for the article URL exists locally."""

        return find_existing_by_url_hash(self.output_dir, article.url) is not None

    def save_article(
        self,
        article: ParsedArticle,
        raw_html: str,
        clean_html: str,
        status_code: int | None,
        overwrite: bool,
    ) -> StorageLocations:
        """Write raw and normalized article HTML to local directories.

        Args:
            article: Parsed article and normalized metadata.
            raw_html: Rendered source page HTML.
            clean_html: Normalized ingestion-ready HTML.
            status_code: HTTP response status, unused by this backend.
            overwrite: Whether existing output files may be replaced.

        Returns:
            Local paths for the clean and raw HTML files.
        """

        del status_code
        outputs = build_output_paths(article, self.output_dir, self.raw_output_dir)
        write_text_file(outputs.raw_path, raw_html, overwrite=overwrite)
        write_text_file(outputs.clean_path, clean_html, overwrite=overwrite)
        return StorageLocations(
            clean_location=str(outputs.clean_path),
            raw_location=str(outputs.raw_path),
        )

    def record_manifest(self, record: dict) -> None:
        """Append one crawl outcome to the local JSONL manifest."""

        append_jsonl(self.manifest_path, record)

    def close(self) -> None:
        """Finish local storage; no buffered resources require flushing."""


class HuggingFaceBucketClient:
    """Thin wrapper around the Hugging Face Bucket Python API.

    Args:
        token: Hugging Face write token used for all bucket requests.
    """

    def __init__(self, token: str) -> None:
        try:
            from huggingface_hub import batch_bucket_files, create_bucket, get_bucket_paths_info
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub with Bucket support is required. "
                "Install crawler/requirements-crawler.txt."
            ) from exc

        self.token = token
        self._batch_bucket_files = batch_bucket_files
        self._create_bucket = create_bucket
        self._get_bucket_paths_info = get_bucket_paths_info

    def ensure_bucket(self, bucket_id: str, private: bool) -> None:
        """Create the bucket when missing and preserve an existing bucket."""

        self._create_bucket(bucket_id, private=private, exist_ok=True, token=self.token)

    def paths_exist(self, bucket_id: str, paths: Iterable[str]) -> set[str]:
        """Return the subset of bucket paths that currently exist."""

        return {
            item.path
            for item in self._get_bucket_paths_info(bucket_id, paths, token=self.token)
        }

    def upload(self, bucket_id: str, files: list[tuple[bytes, str]]) -> None:
        """Upload raw byte payloads to their destination bucket paths."""

        self._batch_bucket_files(bucket_id, add=files, token=self.token)


class HuggingFaceBucketStorage(ArticleStorage):
    """Store compressed crawler outputs in a Hugging Face Bucket.

    Args:
        bucket_id: Bucket identifier in ``namespace/bucket-name`` format.
        token: Hugging Face token with write access to the bucket.
        private: Whether a newly created bucket should be private.
        prefix: Optional object-key prefix used to isolate crawler data.
        client: Optional bucket client override, primarily for testing.
        upload_attempts: Maximum attempts for idempotent bucket uploads.
    """

    def __init__(
        self,
        bucket_id: str,
        token: str,
        private: bool = True,
        prefix: str = "",
        client: HuggingFaceBucketClient | None = None,
        upload_attempts: int = 3,
    ) -> None:
        if not bucket_id.strip():
            raise ValueError("HF_BUCKET_ID is required for Hugging Face Bucket storage.")
        if not token.strip():
            raise ValueError("HF_TOKEN is required for Hugging Face Bucket storage.")
        if upload_attempts < 1:
            raise ValueError("upload_attempts must be at least 1.")

        self.bucket_id = bucket_id.strip().strip("/")
        self.token = token.strip()
        self.private = private
        self.prefix = _normalize_prefix(prefix)
        self.client = client or HuggingFaceBucketClient(self.token)
        self.upload_attempts = upload_attempts
        self.manifest_records: list[dict] = []
        self.run_id = _build_run_id()
        self._closed = False
        self.client.ensure_bucket(self.bucket_id, private=self.private)

    def article_exists(self, article: DiscoveredArticle) -> bool:
        """Check the deterministic metadata object for an existing article."""

        paths = self._article_paths(article.source_key, article.url)
        return self._all_paths_exist(paths)

    def save_article(
        self,
        article: ParsedArticle,
        raw_html: str,
        clean_html: str,
        status_code: int | None,
        overwrite: bool,
    ) -> StorageLocations:
        """Upload compressed HTML and JSON metadata for one article.

        Args:
            article: Parsed article and normalized metadata.
            raw_html: Rendered source page HTML.
            clean_html: Normalized ingestion-ready HTML.
            status_code: HTTP response status from the article fetch.
            overwrite: Whether a previously stored article may be replaced.

        Returns:
            Hugging Face Bucket URIs for clean HTML, raw HTML, and metadata.

        Raises:
            FileExistsError: If the article exists and overwrite is disabled.
            RuntimeError: If every upload attempt fails.
        """

        paths = self._article_paths(article.metadata.source_key, article.metadata.url)
        if not overwrite and self._all_paths_exist(paths):
            raise FileExistsError(f"Article already exists in Hugging Face Bucket: {article.metadata.url}")

        metadata = self._build_metadata(article, raw_html, clean_html, status_code, paths)
        files = [
            (_gzip_text(clean_html), paths["clean"]),
            (_gzip_text(raw_html), paths["raw"]),
            (_json_bytes(metadata), paths["metadata"]),
        ]
        self._upload_with_retry(files)
        return StorageLocations(
            clean_location=self._uri(paths["clean"]),
            raw_location=self._uri(paths["raw"]),
            metadata_location=self._uri(paths["metadata"]),
        )

    def record_manifest(self, record: dict) -> None:
        """Buffer one crawl outcome for a single end-of-run upload."""

        self.manifest_records.append(record)

    def close(self) -> None:
        """Compress and upload the buffered JSONL crawl manifest once."""

        if self._closed:
            return
        self._closed = True
        if not self.manifest_records:
            return

        now = datetime.now(timezone.utc)
        manifest_path = _join_key(
            self.prefix,
            "manifests",
            now.strftime("%Y"),
            now.strftime("%m"),
            now.strftime("%d"),
            f"{self.run_id}.jsonl.gz",
        )
        jsonl = "".join(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
            for record in self.manifest_records
        )
        self._upload_with_retry([(_gzip_text(jsonl), manifest_path)])

    def _article_paths(self, source_key: str, url: str) -> dict[str, str]:
        """Build deterministic bucket paths from a source key and URL."""

        source = safe_slug(source_key, max_length=32)
        article_hash = url_hash(url, length=64)
        base = _join_key(self.prefix, "articles", source, article_hash[:2], article_hash)
        return {
            "clean": f"{base}/clean.html.gz",
            "raw": f"{base}/raw.html.gz",
            "metadata": f"{base}/metadata.json",
        }

    def _all_paths_exist(self, paths: dict[str, str]) -> bool:
        """Return whether every object required for an article exists.

        Args:
            paths: Mapping containing clean HTML, raw HTML, and metadata paths.

        Returns:
            True only when the bucket contains every required object.
        """

        required = set(paths.values())
        return required == self.client.paths_exist(self.bucket_id, required)

    def _build_metadata(
        self,
        article: ParsedArticle,
        raw_html: str,
        clean_html: str,
        status_code: int | None,
        paths: dict[str, str],
    ) -> dict:
        """Build the metadata document stored beside article HTML objects."""

        metadata = article.metadata
        return {
            "schema_version": 1,
            "source": metadata.source,
            "source_key": metadata.source_key,
            "title": metadata.title,
            "url": metadata.url,
            "canonical_url": canonicalize_url(metadata.url),
            "published_date": metadata.published_date,
            "author": metadata.author,
            "category": metadata.category,
            "word_count": article.word_count,
            "status_code": status_code,
            "stored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "compression": "gzip",
            "clean_path": paths["clean"],
            "raw_path": paths["raw"],
            "clean_sha256": _sha256_text(clean_html),
            "raw_sha256": _sha256_text(raw_html),
        }

    def _upload_with_retry(self, files: list[tuple[bytes, str]]) -> None:
        """Retry an idempotent bucket upload up to the configured limit."""

        last_error: Exception | None = None
        for _ in range(self.upload_attempts):
            try:
                self.client.upload(self.bucket_id, files)
                return
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Hugging Face Bucket upload failed: {last_error}") from last_error

    def _uri(self, path: str) -> str:
        """Return an ``hf://`` URI for a path in this bucket."""

        return f"hf://buckets/{self.bucket_id}/{path}"


class CompositeStorage(ArticleStorage):
    """Write crawler outputs to every configured storage backend.

    Args:
        backends: Storage backends that must all receive each article.
    """

    def __init__(self, backends: Iterable[ArticleStorage]) -> None:
        self.backends = tuple(backends)
        if not self.backends:
            raise ValueError("CompositeStorage requires at least one backend.")

    def article_exists(self, article: DiscoveredArticle) -> bool:
        """Return true only when every backend already contains the article."""

        return all(backend.article_exists(article) for backend in self.backends)

    def save_article(
        self,
        article: ParsedArticle,
        raw_html: str,
        clean_html: str,
        status_code: int | None,
        overwrite: bool,
    ) -> StorageLocations:
        """Store the article in every backend and combine their locations."""

        locations = [
            backend.save_article(article, raw_html, clean_html, status_code, overwrite)
            for backend in self.backends
        ]
        return StorageLocations(
            clean_location=";".join(item.clean_location for item in locations if item.clean_location),
            raw_location=";".join(item.raw_location for item in locations if item.raw_location),
            metadata_location=";".join(
                item.metadata_location for item in locations if item.metadata_location
            ),
        )

    def record_manifest(self, record: dict) -> None:
        """Record the same crawl outcome in every backend."""

        for backend in self.backends:
            backend.record_manifest(record)

    def close(self) -> None:
        """Flush every backend and report all close failures together."""

        errors = []
        for backend in self.backends:
            try:
                backend.close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Failed to close storage backend(s): " + "; ".join(errors))


def _normalize_prefix(prefix: str) -> str:
    """Normalize and validate an optional bucket object-key prefix."""

    normalized = prefix.strip().strip("/")
    if any(part in {".", ".."} for part in normalized.split("/") if part):
        raise ValueError("HF_BUCKET_PREFIX cannot contain '.' or '..' path segments.")
    return normalized


def _join_key(*parts: str) -> str:
    """Join non-empty object-key segments with forward slashes."""

    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def _gzip_text(value: str) -> bytes:
    """Encode UTF-8 text as deterministic gzip bytes."""

    return gzip.compress(value.encode("utf-8"), mtime=0)


def _json_bytes(value: dict) -> bytes:
    """Serialize a dictionary as readable UTF-8 JSON bytes."""

    return (json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8")


def _sha256_text(value: str) -> str:
    """Return the SHA-256 digest for UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_run_id() -> str:
    """Build a sortable unique identifier for one crawler run."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"
