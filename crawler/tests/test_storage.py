import gzip
import json
import unittest

from crawler.models import ArticleMetadata, DiscoveredArticle, ParsedArticle
from crawler.storage import HuggingFaceBucketStorage


class FakeBucketClient:
    """In-memory Hugging Face Bucket client used by storage unit tests."""

    def __init__(self):
        """Initialize empty bucket state and captured upload calls."""

        self.bucket_id = ""
        self.private = None
        self.paths = set()
        self.uploads = []

    def ensure_bucket(self, bucket_id, private):
        """Capture bucket creation settings without making a network call."""

        self.bucket_id = bucket_id
        self.private = private

    def paths_exist(self, bucket_id, paths):
        """Return requested paths already present in the fake bucket."""

        self.bucket_id = bucket_id
        return self.paths.intersection(paths)

    def upload(self, bucket_id, files):
        """Capture uploaded byte payloads and mark their paths as present."""

        self.bucket_id = bucket_id
        self.uploads.append(files)
        self.paths.update(path for _, path in files)


class HuggingFaceBucketStorageTests(unittest.TestCase):
    """Verify Hugging Face Bucket object layout and serialization behavior."""

    def setUp(self):
        """Create a fake private bucket backend and representative article."""

        self.client = FakeBucketClient()
        self.storage = HuggingFaceBucketStorage(
            bucket_id="user/newsqa-crawler",
            token="test-token",
            private=True,
            prefix="news",
            client=self.client,
        )
        self.discovered = DiscoveredArticle(
            source_key="reuters",
            source_name="Reuters",
            url="https://www.reuters.com/world/example-2026-07-12/",
            title="Example article",
            category="World",
        )
        self.parsed = ParsedArticle(
            metadata=ArticleMetadata(
                source="Reuters",
                source_key="reuters",
                title="Example article",
                url=self.discovered.url,
                published_date="2026-07-12T08:00:00Z",
                author="Jane Doe",
                category="World",
            ),
            paragraphs=("This is a sufficiently useful article paragraph for a crawler storage test.",),
        )

    def test_uploads_compressed_html_and_metadata_to_deterministic_paths(self):
        self.assertFalse(self.storage.article_exists(self.discovered))

        locations = self.storage.save_article(
            article=self.parsed,
            raw_html="<html>raw</html>",
            clean_html="<html>clean</html>",
            status_code=200,
            overwrite=False,
        )

        self.assertEqual(len(self.client.uploads), 1)
        uploaded = {path: payload for payload, path in self.client.uploads[0]}
        clean_path = next(path for path in uploaded if path.endswith("clean.html.gz"))
        raw_path = next(path for path in uploaded if path.endswith("raw.html.gz"))
        metadata_path = next(path for path in uploaded if path.endswith("metadata.json"))
        self.assertEqual(gzip.decompress(uploaded[clean_path]).decode(), "<html>clean</html>")
        self.assertEqual(gzip.decompress(uploaded[raw_path]).decode(), "<html>raw</html>")
        metadata = json.loads(uploaded[metadata_path])
        self.assertEqual(metadata["source_key"], "reuters")
        self.assertEqual(metadata["status_code"], 200)
        self.assertTrue(locations.clean_location.startswith("hf://buckets/user/newsqa-crawler/"))
        self.assertTrue(self.storage.article_exists(self.discovered))

    def test_close_uploads_one_compressed_manifest(self):
        self.storage.record_manifest({"status": "success", "url": self.discovered.url})
        self.storage.record_manifest({"status": "skipped", "url": "https://example.com/other"})

        self.storage.close()
        self.storage.close()

        self.assertEqual(len(self.client.uploads), 1)
        payload, path = self.client.uploads[0][0]
        records = [json.loads(line) for line in gzip.decompress(payload).decode().splitlines()]
        self.assertEqual([record["status"] for record in records], ["success", "skipped"])
        self.assertIn("news/manifests/", path)
        self.assertTrue(path.endswith(".jsonl.gz"))

    def test_rejects_existing_article_without_overwrite(self):
        paths = self.storage._article_paths(self.discovered.source_key, self.discovered.url)
        self.client.paths.update(paths.values())

        with self.assertRaises(FileExistsError):
            self.storage.save_article(
                article=self.parsed,
                raw_html="<html>raw</html>",
                clean_html="<html>clean</html>",
                status_code=200,
                overwrite=False,
            )

    def test_partial_article_is_not_treated_as_complete(self):
        paths = self.storage._article_paths(self.discovered.source_key, self.discovered.url)
        self.client.paths.add(paths["metadata"])

        self.assertFalse(self.storage.article_exists(self.discovered))


if __name__ == "__main__":
    unittest.main()
