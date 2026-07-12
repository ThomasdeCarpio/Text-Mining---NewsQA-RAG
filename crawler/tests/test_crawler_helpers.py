import unittest

from crawler.filters import should_skip_article
from crawler.models import DiscoveredArticle, IndexPageConfig, SourceConfig, ArticleMetadata, ParsedArticle
from crawler.parser import build_normalized_html
from crawler.sources import extract_index_articles
from crawler.utils import canonicalize_url, safe_slug, url_hash


class CrawlerHelperTests(unittest.TestCase):
    def test_canonical_url_removes_tracking(self):
        url = "HTTPS://Example.com/news/story/?utm_source=x&keep=1&fbclid=abc#section"
        self.assertEqual(canonicalize_url(url), "https://example.com/news/story?keep=1")

    def test_safe_slug_and_hash_are_stable(self):
        self.assertEqual(safe_slug("Fed Raises Rates Again!"), "fed-raises-rates-again")
        self.assertEqual(url_hash("https://example.com/a?utm_source=x"), url_hash("https://example.com/a"))

    def test_vietnam_filter_defaults_to_skip(self):
        article = DiscoveredArticle(
            source_key="reuters",
            source_name="Reuters",
            url="https://www.reuters.com/world/vietnam-example-story-2026-01-01/",
            title="Vietnam policy story",
        )
        self.assertEqual(should_skip_article(article, include_vietnam=False), (True, "vietnam_related"))
        self.assertEqual(should_skip_article(article, include_vietnam=True), (False, ""))

    def test_normalized_html_contains_required_metadata(self):
        article = ParsedArticle(
            metadata=ArticleMetadata(
                source="Reuters",
                source_key="reuters",
                title="Markets rally after rate decision",
                url="https://www.reuters.com/markets/example-2026-01-01/",
                published_date="2026-01-01T10:00:00Z",
                author="Jane Doe",
                category="Business",
            ),
            paragraphs=("Markets rallied after the central bank announced its latest rate decision.",),
        )
        html = build_normalized_html(article)
        self.assertIn('<meta name="source" content="Reuters">', html)
        self.assertIn('<meta name="published_date" content="2026-01-01T10:00:00Z">', html)
        self.assertIn("<article>", html)
        self.assertIn("<p>Markets rallied", html)

    def test_extract_index_articles_keeps_allowed_article_links(self):
        source = SourceConfig(
            key="reuters",
            name="Reuters",
            allowed_domains=("reuters.com", "www.reuters.com"),
        )
        page = IndexPageConfig("https://www.reuters.com/world/", "World")
        html = """
        <html><body>
          <a href="/world/europe/long-enough-story-slug-2026-01-01/">Story title from Reuters</a>
          <a href="/video/">Video page</a>
          <a href="https://example.com/world/not-allowed-2026-01-01/">External</a>
        </body></html>
        """
        articles = extract_index_articles(source, page, html, per_page_limit=10)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].source_name, "Reuters")
        self.assertEqual(articles[0].category, "World")


if __name__ == "__main__":
    unittest.main()
