from pathlib import Path
from unittest import TestCase


class SlidesTest(TestCase):
    def test_deck_is_complete_and_offline(self):
        root = Path(__file__).parents[1] / "docs" / "slides"
        if not (root / "index.html").exists():
            self.skipTest("optional presentation deck is not present in this checkout")
        html = (root / "index.html").read_text(encoding="utf-8")

        self.assertEqual(html.count('<section class="slide'), 15)
        self.assertEqual(html.count('<aside class="notes">'), 15)
        self.assertNotIn("https://", html)
        self.assertTrue((root / "assets" / "slides.css").is_file())
        self.assertTrue((root / "assets" / "slides.js").is_file())
