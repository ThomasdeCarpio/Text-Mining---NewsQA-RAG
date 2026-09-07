import unittest

from newsqa_rag.ingestion.chunker import (
    HierarchicalChunker,
    ParagraphChunker,
    SentenceChunker,
    get_chunker,
)


def article(text: str) -> dict:
    return {"text": text, "metadata": {"url": "", "title": "Example"}}


class BoundaryChunkerTest(unittest.TestCase):
    def test_sentence_chunks_are_verbatim_and_end_at_boundaries(self):
        text = "Alpha happened here. Beta followed later! Gamma ended the report?"
        chunks = SentenceChunker(chunk_size=7, chunk_overlap=0).chunk_article(
            article(text), "article-a"
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(row["text"] in text for row in chunks))
        self.assertTrue(all(row["text"].endswith((".", "!", "?")) for row in chunks))
        self.assertTrue(
            all(row["metadata"]["chunking_strategy"] == "sentence" for row in chunks)
        )

    def test_paragraph_chunks_preserve_paragraph_boundaries(self):
        text = "First paragraph has facts.\n\nSecond paragraph adds details.\n\nFinal paragraph."
        chunks = ParagraphChunker(chunk_size=8, chunk_overlap=0).chunk_article(
            article(text), "article-b"
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(row["text"] in text for row in chunks))
        self.assertTrue(
            all(row["metadata"]["chunking_strategy"] == "paragraph" for row in chunks)
        )

    def test_oversized_sentence_uses_recursive_fallback(self):
        text = " ".join(f"token{i}" for i in range(80)) + "."
        chunks = SentenceChunker(chunk_size=16, chunk_overlap=2).chunk_article(
            article(text), "article-c"
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(row["text"] in text for row in chunks))


class HierarchicalArtifactTest(unittest.TestCase):
    def test_children_carry_reconstructable_parent_payload(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = HierarchicalChunker(
            chunk_size=64,
            chunk_overlap=8,
            child_chunk_size=24,
            child_chunk_overlap=4,
        ).chunk_article(article(text), "article-d")
        self.assertGreater(len(chunks), 1)
        for row in chunks:
            self.assertIn(row["text"], row["parent_text"])
            self.assertIn(row["parent_text"], text)
            self.assertTrue(row["metadata"]["parent_id"])

    def test_factory_exposes_all_registered_phase2c_strategies(self):
        expected = {
            "sentence": SentenceChunker,
            "paragraph": ParagraphChunker,
            "hierarchical": HierarchicalChunker,
        }
        for strategy, expected_type in expected.items():
            chunker = get_chunker(
                {
                    "chunking": {
                        "strategy": strategy,
                        "chunk_size": 128,
                        "chunk_overlap": 16,
                        "child_chunk_size": 64,
                        "child_chunk_overlap": 8,
                    }
                }
            )
            self.assertIsInstance(chunker, expected_type)


if __name__ == "__main__":
    unittest.main()
