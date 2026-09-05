"""Tests for the retrieval configuration the Phase 1 tournament locked in."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from newsqa_app.services import retrieval_service


class LockedConfigTests(unittest.TestCase):
    """The shipped config must be the configuration the tournament selected."""

    def test_config_matches_the_tournament_winner(self):
        """Guard every knob notebooks/13 locked, so a silent edit fails here."""

        config = retrieval_service.load_config()

        self.assertEqual(config["chunking"]["strategy"], "recursive")
        self.assertEqual(config["chunking"]["chunk_size"], 512)
        self.assertEqual(config["chunking"]["chunk_overlap"], 64)

        retrieval = config["retrieval"]
        self.assertEqual(retrieval["retriever"], "sparse")
        self.assertEqual(retrieval["top_k"], 20)
        self.assertEqual(retrieval["sparse"]["method"], "bge-m3")
        self.assertEqual(retrieval["sparse"]["model"], "BAAI/bge-m3")
        self.assertEqual(retrieval["reranker"]["model"], "BAAI/bge-reranker-large")
        self.assertEqual(retrieval["reranker"]["top_n"], 5)
        self.assertFalse(retrieval["hybrid"]["enabled"], "hybrid lost to sparse in round 2")

    def test_reranker_type_key_is_the_one_the_factory_reads(self):
        """`provider` was ignored by get_reranker, so the app ran noop in silence."""

        from newsqa_rag.retrieval.reranker import get_reranker

        info = get_reranker(retrieval_service.load_config()).get_info()

        self.assertEqual(info["type"], "cross-encoder")
        self.assertEqual(info["model"], "BAAI/bge-reranker-large")


class LockedRetrievalPathTests(unittest.TestCase):
    """The locked search path, with the two model-loading stages faked out."""

    def setUp(self):
        retrieval_service._locked = None
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        (self.root / "chunks.jsonl").write_text(
            json.dumps({"id": "c1", "text": "hello", "metadata": {}}) + "\n",
            encoding="utf-8",
        )
        (self.root / "bge_m3_sparse.pkl").write_bytes(b"not a real index")

    def tearDown(self):
        retrieval_service._locked = None
        self._temp.cleanup()

    def _env(self):
        return patch.dict("os.environ", {"RAG_LOCKED_INDEX_DIR": str(self.root)})

    def test_availability_follows_the_artifacts_on_disk(self):
        """Both files present means available; either missing means it is not."""

        with self._env():
            self.assertTrue(retrieval_service.locked_is_available())
            self.assertTrue(retrieval_service.is_available())
            ids = {a["id"]: a["available"] for a in retrieval_service.list_algorithms()}
            self.assertTrue(ids["locked"])

            (self.root / "bge_m3_sparse.pkl").unlink()
            self.assertFalse(retrieval_service.locked_is_available())

    def test_missing_artifacts_name_the_notebook_that_builds_them(self):
        """A 503 that does not say what to run is a support ticket."""

        with patch.dict("os.environ", {"RAG_LOCKED_INDEX_DIR": str(self.root / "absent")}):
            with self.assertRaises(retrieval_service.RetrievalUnavailableError) as caught:
                retrieval_service.search("q", "locked", 5)

        self.assertIn("14_export_locked_index_kaggle", str(caught.exception))

    def test_search_retrieves_the_candidate_pool_then_reranks_to_top_k(self):
        """A reranker can only reorder what it is handed, so top_k must not gate it."""

        retriever = Mock()
        retriever.retrieve.return_value = [
            {"id": f"c{i}", "text": "t", "metadata": {}, "score": 0.5} for i in range(20)
        ]
        reranker = Mock()
        reranker.rerank.return_value = retriever.retrieve.return_value[:3]
        config = retrieval_service.load_config()
        retrieval_service._locked = (retriever, reranker, config)

        with self._env():
            results, timing = retrieval_service.search("who signed it", "locked", 3)

        retriever.retrieve.assert_called_once_with("who signed it", 20)
        reranker.rerank.assert_called_once()
        self.assertEqual(reranker.rerank.call_args[0][2], 3)
        self.assertEqual(len(results), 3)
        # Sparse scores are similarities; the Chroma-shaped response needs a distance.
        self.assertAlmostEqual(results[0]["distance"], 0.5)
        self.assertIn("total_ms", timing)

    def test_sparse_algorithm_skips_the_reranker(self):
        """The un-reranked variant exists to show what the reranker is worth."""

        retriever = Mock()
        retriever.retrieve.return_value = [
            {"id": f"c{i}", "text": "t", "metadata": {}, "score": 0.5} for i in range(20)
        ]
        reranker = Mock()
        retrieval_service._locked = (retriever, reranker, retrieval_service.load_config())

        with self._env():
            results, _ = retrieval_service.search("q", "sparse", 4)

        reranker.rerank.assert_not_called()
        self.assertEqual(len(results), 4)

    def test_top_k_larger_than_the_pool_widens_the_pool(self):
        """Asking for 50 results must not silently return a 20-candidate pool."""

        retriever = Mock()
        retriever.retrieve.return_value = []
        reranker = Mock()
        reranker.rerank.return_value = []
        retrieval_service._locked = (retriever, reranker, retrieval_service.load_config())

        with self._env():
            retrieval_service.search("q", "locked", 50)

        retriever.retrieve.assert_called_once_with("q", 50)


if __name__ == "__main__":
    unittest.main()
