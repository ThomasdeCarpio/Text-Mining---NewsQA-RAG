import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from newsqa_rag.experiments import expand_run_matrix
from newsqa_rag.indexing.bm25_index import BM25Index
from newsqa_rag.indexing.embeddings import SentenceTransformerEmbeddingFunction
from newsqa_rag.indexing.learned_sparse_index import LearnedSparseIndex
from newsqa_rag.retrieval.hybrid import HybridRetriever, SparseIndexRetriever


class _Encoder:
    def __init__(self, rows):
        self.rows = list(rows)

    def encode(self, texts, batch_size=32):
        return [self.rows.pop(0) for _ in texts]


class _Retriever:
    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, query, top_k):
        return [{"id": value, "text": value, "metadata": {}, "score": 1.0} for value in self.ids[:top_k]]


class Phase1RetrievalTests(unittest.TestCase):
    def test_asymmetric_embedding_preprocessing_and_normalization(self):
        model = Mock()
        model.encode.side_effect = lambda texts, **kwargs: Mock(tolist=lambda: [[1.0]] * len(texts))
        embedding = SentenceTransformerEmbeddingFunction("intfloat/e5-base-v2")
        embedding._model = model

        embedding.embed_documents(["article"])
        embedding.embed_queries(["question"])

        self.assertEqual(model.encode.call_args_list[0].args[0], ["passage: article"])
        self.assertEqual(model.encode.call_args_list[1].args[0], ["query: question"])
        self.assertTrue(model.encode.call_args_list[0].kwargs["normalize_embeddings"])

    def test_bge_query_instruction_is_not_added_to_documents(self):
        embedding = SentenceTransformerEmbeddingFunction("BAAI/bge-small-en-v1.5")
        self.assertEqual(embedding._prepare_documents(["article"]), ["article"])
        self.assertIn("searching relevant passages", embedding._prepare_queries(["question"])[0])

    def test_bm25_rejects_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            BM25Index(variant="unknown").build([{"id": "a", "text": "text"}])

    def test_learned_sparse_build_query_and_roundtrip(self):
        chunks = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]
        index = LearnedSparseIndex(encoder=_Encoder([{"1": 2.0}, {"2": 3.0}, {"1": 4.0}]))
        index.build(chunks, batch_size=2)
        self.assertEqual(index.query("query", 1)[0]["id"], "a")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.pkl"
            index.save(str(path))
            loaded = LearnedSparseIndex.load(str(path), encoder=_Encoder([{"2": 2.0}]))
            self.assertEqual(loaded.query("query", 1)[0]["id"], "b")

    def test_hybrid_accepts_generic_sparse_retriever(self):
        hybrid = HybridRetriever(_Retriever(["a", "b"]), sparse=_Retriever(["b", "c"]))
        self.assertEqual(hybrid.retrieve("q", 1)[0]["id"], "b")

    def test_explicit_runs_are_supported(self):
        spec = {
            "schema_version": 1,
            "experiment": {"id": "phase1"},
            "dataset": {"indexes": {"base": {
                "config": "config.yaml", "variant_manifest": "variant.json",
                "testsets": {"original": "testset.jsonl"},
            }}},
            "fixed": {"index": "base", "variant": "original", "partition": "development"},
            "runs": [
                {"retriever": "dense", "reranker": "noop"},
                {"retriever": "sparse", "reranker": "noop"},
            ],
        }
        runs = expand_run_matrix(spec)
        self.assertEqual([row["parameters"]["retriever"] for row in runs], ["dense", "sparse"])


if __name__ == "__main__":
    unittest.main()
