"""Focused tests for resumable benchmark traces and scoring contracts."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from newsqa_rag.agents.rag_agent import RAGAgent
from newsqa_rag.evaluation.benchmark_io import (
    apply_manifest_preflight,
    append_jsonl,
    latest_by_question,
    load_jsonl,
    run_with_retries,
)
from newsqa_rag.evaluation.metrics import evaluate_citations, evaluate_qa, recall_at_k
from newsqa_rag.retrieval.reranker import (
    BGESequenceClassificationReranker,
    CrossEncoderReranker,
    NoOpReranker,
    get_reranker,
)


class _Retriever:
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        return [
            {"id": "a", "text": "First context.", "score": 0.2, "metadata": {}},
            {"id": "b", "text": "Second context.", "score": 0.1, "metadata": {}},
        ][:top_k]


class _LLM:
    def generate_rag_answer(self, question: str, contexts: list[str]) -> str:
        return "The answer is supported [2], while [9] is invalid."


class _CrossEncoderModel:
    def predict(self, pairs, batch_size=None):
        return [0.1, 0.9]


class BenchmarkTraceTests(unittest.TestCase):
    def test_agent_reuses_trace_and_maps_numbered_citations(self):
        agent = RAGAgent(_Retriever(), NoOpReranker(), _LLM(), top_k=2, rerank_top_n=2)
        trace = agent.retrieve_and_rerank("Question?")
        result = agent.generate_from_trace(trace)

        self.assertEqual(result["citation_indices"], [2])
        self.assertEqual(result["citation_chunk_ids"], ["b"])
        self.assertEqual(result["invalid_citation_indices"], [9])
        self.assertEqual(result["retrieved_chunks"], trace["retrieved_chunks"])

    def test_cross_encoder_preserves_retrieval_score_and_reorders(self):
        reranker = CrossEncoderReranker("test-model")
        reranker._model = _CrossEncoderModel()
        results = _Retriever().retrieve("Question?", 2)

        reranked = reranker.rerank("Question?", results, 2)

        self.assertEqual([item["id"] for item in reranked], ["b", "a"])
        self.assertEqual(reranked[0]["retrieval_score"], 0.1)
        self.assertEqual(reranked[0]["reranker_score"], 0.9)

    def test_bge_reranker_uses_native_transformers_backend(self):
        reranker = get_reranker({
            "retrieval": {
                "reranker": {
                    "type": "cross-encoder",
                    "model": "BAAI/bge-reranker-large",
                    "batch_size": 32,
                }
            }
        })

        self.assertIsInstance(reranker, BGESequenceClassificationReranker)
        self.assertEqual(reranker.batch_size, 8)

    def test_bge_reranker_reports_native_backend(self):
        reranker = BGESequenceClassificationReranker("BAAI/bge-reranker-large")
        self.assertEqual(
            reranker.get_info()["backend"],
            "transformers-sequence-classification",
        )


class BenchmarkMetricTests(unittest.TestCase):
    def test_qa_uses_best_accepted_answer_and_ignores_citation_markers(self):
        result = evaluate_qa(
            [{
                "prediction": "35,000 Canadian troops [1]",
                "ground_truth": "troops",
                "accepted_answers": ["troops", "35,000 Canadian troops"],
            }]
        )
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_citation_metrics_compare_cited_and_gold_chunk_ids(self):
        result = evaluate_citations(
            [{
                "citation_chunk_ids": ["gold", "noise"],
                "invalid_citation_indices": [7],
                "relevant_chunk_ids": ["gold"],
            }]
        )
        self.assertEqual(result["citation_validity"], 0.6667)
        self.assertEqual(result["citation_precision"], 0.5)
        self.assertEqual(result["citation_recall"], 1.0)

    def test_recall_does_not_double_count_duplicate_retrieval_ids(self):
        self.assertEqual(recall_at_k(["a", "b"], ["a", "a"], 2), 0.5)


class BenchmarkCacheTests(unittest.TestCase):
    def test_sparse_preflight_requires_sparse_artifact_not_chroma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            testset = root / "testset.jsonl"
            chunks = root / "chunks.jsonl"
            sparse = root / "bm25.pkl"
            config = {"retrieval": {"sparse": {"method": "bm25"}}}
            testset.write_text('{}\n', encoding="utf-8")
            chunks.write_text('{}\n', encoding="utf-8")
            sparse.write_bytes(b"index")
            from newsqa_rag.evaluation.testset import sha256_file
            from newsqa_rag.evaluation.benchmark_io import stable_hash

            manifest = root / "variant.json"
            manifest.write_text(json.dumps({
                "pipeline": {"config_sha256": stable_hash(config)},
                "database": {"indexed": False, "path": "unused", "collection": "unused"},
                "artifacts": {
                    "testset_original": {"path": str(testset), "sha256": sha256_file(testset)},
                    "chunks": {"path": str(chunks), "sha256": sha256_file(chunks)},
                    "bm25": {"path": str(sparse), "sha256": sha256_file(sparse)},
                },
            }), encoding="utf-8")
            args = SimpleNamespace(
                retriever="sparse", variant_manifest=str(manifest), testset=str(testset),
                collection=None, db_path=None, chunks_path=None, bm25_path=None,
            )

            apply_manifest_preflight(args, config, project_root=root)

            self.assertIsNone(args.db_path)
            self.assertEqual(args.bm25_path, str(sparse))

    def test_retry_records_attempts_and_reuses_transient_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts = Path(directory) / "attempts.jsonl"
            calls = 0

            class TemporaryError(Exception):
                status_code = 429

            def operation():
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise TemporaryError("slow down")
                return "ok"

            result, error, count = run_with_retries(
                operation,
                stage="generation",
                question_id="q1",
                attempts_path=attempts,
                max_attempts=3,
                sleep=lambda _: None,
            )

            self.assertEqual((result, error, count), ("ok", None, 2))
            self.assertEqual(len(load_jsonl(attempts)), 2)

    def test_only_a_malformed_final_line_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            append_jsonl(path, {"question_id": "q1", "status": "success"})
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"question_id":')

            records = load_jsonl(path, recover_final_line=True)

            self.assertEqual(records[0]["question_id"], "q1")
            self.assertEqual(len(list(Path(directory).glob("*.partial-*"))), 1)

    def test_duplicate_success_is_rejected(self):
        records = [
            {"question_id": "q1", "status": "success"},
            {"question_id": "q1", "status": "success"},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate successful"):
            latest_by_question(records)


if __name__ == "__main__":
    unittest.main()
