"""The ablation is two data transforms; these are what must not break.

Both feed unmodified pipeline scripts, so a silent mistake here would look like
a retrieval result rather than a bug.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "retrieval_ablation", PROJECT_ROOT / "scripts" / "run_retrieval_ablation_kaggle.py"
)
ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation)


class ContextualChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rows = [
            {"id": "a_chunk_0", "text": "Storm hits Cocodrie. Body.", "metadata": {"title": "Storm hits Cocodrie"}},
            {"id": "a_chunk_1", "text": "Continuation with no context.", "metadata": {"title": "Storm hits Cocodrie"}},
            {"id": "b_chunk_0", "text": "No context available.", "metadata": {"title": ""}},
        ]
        self.source = self.tmp / "chunks.jsonl"
        self.source.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")

    def test_prepends_only_to_continuation_chunks(self):
        got = ablation.read_jsonl(ablation.contextual_chunks(self.source, self.tmp / "out.jsonl"))
        self.assertEqual(got[0]["text"], "Storm hits Cocodrie. Body.", "chunk 0 already opens with it")
        self.assertEqual(got[1]["text"], "Storm hits Cocodrie\n\nContinuation with no context.")
        self.assertEqual(got[2]["text"], "No context available.", "empty context must be left alone")

    def test_chunk_ids_survive(self):
        # Gold labels reference these IDs and every comparison is paired on them.
        got = ablation.read_jsonl(ablation.contextual_chunks(self.source, self.tmp / "out.jsonl"))
        self.assertEqual([r["id"] for r in got], [r["id"] for r in self.rows])

    def test_idempotent_so_a_resumed_run_never_double_prepends(self):
        target = self.tmp / "out.jsonl"
        first = ablation.read_jsonl(ablation.contextual_chunks(self.source, target))
        second = ablation.read_jsonl(ablation.contextual_chunks(self.source, target))
        self.assertEqual(first, second)


class PromptedTestsetTest(unittest.TestCase):
    def test_rewrites_question_and_keeps_labels(self):
        tmp = Path(tempfile.mkdtemp())
        source = tmp / "testset.jsonl"
        source.write_text(json.dumps(
            {"question_id": "q1", "question": "Who won?", "relevant_chunk_ids": ["a_chunk_0"]}
        ) + "\n", encoding="utf-8")
        got = ablation.read_jsonl(ablation.prompted_testset(source, tmp / "out.jsonl"))
        self.assertEqual(got[0]["question"], "A passage that answers the question: Who won?")
        self.assertEqual(got[0]["relevant_chunk_ids"], ["a_chunk_0"])
        self.assertEqual(got[0]["question_id"], "q1", "IDs must survive so runs stay paired")


if __name__ == "__main__":
    unittest.main()
