import csv
import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.evaluation.question_review import (
    ArticleCandidateIndex,
    build_article_request,
    create_review_queue,
    load_approved_annotations,
    review_status,
    validate_predictions,
)
from src.evaluation.testset import (
    DatasetBuildError,
    SampleSpec,
    article_id_for_context,
    canonical_json,
    derive_chunked_testsets,
    derive_reviewed_testsets,
    load_testset,
    sample_articles,
    save_jsonl,
    sha256_file,
    sha256_text,
)
from scripts.prepare_evaluation_dataset import build_baseline, build_parser, finalize
from scripts.format_review_queue import build_readable_queue
from scripts.run_benchmark import _apply_manifest_preflight


def row(context: str, index: int, answer: str = "answer") -> dict:
    start = context.index(answer)
    return {
        "context": context,
        "question": f"Question {index}?",
        "answers": [answer],
        "key": f"q-{index}",
        "labels": [{"start": [start], "end": [start + len(answer) - 1]}],
    }


class FakeChunker:
    def chunk_article(self, article_data, filename):
        text = article_data["text"]
        midpoint = max(1, len(text) // 2)
        article_id = filename[-12:]
        return [
            {"id": f"{article_id}_chunk_0", "text": text[: midpoint + 3], "metadata": {}},
            {"id": f"{article_id}_chunk_1", "text": text[midpoint:], "metadata": {}},
        ]


class EvaluationDatasetSelectionTests(unittest.TestCase):
    def test_professional_workflow_defaults(self):
        args = build_parser().parse_args(["stage1"])
        self.assertEqual(200, args.evaluation_articles)
        self.assertEqual(800, args.distractor_articles)
        self.assertEqual(42, args.seed)
        self.assertEqual("gemini-3.1-flash-lite", args.model)
        self.assertIn("newsqa_200_1000", args.output_root)
        self.assertIn("newsqa_200_1000.selection.json", args.selection_manifest)

    def test_sampling_is_row_order_independent_and_collects_all_questions(self):
        contexts = [f"Article {i} has answer value." for i in range(6)]
        rows = [row(context, i * 10 + q) for i, context in enumerate(contexts) for q in range(3)]

        def make_factory(seed):
            calls = 0

            def factory(_split):
                nonlocal calls
                calls += 1
                values = list(rows)
                random.Random(seed + calls).shuffle(values)
                return values

            return factory

        spec = SampleSpec("validation", 3, 42, "evaluation")
        first, first_stats = sample_articles(make_factory(1), spec)
        second, second_stats = sample_articles(make_factory(90), spec)
        self.assertEqual(first_stats["selected_article_ids"], second_stats["selected_article_ids"])
        self.assertEqual(3, len(first))
        self.assertTrue(all(len(article["questions"]) == 3 for article in first))
        self.assertEqual(9, first_stats["selected_questions"])

    def test_sampling_fails_when_request_exceeds_unique_articles(self):
        values = [row("Only article with answer.", 1)]
        with self.assertRaisesRegex(DatasetBuildError, "only 1"):
            sample_articles(
                lambda _split: values,
                SampleSpec("validation", 2, 42, "evaluation"),
            )

    def test_derived_rows_share_chunks_across_original_and_clarified(self):
        context = "Case Alpha has answer near the center of this article."
        article_id = article_id_for_context(context)
        question = row(context, 1)
        start = context.index("answer")
        article = {
            "article_id": article_id,
            "context": context,
            "split": "validation",
            "role": "evaluation",
            "metadata": {"title": "Case Alpha", "publisher": "CNN"},
            "questions": [
                {
                    "question_id": question["key"],
                    "article_id": article_id,
                    "question": question["question"],
                    "ground_truth": "answer",
                    "evidence_spans": [{"start": start, "end": start + 6, "text": "answer"}],
                }
            ],
        }
        annotations = {
            "q-1": {
                "final_label": "human_non_standalone",
                "reason_codes": ["missing_subject"],
                "final_clarified_question": "What was the answer in Case Alpha?",
            }
        }
        original, clarified, chunks = derive_chunked_testsets([article], [], FakeChunker(), annotations)
        self.assertEqual(1, len(original))
        self.assertEqual(1, len(clarified))
        self.assertEqual(original[0]["relevant_chunk_ids"], clarified[0]["relevant_chunk_ids"])
        self.assertTrue(set(original[0]["relevant_chunk_ids"]) <= {item["id"] for item in chunks})

    def test_reviewed_variants_preserve_count_evidence_and_chunk_mapping(self):
        original = [
            {
                "question_id": "q-1",
                "question": "Who was acquitted?",
                "ground_truth": "Alice",
                "evidence_spans": [{"start": 0, "end": 5, "text": "Alice"}],
                "relevant_chunk_ids": ["chunk-1"],
                "standalone_label": "unreviewed",
                "ambiguity_reasons": [],
            },
            {
                "question_id": "q-2",
                "question": "When was the hearing?",
                "ground_truth": "Monday",
                "evidence_spans": [{"start": 6, "end": 12, "text": "Monday"}],
                "relevant_chunk_ids": ["chunk-2"],
                "standalone_label": "unreviewed",
                "ambiguity_reasons": [],
            },
        ]
        annotations = {
            "q-1": {
                "final_label": "human_non_standalone",
                "reason_codes": ["missing_subject"],
                "final_clarified_question": "Who was acquitted in Case Alpha?",
            },
            "q-2": {
                "final_label": "human_standalone",
                "reason_codes": [],
                "final_clarified_question": None,
            },
        }

        clarified, resolved = derive_reviewed_testsets(original, annotations)

        self.assertEqual(1, len(clarified))
        self.assertEqual(2, len(resolved))
        self.assertEqual("q-1::clarified", clarified[0]["question_id"])
        self.assertEqual(original[0]["ground_truth"], clarified[0]["ground_truth"])
        self.assertEqual(original[0]["relevant_chunk_ids"], clarified[0]["relevant_chunk_ids"])
        self.assertEqual("original", resolved[1]["question_variant"])

    def test_baseline_is_available_before_review_and_finalize_reuses_it(self):
        context = "Case Alpha involved several people. Alice was acquitted on Monday."
        answer_start = context.index("Alice")
        article_id = article_id_for_context(context)
        question = {
            "question_id": "q-ambiguous",
            "article_id": article_id,
            "question": "Who was acquitted?",
            "ground_truth": "Alice",
            "evidence_spans": [
                {"start": answer_start, "end": answer_start + 5, "text": "Alice"}
            ],
        }
        article = {
            "article_id": article_id,
            "context": context,
            "split": "validation",
            "role": "evaluation",
            "metadata": {"title": "Case Alpha", "publisher": "CNN"},
            "questions": [question],
        }
        prediction = {
            "question_id": "q-ambiguous",
            "standalone_label": "non_standalone",
            "reason_codes": ["missing_subject"],
            "rationale": "The case is not named.",
            "candidate_clarified_question": "Who was acquitted in Case Alpha?",
            "added_context": [
                {"text": "Case Alpha", "supporting_context_quote": "Case Alpha"}
            ],
            "confidence": 0.95,
            "validation_warnings": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_path = root / "staging/corpus/evaluation_articles.jsonl"
            distractor_path = root / "staging/corpus/distractor_articles.jsonl"
            predictions_path = root / "staging/triage/all_predictions.jsonl"
            review_jsonl = root / "staging/review/review_queue.jsonl"
            review_csv = root / "staging/review/review_queue.csv"
            review_readable = root / "staging/review/review_queue_readable.json"
            config_path = root / "config.yaml"
            selection_manifest = root / "selection.json"
            variant_manifest = root / "variant.json"
            save_jsonl([article], evaluation_path)
            save_jsonl([], distractor_path)
            config_path.write_text("chunking: {strategy: recursive}\nembedding: {}\n", encoding="utf-8")
            selection_manifest.write_text(
                json.dumps({"sampling": {"seed": 42}}), encoding="utf-8"
            )
            common = {
                "output_root": str(root),
                "selection_manifest": str(selection_manifest),
                "variant_manifest": str(variant_manifest),
                "config": str(config_path),
                "collection": None,
            }
            baseline_args = Namespace(
                **common,
                db_path=str(root / "chroma"),
                overwrite=False,
                skip_index=True,
            )
            with patch(
                "scripts.prepare_evaluation_dataset.get_chunker", return_value=FakeChunker()
            ):
                build_baseline(baseline_args)

            original_path = root / "final/testset_original.jsonl"
            chunks_path = root / "final/chunks.jsonl"
            self.assertTrue(original_path.exists())
            self.assertEqual("baseline_ready", json.loads(variant_manifest.read_text())["status"])
            original_hash = sha256_file(original_path)
            chunks_hash = sha256_file(chunks_path)

            save_jsonl([{"article_id": article_id, "predictions": [prediction]}], predictions_path)
            queue = create_review_queue(
                [article],
                [{"article_id": article_id, "predictions": [prediction]}],
                review_jsonl,
                review_csv,
            )
            with open(review_csv, encoding="utf-8-sig", newline="") as handle:
                review_rows = list(csv.DictReader(handle))
                fields = review_rows[0].keys()
            review_rows[0]["review_decision"] = "approve"
            review_rows[0]["reviewer_id"] = "reviewer-1"
            with open(review_csv, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(review_rows)
            review_document = build_readable_queue(
                queue,
                {"q-ambiguous": review_rows[0]},
                {"q-ambiguous": question},
            )
            self.assertEqual(
                prediction["candidate_clarified_question"],
                review_document["articles"][0]["questions"][0]["comparison"][
                    "reviewed_candidate_clarified_question"
                ],
            )
            review_readable.write_text(
                json.dumps(review_document), encoding="utf-8"
            )

            finalize(Namespace(**common))

            self.assertEqual(original_hash, sha256_file(original_path))
            self.assertEqual(chunks_hash, sha256_file(chunks_path))
            self.assertEqual(1, len(load_testset(root / "final/testset_resolved.jsonl")))
            self.assertEqual(1, len(load_testset(root / "final/testset_clarified.jsonl")))
            self.assertEqual("review_complete", json.loads(variant_manifest.read_text())["status"])

    def test_manifest_preflight_accepts_resolved_testset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolved = root / "testset_resolved.jsonl"
            chunks = root / "chunks.jsonl"
            manifest_path = root / "variant.json"
            save_jsonl([{"question": "Resolved question"}], resolved)
            save_jsonl([{"id": "chunk-1"}], chunks)
            config = {"chunking": {"strategy": "recursive"}}
            manifest_path.write_text(
                json.dumps(
                    {
                        "pipeline": {"config_sha256": sha256_text(canonical_json(config))},
                        "database": {
                            "indexed": True,
                            "path": str(root / "chroma"),
                            "collection": "shared-newsqa",
                            "chunk_count": 1,
                        },
                        "artifacts": {
                            "testset_resolved": {
                                "path": str(resolved),
                                "sha256": sha256_file(resolved),
                            },
                            "chunks": {
                                "path": str(chunks),
                                "sha256": sha256_file(chunks),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                variant_manifest=str(manifest_path),
                testset=str(resolved),
                collection=None,
                db_path=None,
                chunks_path=None,
                bm25_path=None,
            )

            _apply_manifest_preflight(args, config)

            self.assertEqual("shared-newsqa", args.collection)
            self.assertEqual(str(chunks), args.chunks_path)


class QuestionReviewTests(unittest.TestCase):
    def setUp(self):
        self.context = "Case Alpha involved several people. Alice was acquitted on Monday."
        answer_start = self.context.index("Alice")
        self.article_id = article_id_for_context(self.context)
        self.question = {
            "question_id": "q-ambiguous",
            "article_id": self.article_id,
            "question": "Who was acquitted?",
            "ground_truth": "Alice",
            "evidence_spans": [
                {"start": answer_start, "end": answer_start + 5, "text": "Alice"}
            ],
        }
        self.article = {
            "article_id": self.article_id,
            "context": self.context,
            "split": "validation",
            "role": "evaluation",
            "metadata": {"title": "Case Alpha", "publisher": "CNN"},
            "questions": [self.question],
        }
        self.prediction = {
            "question_id": "q-ambiguous",
            "standalone_label": "non_standalone",
            "reason_codes": ["missing_subject"],
            "rationale": "The case is not named.",
            "candidate_clarified_question": "Who was acquitted in Case Alpha?",
            "added_context": [
                {"text": "Case Alpha", "supporting_context_quote": "Case Alpha"}
            ],
            "confidence": 0.95,
        }

    def test_prediction_validation_routes_answer_leakage_to_review(self):
        leaking = {**self.prediction, "candidate_clarified_question": "Was Alice acquitted?"}
        validated = validate_predictions(self.article, [leaking])[0]
        self.assertEqual("uncertain", validated["standalone_label"])
        self.assertIn(
            "candidate_clarification_contains_answer", validated["validation_warnings"]
        )

    def test_article_request_exposes_answer_only_as_explicit_constraint(self):
        request = build_article_request(
            self.article,
            ArticleCandidateIndex([self.article]),
        )
        question = request["questions"][0]
        self.assertEqual("Alice", question["expected_answer"])
        self.assertNotIn("Alice", question["redacted_source_context"])
        self.assertNotIn("title", request["article"])

    def test_review_gate_requires_approval_and_reviewer(self):
        prediction = validate_predictions(self.article, [self.prediction])[0]
        records = [{"article_id": self.article_id, "predictions": [prediction]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "review.csv"
            create_review_queue([self.article], records, root / "review.jsonl", csv_path)
            self.assertFalse(review_status(csv_path)["ready"])
            with self.assertRaisesRegex(DatasetBuildError, "incomplete"):
                load_approved_annotations(csv_path, [self.article], records)

            with open(csv_path, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = rows[0].keys()
            rows[0]["review_decision"] = "approve"
            rows[0]["reviewer_id"] = "reviewer-1"
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            annotations = load_approved_annotations(csv_path, [self.article], records)
            self.assertEqual("human_non_standalone", annotations["q-ambiguous"]["final_label"])
            self.assertEqual(
                "Who was acquitted in Case Alpha?",
                annotations["q-ambiguous"]["final_clarified_question"],
            )
            create_review_queue([self.article], records, root / "review.jsonl", csv_path)
            self.assertTrue(review_status(csv_path)["ready"])

    def test_hierarchical_review_applies_answer_correction(self):
        source_question = {
            **self.question,
            "ground_truth": "lice",
            "evidence_spans": [
                {
                    "start": self.question["evidence_spans"][0]["start"] + 1,
                    "end": self.question["evidence_spans"][0]["end"],
                    "text": "lice",
                }
            ],
        }
        article = {**self.article, "questions": [source_question]}
        prediction = validate_predictions(article, [self.prediction])[0]
        document = {
            "schema_version": "2.0",
            "articles": [
                {
                    "article_id": self.article_id,
                    "questions": [
                        {
                            "question_id": source_question["question_id"],
                            "comparison": {
                                "original_question": source_question["question"],
                                "candidate_clarified_question": prediction[
                                    "candidate_clarified_question"
                                ],
                                "final_clarified_question": prediction[
                                    "candidate_clarified_question"
                                ],
                            },
                            "answer_and_evidence": {
                                "source_expected_answer": "lice",
                                "expected_answer": "Alice",
                                "accepted_answers": ["Alice"],
                                "source_evidence_spans": source_question[
                                    "evidence_spans"
                                ],
                                "evidence_spans": self.question["evidence_spans"],
                                "evidence_text": "Alice",
                                "answer_modified": True,
                                "answer_review_notes": "Expanded a truncated name.",
                            },
                            "llm_assessment": {
                                "label": prediction["standalone_label"],
                                "reason_codes": prediction["reason_codes"],
                                "validation_warnings": prediction["validation_warnings"],
                            },
                            "human_review": {
                                "decision": "edit",
                                "reviewer_id": "reviewer-1",
                                "supporting_quotes": ["Case Alpha"],
                                "notes": "Clarified and corrected.",
                            },
                        }
                    ],
                }
            ],
        }
        records = [{"article_id": self.article_id, "predictions": [prediction]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            annotations = load_approved_annotations(path, [article], records)

        annotation = annotations[source_question["question_id"]]
        self.assertTrue(annotation["answer_modified"])
        self.assertEqual("Alice", annotation["ground_truth"])
        self.assertEqual(self.question["evidence_spans"], annotation["evidence_spans"])


if __name__ == "__main__":
    unittest.main()
