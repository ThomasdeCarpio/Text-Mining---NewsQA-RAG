import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from newsqa_rag.evaluation.cloud_dataset import (
    CANONICAL_ARTIFACTS,
    export_canonical_bundle,
    index_fingerprint,
    materialize_canonical_source,
    rebind_dedup_approval,
    verify_canonical_bundle,
)
from newsqa_rag.evaluation.question_review import create_full_review_document
from newsqa_rag.evaluation.testset import (
    DatasetBuildError,
    article_id_for_context,
    save_jsonl,
    sha256_file,
)
from scripts.materialize_evaluation_dataset import materialize


class CloudEvaluationDatasetTests(unittest.TestCase):
    def _source_fixture(self, root: Path) -> tuple[Path, Path]:
        project = root / "project"
        source = project / "data/evaluation/newsqa_200_11064"
        context = "Case Alpha reports that Alice won the hearing on Monday."
        article_id = article_id_for_context(context)
        start = context.index("Alice")
        question = {
            "question_id": "q-1",
            "article_id": article_id,
            "question": "Who won the hearing in Case Alpha?",
            "ground_truth": "Alice",
            "evidence_spans": [{"start": start, "end": start + 5, "text": "Alice"}],
        }
        article = {
            "article_id": article_id,
            "context": context,
            "context_sha256": "fixture",
            "normalized_context_sha256": "fixture",
            "split": "validation",
            "role": "evaluation",
            "metadata": {"title": "Case Alpha", "publisher": "CNN"},
            "questions": [question],
            "source_question_count": 1,
        }
        save_jsonl([article], source / "staging/corpus/evaluation_articles.jsonl")
        save_jsonl([], source / "staging/corpus/distractor_articles.jsonl")
        save_jsonl([question], source / "staging/questions/original_questions.jsonl")

        document = create_full_review_document([article])
        review = document["articles"][0]["questions"][0]
        review["codex_assessment"].update(
            {
                "label": "standalone",
                "issue_codes": [],
                "rationale": "The subject and event are explicit.",
            }
        )
        review["codex_assessment"]["proposal"].update(
            {
                "status": "proposed",
                "batch_id": "fixture",
                "created_at": "2026-08-08T00:00:00Z",
            }
        )
        review["human_review"].update(
            {"decision": "mark_standalone", "reviewer_id": "reviewer"}
        )
        review_path = source / "staging/review/review_queue_readable.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps(document), encoding="utf-8")
        (source / "staging/review/manifest.json").write_text("{}\n", encoding="utf-8")
        schema = source / "staging/review/audits/schema.json"
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text("{}\n", encoding="utf-8")

        resolved = {
            "question_id": "q-1",
            "source_question_id": "q-1",
            "article_key": article_id,
            "question": question["question"],
            "ground_truth": "Alice",
            "accepted_answers": ["Alice"],
            "evidence_spans": question["evidence_spans"],
            "relevant_chunk_ids": ["old_chunk"],
        }
        resolved_path = source / "final/testset_resolved.jsonl"
        save_jsonl([resolved], resolved_path)
        decisions = {
            "schema_version": "1.0",
            "base_testset_sha256": sha256_file(resolved_path),
            "clusters": [],
        }
        decisions_path = (
            project
            / "evaluation/question_dedup/newsqa_200_11064.semantic_clusters.json"
        )
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
        approval = {
            "schema_version": "1.0",
            "proposal_sha256": sha256_file(decisions_path),
            "base_testset_sha256": sha256_file(resolved_path),
            "cluster_reviews": [],
            "human_review": {
                "status": "approved",
                "reviewer_id": "reviewer",
                "reviewed_at": "2026-08-08T00:00:00Z",
            },
        }
        approval_path = (
            project / "evaluation/question_dedup/newsqa_200_11064.human_approval.json"
        )
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        selection = project / "evaluation/manifests/newsqa_200_11064.selection.json"
        selection.parent.mkdir(parents=True, exist_ok=True)
        selection.write_text(
            json.dumps(
                {
                    "dataset": {"name": "fixture", "revision": "source-revision"},
                    "sampling": {"seed": 42},
                }
            ),
            encoding="utf-8",
        )
        return project, source

    def test_export_verify_and_materialize_are_chunk_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, source = self._source_fixture(root)
            bundle = root / "bundle"
            manifest = export_canonical_bundle(
                project_root=project,
                source_root=source,
                output_dir=bundle,
                version="v1.0.0",
            )

            self.assertEqual(set(CANONICAL_ARTIFACTS), set(manifest["artifacts"]))
            self.assertFalse(any("chunk" in item["path"] for item in manifest["artifacts"].values()))
            self.assertEqual(manifest, verify_canonical_bundle(bundle))

            output = root / "downloaded"
            paths = materialize_canonical_source(bundle, output)
            self.assertTrue(paths["review_queue"].exists())
            self.assertTrue(paths["dedup_approval"].exists())
            self.assertEqual(
                manifest["dataset_sha256"],
                json.loads(paths["source_manifest"].read_text())["dataset_sha256"],
            )

    def test_verification_rejects_tampered_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, source = self._source_fixture(root)
            bundle = root / "bundle"
            export_canonical_bundle(
                project_root=project,
                source_root=source,
                output_dir=bundle,
                version="v1.0.0",
            )
            with (bundle / "questions/original_questions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(DatasetBuildError, "integrity check"):
                verify_canonical_bundle(bundle)

    def test_rebind_preserves_review_and_targets_new_chunk_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, source = self._source_fixture(root)
            decisions = project / "evaluation/question_dedup/newsqa_200_11064.semantic_clusters.json"
            approval = project / "evaluation/question_dedup/newsqa_200_11064.human_approval.json"
            new_resolved = root / "new_resolved.jsonl"
            row = json.loads((source / "final/testset_resolved.jsonl").read_text())
            row["relevant_chunk_ids"] = ["new_chunk"]
            save_jsonl([row], new_resolved)

            rebound_decisions, rebound_approval = rebind_dedup_approval(
                decisions_path=decisions,
                approval_path=approval,
                resolved_testset=new_resolved,
                output_dir=root / "rebound",
            )
            rebound_decision_value = json.loads(rebound_decisions.read_text())
            rebound_approval_value = json.loads(rebound_approval.read_text())
            self.assertEqual(sha256_file(new_resolved), rebound_decision_value["base_testset_sha256"])
            self.assertEqual(sha256_file(rebound_decisions), rebound_approval_value["proposal_sha256"])
            self.assertEqual("reviewer", rebound_approval_value["human_review"]["reviewer_id"])

    def test_index_fingerprint_tracks_only_index_inputs(self):
        base = {
            "chunking": {"chunk_size": 512, "chunk_overlap": 64},
            "embedding": {"model_name": "model-a", "dimensions": 384},
            "database": {"hnsw": {"space": "cosine"}},
            "llm": {"model": "generator-a"},
        }
        first = index_fingerprint(dataset_sha256="data", dataset_commit="commit", config=base)
        generator_changed = index_fingerprint(
            dataset_sha256="data",
            dataset_commit="commit",
            config={**base, "llm": {"model": "generator-b"}},
        )
        chunking_changed = index_fingerprint(
            dataset_sha256="data",
            dataset_commit="commit",
            config={**base, "chunking": {"chunk_size": 256, "chunk_overlap": 32}},
        )
        self.assertEqual(first, generator_changed)
        self.assertNotEqual(first, chunking_changed)

    def test_offline_materializer_builds_reviewed_testsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, source = self._source_fixture(root)
            bundle = root / "bundle"
            export_canonical_bundle(
                project_root=project,
                source_root=source,
                output_dir=bundle,
                version="v1.0.0",
            )
            config = root / "config.yaml"
            config.write_text(
                "chunking:\n  strategy: recursive\n  chunk_size: 64\n  chunk_overlap: 8\n"
                "embedding: {}\ndatabase: {}\n",
                encoding="utf-8",
            )
            output = root / "materialized"
            def fake_run(command):
                if "build-baseline" in command:
                    (output / "final").mkdir(parents=True, exist_ok=True)
                    save_jsonl([{"question_id": "q-1"}], output / "final/testset_resolved.jsonl")
                    save_jsonl([{"id": "chunk-1"}], output / "final/chunks.jsonl")
                    (output / "manifests/variant.json").write_text("{}\n", encoding="utf-8")

            with patch("scripts.materialize_evaluation_dataset._run", side_effect=fake_run):
                result = materialize(
                    Namespace(
                        repo_id=None,
                        revision="v1.0.0",
                        output_root=str(output),
                        config=str(config),
                        db_path=str(root / "chroma"),
                        cache_dir=None,
                        token=None,
                        local_bundle=str(bundle),
                        skip_index=True,
                        deduplicate=False,
                        overwrite=False,
                    )
                )
            self.assertFalse(result["resumed"])
            self.assertTrue((output / "final/testset_resolved.jsonl").exists())
            variant = json.loads((output / "manifests/variant.json").read_text())
            self.assertEqual(result["index_fingerprint"], variant["cloud_source"]["index_fingerprint"])
            with patch("scripts.materialize_evaluation_dataset._run") as resumed_run:
                resumed = materialize(
                    Namespace(
                        repo_id=None,
                        revision="v1.0.0",
                        output_root=str(output),
                        config=str(config),
                        db_path=str(root / "chroma"),
                        cache_dir=None,
                        token=None,
                        local_bundle=str(bundle),
                        skip_index=True,
                        deduplicate=False,
                        overwrite=False,
                    )
                )
            self.assertTrue(resumed["resumed"])
            resumed_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
