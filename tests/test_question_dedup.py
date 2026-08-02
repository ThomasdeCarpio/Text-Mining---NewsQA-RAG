import unittest

from src.evaluation.question_dedup import (
    DEDUP_APPROVAL_SCHEMA_VERSION,
    DEDUP_SCHEMA_VERSION,
    derive_deduplicated_artifacts,
    normalize_question,
    validate_cluster_decisions,
    validate_human_approval,
)
from src.evaluation.testset import DatasetBuildError


def item(question_id, question, answer, article="article-1", variant="clarified"):
    return {
        "question_id": question_id,
        "source_question_id": question_id,
        "article_key": article,
        "question": question,
        "question_variant": variant,
        "ground_truth": answer,
        "accepted_answers": [answer],
        "evidence": answer,
        "evidence_spans": [{"start": 0, "end": len(answer), "text": answer}],
        "relevant_chunk_ids": [f"{article}_chunk_0"],
    }


class QuestionDedupTests(unittest.TestCase):
    def test_normalization_ignores_case_spacing_and_punctuation(self):
        self.assertEqual(
            normalize_question("Who  Won?"), normalize_question("who won")
        )

    def test_partition_merges_paired_variants_and_answers(self):
        original = [
            item("q-1", "When was her surgery?", "Tuesday", variant="original"),
            item("q-2", "When did it occur?", "on Tuesday", variant="original"),
        ]
        resolved = [
            item("q-1", "When was Alice's surgery?", "Tuesday"),
            item("q-2", "When was Alice's surgery?", "on Tuesday"),
        ]
        decisions = {
            "schema_version": DEDUP_SCHEMA_VERSION,
            "clusters": [
                {
                    "article_id": "article-1",
                    "member_question_ids": ["q-1", "q-2"],
                    "representative_question_id": "q-1",
                    "semantic_target": "Date of Alice's surgery",
                    "rationale": "Both questions ask for the same date.",
                }
            ],
        }
        clusters = validate_cluster_decisions(decisions, resolved)
        dedup_original, dedup_resolved, clarified, removed, records = (
            derive_deduplicated_artifacts(original, resolved, clusters)
        )
        self.assertEqual(1, len(dedup_original))
        self.assertEqual(1, len(dedup_resolved))
        self.assertEqual(["Tuesday", "on Tuesday"], dedup_resolved[0]["accepted_answers"])
        self.assertEqual(["q-1", "q-2"], dedup_resolved[0]["merged_source_question_ids"])
        self.assertEqual(1, len(clarified))
        self.assertEqual("q-2", removed[0]["question_id"])
        self.assertEqual(2, records[0]["member_count"])

    def test_cluster_cannot_cross_articles(self):
        resolved = [
            item("q-1", "Who won?", "Alice"),
            item("q-2", "Who won?", "Bob", "article-2"),
        ]
        decisions = {
            "schema_version": DEDUP_SCHEMA_VERSION,
            "clusters": [
                {
                    "article_id": "article-1",
                    "member_question_ids": ["q-1", "q-2"],
                    "representative_question_id": "q-1",
                    "semantic_target": "Winner",
                    "rationale": "Invalid cross-article proposal.",
                }
            ],
        }
        with self.assertRaisesRegex(DatasetBuildError, "article boundaries"):
            validate_cluster_decisions(decisions, resolved)

    def test_human_approval_requires_exact_cluster_coverage(self):
        resolved = [
            item("q-1", "Who won?", "Alice"),
            item("q-2", "Who was the winner?", "Alice"),
        ]
        decisions = {
            "schema_version": DEDUP_SCHEMA_VERSION,
            "clusters": [
                {
                    "article_id": "article-1",
                    "member_question_ids": ["q-1", "q-2"],
                    "representative_question_id": "q-1",
                    "semantic_target": "Winner",
                    "rationale": "Both ask for the winner.",
                }
            ],
        }
        clusters = validate_cluster_decisions(decisions, resolved)
        approval = {
            "schema_version": DEDUP_APPROVAL_SCHEMA_VERSION,
            "proposal_sha256": "proposal-hash",
            "base_testset_sha256": "testset-hash",
            "human_review": {
                "status": "approved",
                "reviewer_id": "reviewer",
                "reviewed_at": "2026-07-17T00:00:00Z",
            },
            "cluster_reviews": [
                {"cluster_id": clusters[0]["cluster_id"], "decision": "approve"}
            ],
        }
        validate_human_approval(
            approval,
            proposal_sha256="proposal-hash",
            base_testset_sha256="testset-hash",
            clusters=clusters,
        )
        approval["cluster_reviews"] = []
        with self.assertRaisesRegex(DatasetBuildError, "exactly cover"):
            validate_human_approval(
                approval,
                proposal_sha256="proposal-hash",
                base_testset_sha256="testset-hash",
                clusters=clusters,
            )


if __name__ == "__main__":
    unittest.main()
