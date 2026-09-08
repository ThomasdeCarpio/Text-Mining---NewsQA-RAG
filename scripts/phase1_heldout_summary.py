#!/usr/bin/env python3
"""Summarise the Phase 1 final-test run and place it beside the development set.

Phase 1 locked a retrieval configuration on 50 development articles, then ran it
once on the 150 articles it had never touched. Those two partitions are disjoint,
so there is no paired difference to take here -- what the report needs instead is
each partition's own interval, wide enough to say whether the drop is real.

Two facts make this worth a script rather than a pair of averages. Questions from
one article are not independent draws, so the interval resamples whole articles.
And the Phase 2 held-out set turns out to be a subset of this one, which lets the
same 284 questions be scored by two independent runs -- a consistency check the
project gets for free and should not skip.

    python scripts/phase1_heldout_summary.py

Writes docs/reports/phase1/heldout/heldout_significance.json.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

BOOTSTRAP_SAMPLES = 2000
SEED = 42
METRICS = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "mrr@5", "ndcg@5", "recall@5"]

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "docs/reports/phase1/heldout"
# Retrieval is frozen across every Phase 2 run, so its development score file
# carries the same locked retrieval this run reproduces on the other partition.
DEVELOPMENT = ROOT / "docs/reports/phase2/scores/p2_d5_development.jsonl"
PHASE2_HELDOUT = ROOT / "docs/reports/phase2/scores/p2_d5_heldout.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cluster_interval(rows: list[dict], metric: str) -> dict:
    """Mean of the metric, with a 95% percentile bootstrap over articles.

    Resampling articles rather than questions is what keeps the interval honest:
    a dozen questions can share one article, so treating them as independent
    draws would understate the spread.
    """
    per_article: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_article[row["article_key"]].append(row["retrieval"][metric])
    flat = [value for values in per_article.values() for value in values]
    observed = sum(flat) / len(flat)

    keys = list(per_article)
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        drawn = [v for _ in keys for v in per_article[rng.choice(keys)]]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    return {
        "value": round(observed, 6),
        "ci95_low": round(means[int(0.025 * BOOTSTRAP_SAMPLES)], 6),
        "ci95_high": round(means[int(0.975 * BOOTSTRAP_SAMPLES) - 1], 6),
        "se_bootstrap": round(statistics.pstdev(means), 6),
        "n_questions": len(flat),
        "n_articles": len(keys),
    }


def rerank_gain(rows: list[dict], metric: str) -> dict:
    """How much the cross-encoder moved the metric, paired per question."""
    per_article: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_article[row["article_key"]].append(row["retrieval"][metric] - row["retrieval_initial"][metric])
    flat = [value for values in per_article.values() for value in values]
    keys = list(per_article)
    rng = random.Random(SEED)
    means = sorted(sum(v for _ in keys for v in per_article[rng.choice(keys)]) / len(flat)
                   for _ in range(BOOTSTRAP_SAMPLES))
    return {
        "delta": round(sum(flat) / len(flat), 6),
        "ci95_low": round(means[int(0.025 * BOOTSTRAP_SAMPLES)], 6),
        "ci95_high": round(means[int(0.975 * BOOTSTRAP_SAMPLES) - 1], 6),
    }


def main() -> None:
    heldout = read(HELDOUT / "heldout_question_scores.jsonl")
    development = read(DEVELOPMENT)
    phase2 = read(PHASE2_HELDOUT)

    # The run wrote its own micro averages. Recomputing them from the per-question
    # file and checking they agree is the cheapest guard against reading the wrong
    # column, so do it before anything downstream trusts these numbers.
    reported = json.loads((HELDOUT / "final_comparison.json").read_text(encoding="utf-8"))["runs"][0]
    for metric in METRICS:
        # The comparison file stores flat dotted keys, rounded to four places.
        expected = reported[f"retrieval.{metric}"]
        actual = sum(row["retrieval"][metric] for row in heldout) / len(heldout)
        assert abs(expected - actual) < 1e-4, f"{metric}: run says {expected}, scores say {actual}"

    # Phase 2 drew its held-out articles from inside this partition, so the same
    # questions were scored twice by two independent runs.
    phase2_ids = {row["question_id"] for row in phase2}
    heldout_ids = {row["question_id"] for row in heldout}
    shared = phase2_ids & heldout_ids
    subset = [row for row in heldout if row["question_id"] in shared]
    remainder = [row for row in heldout if row["question_id"] not in shared]

    payload = {
        "schema_version": 1,
        "method": "percentile bootstrap over articles, per-question retrieval scores",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "note": ("Development and final-test hold disjoint articles, so these are "
                 "intervals on each partition, not a paired difference."),
        "partitions": {
            "development": {metric: cluster_interval(development, metric) for metric in METRICS},
            "final_test": {metric: cluster_interval(heldout, metric) for metric in METRICS},
        },
        # The run's own intervals resample questions. Keeping both says what the
        # article clustering costs instead of asserting it.
        "final_test_question_bootstrap_from_run": {
            metric: {"ci95_low": reported[f"retrieval.{metric}.ci95_low"],
                     "ci95_high": reported[f"retrieval.{metric}.ci95_high"]}
            for metric in METRICS if f"retrieval.{metric}.ci95_low" in reported},
        "reranker_gain_final_test": {metric: rerank_gain(heldout, metric)
                                     for metric in ("mrr@5", "ndcg@5", "ndcg@1")},
        "evidence_missing_at_5": {
            "development": {"n": sum(1 for r in development if r["retrieval"]["hit_rate@5"] == 0),
                            "of": len(development)},
            "final_test": {"n": sum(1 for r in heldout if r["retrieval"]["hit_rate@5"] == 0),
                           "of": len(heldout)},
        },
        "phase2_heldout_overlap": {
            "shared_questions": len(shared),
            "phase2_heldout_questions": len(phase2_ids),
            "is_subset": shared == phase2_ids,
            "phase2_articles_inside": len({r["article_key"] for r in subset}),
            "scored_twice_hit_rate@5": {
                "phase1_run": round(sum(r["retrieval"]["hit_rate@5"] for r in subset) / len(subset), 6),
                "phase2_run": round(sum(r["retrieval"]["hit_rate@5"] for r in phase2) / len(phase2), 6),
            },
            "same_50_articles": {m: cluster_interval(subset, m) for m in ("hit_rate@5", "ndcg@5")},
            "other_100_articles": {m: cluster_interval(remainder, m) for m in ("hit_rate@5", "ndcg@5")},
        },
    }

    print(f"development {len(development)} questions / {len({r['article_key'] for r in development})} articles")
    print(f"final-test  {len(heldout)} questions / {len({r['article_key'] for r in heldout})} articles\n")
    print(f"{'metric':12} {'development':>12} {'final-test':>12} {'delta':>9}   {'CI95 final-test':>22}")
    for metric in METRICS:
        dev = payload["partitions"]["development"][metric]
        fin = payload["partitions"]["final_test"][metric]
        print(f"{metric:12} {dev['value']:12.4f} {fin['value']:12.4f} "
              f"{fin['value'] - dev['value']:+9.4f}   [{fin['ci95_low']:.4f}; {fin['ci95_high']:.4f}]")

    overlap = payload["phase2_heldout_overlap"]
    twice = overlap["scored_twice_hit_rate@5"]
    print(f"\nPhase 2 held-out is a subset: {overlap['is_subset']} "
          f"({overlap['shared_questions']}/{overlap['phase2_heldout_questions']} questions, "
          f"{overlap['phase2_articles_inside']} articles)")
    print(f"   same questions, two runs -- Hit@5 {twice['phase1_run']:.4f} vs {twice['phase2_run']:.4f}")

    output = HELDOUT / "heldout_significance.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
