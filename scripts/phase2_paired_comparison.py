#!/usr/bin/env python3
"""Decide the Phase 2B winner from the per-question scores, by the registered rules.

The tuning plan (docs/Detailed Test Plans/phase_2_generation_tuning_plan.md, s7)
fixes both halves of the decision before any result is seen: four guardrails a
configuration must clear to be eligible at all, and only then the highest Answer
Correctness. It also requires every comparison to be a paired per-question
difference with a 95% article-cluster bootstrap CI, because questions from one
article are not independent observations.

This applies both. It reads the deterministic score files the runs emit, checks
each candidate against P0, and reports the paired CIs. It does not pick a winner
by eyeballing a scalar.

    python scripts/phase2_paired_comparison.py \
        --scores-dir docs/reports/phase2/scores \
        --output docs/reports/phase2/paired_significance.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

BOOTSTRAP_SAMPLES = 2000
SEED = 42

# The metric each guardrail watches, and how far it may fall below P0.
GUARDRAILS = [
    ("ragas.faithfulness", "Faithfulness", 0.02),
    ("citations.citation_f1", "Citation F1", 0.01),
    ("citations.citation_validity", "Citation Validity", 0.01),
]
# Phase 2C is allowed to change how the corpus is cut, so its plan adds two
# retrieval guardrails that 2B, which froze retrieval, did not need.
RETRIEVAL_GUARDRAILS = [
    ("retrieval.hit_rate@5", "Hit@5", 0.01),
    ("retrieval.recall@5", "Recall@5", 0.01),
]
PRIMARY = "ragas.answer_correctness"
REPORTED = [
    (PRIMARY, "Answer Correctness"),
    ("qa.exact_match", "Exact Match"),
    ("qa.f1", "Token F1"),
    ("ragas.faithfulness", "Faithfulness"),
    ("citations.citation_f1", "Citation F1"),
    ("citations.citation_validity", "Citation Validity"),
    ("ragas.answer_relevancy", "Answer Relevancy"),
]
MIN_COVERAGE = 0.95


def read_scores(path: Path) -> dict[str, dict]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {row["question_id"]: row for row in rows}


def value(row: dict, dotted: str):
    node = row
    for key in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def paired_delta(base: dict, cand: dict, clusters: dict[str, list[str]], metric: str) -> dict:
    """Mean of cand - base per question, with a cluster bootstrap CI over articles.

    Resampling whole articles rather than questions is what keeps the interval
    honest: several questions can share one article, so treating them as
    independent draws would understate the spread.
    """
    per_cluster = {
        key: [value(cand[q], metric) - value(base[q], metric) for q in questions]
        for key, questions in clusters.items()
    }
    flat = [delta for deltas in per_cluster.values() for delta in deltas]
    observed = sum(flat) / len(flat)

    keys = list(clusters)
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        drawn = [d for _ in keys for d in per_cluster[rng.choice(keys)]]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    low = means[int(0.025 * BOOTSTRAP_SAMPLES)]
    high = means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]

    # How fine a difference this design can actually resolve. The cluster
    # bootstrap's spread is the honest standard error; comparing it against the
    # question-level one says how much the article clustering costs us, and the
    # minimum detectable effect says which guardrail thresholds are even
    # measurable at this sample size.
    se_cluster = statistics.pstdev(means)
    se_question = statistics.pstdev(_question_bootstrap(flat))
    design_effect = (se_cluster / se_question) ** 2 if se_question else float("nan")
    return {
        "delta": round(observed, 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
        "n_pairs": len(flat),
        "n_clusters": len(keys),
        "significant": low > 0 or high < 0,
        "resolution": {
            "se_cluster_bootstrap": round(se_cluster, 6),
            "se_question_bootstrap": round(se_question, 6),
            "design_effect": round(design_effect, 3),
            "effective_n": round(len(flat) / design_effect, 1),
            "ci95_half_width": round((high - low) / 2, 6),
            # Two-sided alpha 0.05, power 0.80: |delta| must exceed 2.802 SE.
            "min_detectable_effect": round(2.802 * se_cluster, 6),
        },
    }


def _question_bootstrap(deltas: list[float]) -> list[float]:
    """The same bootstrap ignoring article structure, for comparison only.

    Phase 1 and the chunking ablation resample this way. Running both makes the
    cost of that choice a number instead of an assertion.
    """
    rng = random.Random(SEED)
    n = len(deltas)
    return [sum(rng.choice(deltas) for _ in range(n)) / n
            for _ in range(BOOTSTRAP_SAMPLES)]


def mean(scores: dict, metric: str) -> float:
    values = [value(row, metric) for row in scores.values()]
    return sum(values) / len(values)


def article_macro(scores: dict, clusters: dict[str, list[str]], metric: str) -> float:
    """Mean over articles of the per-article mean.

    The plain mean lets an article that contributed twelve questions outvote one
    that contributed two. Both test plans ask for this alongside it.
    """
    per_article = [sum(value(scores[q], metric) for q in qs) / len(qs)
                   for qs in clusters.values()]
    return sum(per_article) / len(per_article)


def strata(base: dict) -> dict[str, list[str]]:
    """Split the questions by whether retrieval put gold in front of the model.

    Retrieval is frozen across every Phase 2 run, so this split is a property of
    the question set, not of the configuration being scored. It separates a
    generation failure from a retrieval failure the generator could not recover
    from.
    """
    hit = lambda qid: base[qid]["retrieval"]["hit_rate@5"] == 1.0
    return {
        "gold_in_top5": [q for q in base if hit(q)],
        "gold_not_in_top5": [q for q in base if not hit(q)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="p0_d5_development.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", choices=("2b", "2c"), default="2b",
                        help="2c adds the Hit@5/Recall@5 guardrails of its plan section 9")
    args = parser.parse_args()
    guardrails = GUARDRAILS + (RETRIEVAL_GUARDRAILS if args.plan == "2c" else [])
    reported = REPORTED + ([(metric, label) for metric, label, _ in RETRIEVAL_GUARDRAILS]
                           if args.plan == "2c" else [])

    # Compare within one partition only. The held-out run has no paired baseline
    # by design -- it is allowed exactly one execution -- so sweeping its score
    # file in here would fail the pairing check for the right reason but the
    # wrong cause.
    partition = "_" + Path(args.baseline).stem.rsplit("_", 1)[-1]
    files = sorted(args.scores_dir.glob(f"*{partition}.jsonl"))
    runs = {path.stem.replace(partition, ""): read_scores(path) for path in files}
    base_name = Path(args.baseline).stem.replace(partition, "")
    if base_name not in runs:
        parser.error(f"baseline {base_name} not among {sorted(runs)}")
    base = runs[base_name]

    # Every run must cover the same questions, or the pairing is not a pairing.
    for name, scores in runs.items():
        if set(scores) != set(base):
            parser.error(f"{name} does not cover the same question set as {base_name}")

    clusters: dict[str, list[str]] = defaultdict(list)
    for qid, row in base.items():
        clusters[row["article_key"]].append(qid)

    groups = strata(base)
    print(f"{len(base)} questions across {len(clusters)} articles; baseline = {base_name}")
    print("   strata: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    print()

    candidates = [name for name in runs if name != base_name]
    results, eligible = {}, []
    for name in candidates:
        cand = runs[name]
        coverage = len(cand) / len(base)
        checks, passed = [], coverage >= MIN_COVERAGE
        for metric, label, tolerance in guardrails:
            delta = mean(cand, metric) - mean(base, metric)
            ok = delta >= -tolerance
            passed &= ok
            checks.append({"metric": label, "delta": round(delta, 6),
                           "tolerance": -tolerance, "passed": ok})
        comparisons = {label: paired_delta(base, cand, clusters, metric)
                       for metric, label in reported}
        results[name] = {
            "coverage": round(coverage, 4),
            "guardrails": checks,
            "eligible": passed,
            "answer_correctness": round(mean(cand, PRIMARY), 6),
            "answer_correctness_article_macro": round(article_macro(cand, clusters, PRIMARY), 6),
            "paired_vs_baseline": comparisons,
            "by_stratum": {
                stratum: {
                    "n": len(qids),
                    "baseline": round(sum(value(base[q], PRIMARY) for q in qids) / len(qids), 6),
                    "candidate": round(sum(value(cand[q], PRIMARY) for q in qids) / len(qids), 6),
                }
                for stratum, qids in groups.items() if qids
            },
        }
        if passed:
            eligible.append(name)

        print(f"-- {name} --")
        for check in checks:
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"   {check['metric']:20} {check['delta']:+.6f}  "
                  f"tol {check['tolerance']:+.2f}  {mark}")
        print(f"   {'coverage':20} {coverage:.2%}  "
              f"{'PASS' if coverage >= MIN_COVERAGE else 'FAIL'}")
        print(f"   => {'ELIGIBLE' if passed else 'DISQUALIFIED'}, "
              f"Answer Correctness {mean(cand, PRIMARY):.4f} "
              f"(article macro {article_macro(cand, clusters, PRIMARY):.4f})")
        for stratum, stats in results[name]["by_stratum"].items():
            print(f"   {stratum:20} n={stats['n']:3}  "
                  f"{stats['baseline']:.4f} -> {stats['candidate']:.4f}")
        print()

    winner = max(eligible, key=lambda name: results[name]["answer_correctness"], default=None)
    print(f"Winner: {winner or 'none eligible - keep the baseline'}")
    if eligible and winner:
        losers = [n for n in candidates if n != winner]
        best_disqualified = [n for n in losers
                             if results[n]["answer_correctness"] > results[winner]["answer_correctness"]]
        if best_disqualified:
            print("Note: " + ", ".join(best_disqualified) +
                  " scored higher on the primary metric but failed a guardrail.")

    payload = {
        "schema_version": 1,
        "method": "paired article-cluster percentile bootstrap over per-question scores",
        "rule_source": ("docs/Detailed Test Plans/phase_2_generation_tuning_plan.md section 7"
                        if args.plan == "2b" else
                        "docs/Detailed Test Plans/phase_2c_chunking_strategy_test_plan.md section 9"),
        "plan": args.plan,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "baseline": base_name,
        "n_questions": len(base),
        "n_articles": len(clusters),
        "strata_sizes": {k: len(v) for k, v in groups.items()},
        "baseline_answer_correctness": round(mean(base, PRIMARY), 6),
        "baseline_answer_correctness_article_macro": round(article_macro(base, clusters, PRIMARY), 6),
        "winner": winner,
        "runs": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
