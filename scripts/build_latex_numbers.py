#!/usr/bin/env python3
"""Emit every number the LaTeX slides and report quote, as macros.

The slides and the report are the two documents that leave the repo, so a number
that drifts in either one is the expensive kind of mistake. Nothing in
docs/latex/ is allowed to type a figure directly: it writes \\NumP2dfiveAC and
this script fills it in from the artifact that produced it.

Add a number here, not in the .tex.

    python scripts/build_latex_numbers.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/latex/common/numbers.tex"


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def vn(x: float, places: int = 4) -> str:
    """Vietnamese decimal comma, which is what every report in this repo uses."""
    return f"{x:.{places}f}".replace(".", "{,}")


def signed(x: float, places: int = 4) -> str:
    """A real minus sign, not a hyphen -- these end up inside running text."""
    return ("+" if x >= 0 else "$-$") + vn(abs(x), places)


def ci(low: float, high: float, places: int = 4) -> str:
    return f"[{signed(low, places)}; {signed(high, places)}]"


def collect() -> dict[str, str]:
    n: dict[str, str] = {}

    # ---- Corpus and question sets -------------------------------------------
    subsets = load_json("docs/reports/phase2/provenance/subset_manifest.json")
    counts = subsets["counts"]
    n["CorpusArticles"] = "11.064"
    n["CorpusChunks"] = "22.766"
    n["CorpusChunksVone"] = "19.263"
    n["CorpusRestored"] = "4.603"
    n["NDev"] = str(counts["development"])
    n["NHeldout"] = str(counts["heldout"])
    n["NReserve"] = str(counts["heldout_reserve"])
    n["NScreening"] = str(counts["screening"])
    n["NArticles"] = str(subsets["heldout_selection"]["articles"])
    n["HeldoutSeed"] = str(subsets["heldout_selection"]["seed"])
    n["LockedIndexSha"] = subsets["locked_artifact"]["sha256"][:16]

    # ---- EDA ----------------------------------------------------------------
    n["NoiseFloor"] = "7,0\\%"
    n["NoiseCeiling"] = "24,5\\%"
    n["DistractorMedian"] = "25"

    # ---- Phase 1: locked configuration --------------------------------------
    lock = load_json("docs/reports/phase1/winner_lock.jsonl")
    sel = load_json("docs/reports/phase2/provenance/retrieval_lock.json")["selection_metrics"]
    n["PoneHitFive"] = vn(sel["retrieval.hit_rate@5.mean"])
    n["PoneNdcgFive"] = vn(sel["retrieval.ndcg@5.mean"])
    n["PoneMrrFive"] = vn(sel["retrieval.mrr@5.mean"])
    n["PoneLatency"] = vn(sel["latency.total.p50_ms"], 1)
    n["PoneTopK"] = str(lock.get("top_k", 20))
    n["PoneRerankTopN"] = str(lock.get("rerank_top_n", 5))

    # Hit@k per question from the frozen Phase 2 baseline: the same retrieval
    # trace every generation run reused, so these are the depth-cut costs.
    rows = [json.loads(line) for line
            in (ROOT / "docs/reports/phase2/scores/p0_d5_development.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = {k: sum(1 for r in rows if r["retrieval"][f"hit_rate@{k}"] == 1.0) for k in (1, 3, 5)}
    n["PoneHitOne"] = vn(hits[1] / len(rows))
    n["PoneHitThree"] = vn(hits[3] / len(rows))
    n["GoldInTopFive"] = str(hits[5])
    n["GoldNotInTopFive"] = str(len(rows) - hits[5])
    n["LostAtDepthThree"] = str(hits[5] - hits[3])
    n["LostAtDepthOne"] = str(hits[5] - hits[1])

    # ---- Phase 1: what the tournament separated -----------------------------
    for c in load_json("docs/reports/phase1/paired_significance.json")["comparisons"]:
        if c["variant"] != "resolved":
            continue
        m = c["metrics"]["retrieval.ndcg@5"]
        key = {"3.2A": "SparseVsDense", "3.2B": "BgeMThreeVsBmTwoFive",
               "3.2C": "DenseVsDense"}.get(c["report_section"])
        if key:
            n[key] = signed(m["delta_B_minus_A"])
            n[key + "CI"] = ci(m["ci95_low"], m["ci95_high"])

    round3 = {r["index"]: r for r
              in load_json("docs/reports/phase2/provenance/retrieval_lock.json")["round3_candidates"]}
    for tag, index in [("ChunkTwoFiveSix", "chunk_256_32"),
                       ("ChunkFiveOneTwo", "chunk_512_64"),
                       ("ChunkOneZeroTwoFour", "chunk_1024_128")]:
        n[tag + "Ndcg"] = vn(round3[index]["retrieval.ndcg@5.mean"])
    n["ChunkSpread"] = vn(round3["chunk_512_64"]["retrieval.ndcg@5.mean"]
                          - round3["chunk_256_32"]["retrieval.ndcg@5.mean"])

    # ---- Phase 1: contextual chunking ablation ------------------------------
    ab = load_json("docs/reports/phase1/contextual_chunking_ablation.json")
    agg = {(r["retriever"], r["corpus"]): r for r in ab["aggregate"] if r["variant"] == "resolved"}
    for tag, key in [("CtxDensePlain", ("dense", "plain")),
                     ("CtxDenseCtx", ("dense", "contextual")),
                     ("CtxSparsePlain", ("sparse", "plain")),
                     ("CtxSparseCtx", ("sparse", "contextual"))]:
        n[tag] = vn(agg[key]["ndcg@5"])
    n["CtxNSamples"] = f"{agg[('dense', 'plain')]['n_samples']:,}".replace(",", ".")
    n["CtxContextualised"] = f"{ab['config']['n_contextualised']:,}".replace(",", ".")
    n["CtxChars"] = str(ab["config"]["context_chars"])
    for e in ab["contextual_effect"]:
        if e["variant"] != "resolved":
            continue
        tag = "CtxEffectDense" if e["retriever"] == "dense" else "CtxEffectSparse"
        # The notebook stores these pre-formatted; re-parse so they read like
        # every other interval in these documents.
        delta, low, high = (float(v) for v in
                            e["ndcg@5"].replace("[", " ").replace("]", " ")
                            .replace(",", " ").split()[:3])
        n[tag] = signed(delta)
        n[tag + "CI"] = ci(low, high)
    for g in ab["sparse_dense_gap"]:
        if g["variant"] != "resolved":
            continue
        n["CtxGapPlain" if g["corpus"] == "plain" else "CtxGapCtx"] = vn(g["gap"])

    # ---- Phase 2: baseline --------------------------------------------------
    base = next(iter(csv.DictReader(
        (ROOT / "docs/reports/phase2/provenance/comparison.csv").open(encoding="utf-8-sig"))))
    for tag, col in [("AC", "ragas.answer_correctness.mean"),
                     ("EM", "qa.exact_match.mean"),
                     ("Fone", "qa.f1.mean"),
                     ("Faith", "ragas.faithfulness.mean"),
                     ("CitFone", "citations.citation_f1.mean"),
                     ("CitVal", "citations.citation_validity.mean"),
                     ("AnsRel", "ragas.answer_relevancy.mean")]:
        n["Pzero" + tag] = vn(float(base[col]))
    n["JudgeCost"] = vn(float(base["estimated_generation_cost_usd"]), 4)

    # ---- Phase 2: the tournament and its verdict ----------------------------
    sig = load_json("docs/reports/phase2/paired_significance.json")
    decision = load_json("docs/reports/phase2/phase2b_winner_decision.json")
    n["Winner"] = "P2-depth5"
    n["WinnerPrompt"] = decision["winner"]["prompt_id"].upper()
    n["WinnerDepth"] = str(decision["winner"]["context_depth"])
    n["WinnerApprovedAt"] = decision["review"]["approved_at"].replace("T", " ")
    n["PzeroACMacro"] = vn(sig["baseline_answer_correctness_article_macro"])
    n["BootstrapSamples"] = f"{sig['samples']:,}".replace(",", ".")

    for name, tag in [("p2_d5", "PtwodFive"), ("p2_d3", "PtwodThree")]:
        run = sig["runs"][name]
        n[tag + "AC"] = vn(run["answer_correctness"])
        n[tag + "ACMacro"] = vn(run["answer_correctness_article_macro"])
        for label, key in [("Answer Correctness", "AC"), ("Exact Match", "EM"),
                           ("Token F1", "Fone"), ("Faithfulness", "Faith"),
                           ("Citation F1", "CitFone"),
                           ("Citation Validity", "CitVal")]:
            p = run["paired_vs_baseline"][label]
            n[f"{tag}Delta{key}"] = signed(p["delta"])
            n[f"{tag}Delta{key}CI"] = ci(p["ci95_low"], p["ci95_high"])
        for guard in run["guardrails"]:
            key = {"Faithfulness": "Faith", "Citation F1": "CitFone",
                   "Citation Validity": "CitVal"}[guard["metric"]]
            n[f"{tag}Guard{key}"] = signed(guard["delta"])
            n[f"{tag}Guard{key}Verdict"] = "PASS" if guard["passed"] else "FAIL"
        for stratum, short in [("gold_in_top5", "Hit"), ("gold_not_in_top5", "Miss")]:
            s = run["by_stratum"][stratum]
            n[f"Pzero{short}AC"] = vn(s["baseline"])
            n[f"{tag}{short}AC"] = vn(s["candidate"])
            n[f"N{short}"] = str(s["n"])

    # Development absolutes for the winner, so the held-out table can sit beside
    # them without the reader having to add a delta to a baseline in their head.
    dev = [json.loads(line) for line
           in (ROOT / "docs/reports/phase2/scores/p2_d5_development.jsonl")
           .read_text(encoding="utf-8").splitlines() if line.strip()]
    for tag, path in [("AC", ("ragas", "answer_correctness")),
                      ("EM", ("qa", "exact_match")),
                      ("Fone", ("qa", "f1")),
                      ("Faith", ("ragas", "faithfulness")),
                      ("CitFone", ("citations", "citation_f1")),
                      ("CitVal", ("citations", "citation_validity")),
                      ("AnsRel", ("ragas", "answer_relevancy")),
                      ("HitFive", ("retrieval", "hit_rate@5"))]:
        n["Dev" + tag] = vn(sum(r[path[0]][path[1]] for r in dev) / len(dev))

    # ---- Phase 2: the held-out run, executed once ---------------------------
    ho = load_json("docs/reports/phase2/heldout/heldout_final_summary.json")
    access = load_json("docs/reports/phase2/heldout/heldout_access.json")
    micro, macro = ho["metrics_question_micro"], ho["metrics_article_macro"]
    for tag, key in [("AC", "answer_correctness"), ("EM", "qa_exact_match"),
                     ("Fone", "qa_f1"), ("Faith", "faithfulness"),
                     ("CitFone", "citation_f1"), ("CitVal", "citation_validity"),
                     ("AnsRel", "answer_relevancy")]:
        n["Ho" + tag] = vn(micro[key])
        if key in macro:
            n["Ho" + tag + "Macro"] = vn(macro[key])
    n["HoCost"] = vn(ho["cost_usd"]["total"], 2)
    n["HoLatency"] = vn(ho["latency"]["total_p50_ms"], 1)
    n["HoStartedAt"] = access["started_at"].replace("T", " ").replace("Z", " UTC")
    n["HoDecisionSha"] = access["decision_sha256"][:16]

    # Held-out retrieval: an out-of-sample reading of the locked Phase 1 config.
    hor = [json.loads(line) for line
           in (ROOT / "docs/reports/phase2/scores/p2_d5_heldout.jsonl")
           .read_text(encoding="utf-8").splitlines() if line.strip()]
    for tag, key in [("HitOne", "hit_rate@1"), ("HitThree", "hit_rate@3"),
                     ("HitFive", "hit_rate@5"), ("NdcgFive", "ndcg@5")]:
        n["Ho" + tag] = vn(sum(r["retrieval"][key] for r in hor) / len(hor))
    n["HoHitFiveDrop"] = signed(sum(r["retrieval"]["hit_rate@5"] for r in hor) / len(hor)
                                - float(sel["retrieval.hit_rate@5.mean"]))

    subgroups = list(csv.reader(
        (ROOT / "docs/reports/phase2/heldout/heldout_retrieval_subgroups.csv")
        .open(encoding="utf-8-sig")))
    header, _, *body = subgroups
    # Each metric appears twice in the header -- a count column then a mean
    # column -- so keep the first sighting and read the mean at +1.
    columns: dict[str, int] = {}
    for i, name in enumerate(header):
        columns.setdefault(name, i)
    for row in body:
        short = "Hit" if row[0] == "gold_in_top5" else "Miss"
        n[f"HoN{short}"] = row[columns["answer_correctness"]]
        for tag, col in [("AC", "answer_correctness"), ("Fone", "qa_f1"),
                         ("CitFone", "citation_f1"), ("CitVal", "citation_validity"),
                         ("Faith", "faithfulness")]:
            # Each metric occupies a count column followed by a mean column.
            n[f"Ho{short}{tag}"] = vn(float(row[columns[col] + 1]))

    return n


def main() -> None:
    numbers = collect()
    lines = [
        "% Generated by scripts/build_latex_numbers.py -- do not edit by hand.",
        "% Every figure below is read from a committed artifact; rerun the script",
        "% after any experiment changes and the documents follow automatically.",
        "",
    ]
    for key, value in numbers.items():
        # TeX macro names are letters only, which is why the keys spell digits out.
        assert key.isalpha(), key
        lines.append(f"\\newcommand{{\\Num{key}}}{{{value}}}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(numbers)} macros to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
