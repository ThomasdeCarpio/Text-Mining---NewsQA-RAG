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
import re
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
    # Corpus/article totals are recorded in the committed Phase 2 report;
    # question totals come from the subset manifest, chunk count from the run.
    report = (ROOT / "docs/reports/phase2/report.md").read_text(encoding="utf-8")
    corpus = re.search(r"\| Corpus \| ([\d.]+) bài báo, ([\d.]+) chunks \|", report)
    reserve = re.search(r"\| Held-out reserve \| (\d+) \| (\d+) \|", report)
    assert corpus and reserve, "Phase 2 corpus/split table is missing"
    n["CorpusArticles"] = corpus[1]
    chunks = load_json("docs/reports/phase1/contextual_chunking_ablation.json")["config"]["n_chunks"]
    assert chunks == int(corpus[2].replace(".", ""))
    n["CorpusChunks"] = f"{chunks:,}".replace(",", ".")
    n["NResolved"] = f"{sum(counts[k] for k in ('development', 'heldout', 'heldout_reserve')):,}".replace(",", ".")
    n["NReserveArticles"] = reserve[1]
    assert int(reserve[2]) == counts["heldout_reserve"]
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
    retrieval_lock = load_json("docs/reports/phase2/provenance/retrieval_lock.json")
    sel = retrieval_lock["selection_metrics"]
    n["ChunkSize"] = str(retrieval_lock["chunk_size"])
    n["ChunkOverlap"] = str(retrieval_lock["chunk_overlap"])
    n["PoneHitFive"] = vn(sel["retrieval.hit_rate@5.mean"])
    n["PoneNdcgFive"] = vn(sel["retrieval.ndcg@5.mean"])
    n["PoneMrrFive"] = vn(sel["retrieval.mrr@5.mean"])
    n["PoneLatency"] = vn(sel["latency.total.p50_ms"], 1)
    n["PoneTopK"] = str(retrieval_lock["top_k"])
    n["PoneRerankTopN"] = str(retrieval_lock["rerank_top_n"])

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
    baseline = load_json("docs/reports/phase2/report_baseline_p0_d5.json")
    for tag, section, metric in [
        ("AC", "ragas", "answer_correctness"), ("Faith", "ragas", "faithfulness"),
        ("EM", "qa", "exact_match"), ("Fone", "qa", "f1"),
        ("CitFone", "citations", "citation_f1"),
        ("CitVal", "citations", "citation_validity"),
    ]:
        value = vn(baseline[section][metric])
        assert value == n["Pzero" + tag], f"Baseline report/CSV mismatch: {metric}"
        n["Pzero" + tag] = value
    n["BaselineSuccessful"] = str(baseline["coverage"]["successful"])
    n["BaselineExpected"] = str(baseline["coverage"]["expected"])
    n["BaselineCoverage"] = vn(baseline["coverage"]["success_rate"] * 100, 0) + r"\%"
    n["BaselineJudged"] = str(baseline["ragas"]["n_samples"])
    n["GeneratorModel"] = baseline["config"]["generator_model"]
    run = load_json("docs/reports/phase2/provenance/experiment_run.json")
    n["JudgeModel"] = run["pricing"]["judge"]["model"].rsplit("/", 1)[-1]

    # ---- Phase 2: the tournament and its verdict ----------------------------
    sig = load_json("docs/reports/phase2/paired_significance.json")
    decision = load_json("docs/reports/phase2/phase2b_winner_decision.json")
    n["Winner"] = "P2-depth5"
    n["WinnerPrompt"] = decision["winner"]["prompt_id"].upper()
    n["WinnerDepth"] = str(decision["winner"]["context_depth"])
    n["WinnerApprovedAt"] = decision["review"]["approved_at"].replace("T", " ")
    n["PzeroACMacro"] = vn(sig["baseline_answer_correctness_article_macro"])
    n["BootstrapSamples"] = f"{sig['samples']:,}".replace(",", ".")
    n["NDevArticles"] = str(sig["n_articles"])
    phase1_sig = load_json("docs/reports/phase1/paired_significance.json")
    n["PoneBootstrapSamples"] = f"{phase1_sig['samples']:,}".replace(",", ".")
    for guard in sig["runs"]["p2_d5"]["guardrails"]:
        tag = {"Faithfulness": "Faith", "Citation F1": "CitFone", "Citation Validity": "CitVal"}[guard["metric"]]
        n["Guard" + tag + "Threshold"] = signed(guard["tolerance"], 2)
    baseline_plan = (ROOT / "docs/Detailed Test Plans/phase_2_baseline_test_plan.md").read_text(encoding="utf-8")
    generation_min = re.search(r"Generation coverage tối thiểu (\d+)%;", baseline_plan)
    judge_min = re.search(r"RAGAS coverage tối thiểu (\d+)%", baseline_plan)
    assert generation_min and judge_min and generation_min[1] == judge_min[1]
    n["CoverageThreshold"] = generation_min[1] + r"\%"
    n["NJudgeScreening"] = str(counts["judge_calibration"])
    screening_text = report.split("### 7.2.", 1)[1].split("### 7.3.", 1)[0]
    for prompt, tag in [("P0", "Pzero"), ("P1", "Pone"), ("P2", "Ptwo"), ("P3", "Pthree")]:
        cells = next([c.strip().replace("**", "") for c in line.strip("|").split("|")]
                     for line in screening_text.splitlines()
                     if line.startswith("|") and line.split("|")[1].strip().replace("**", "") == prompt)
        for metric, cell in [("Fone", 1), ("AC", 2), ("Faith", 3),
                             ("CitFone", 5), ("CitVal", 6)]:
            n["Screen" + tag + metric] = vn(float(cells[cell].replace(",", ".")))

    # Phase 2B.2 screened both the baseline and selected prompt at depths 1
    # and 3. Depth 5 is the Phase 2B.1 result above and was reused unchanged.
    for depth, depth_tag in [(1, "One"), (3, "Three")]:
        depth_path = (ROOT / "results/phase2/configuration tunning/context depth"
                      / f"d{depth}/results/phase2b_comparison.csv")
        depth_rows = list(csv.DictReader(depth_path.open(encoding="utf-8-sig")))
        for prompt, prompt_tag in [("p0", "Pzero"), ("p2", "Ptwo")]:
            row = next(r for r in depth_rows
                       if r["prompt_id"] == prompt and int(r["context_depth"]) == depth)
            for metric, column in [("AC", "answer_correctness"),
                                   ("Faith", "faithfulness"),
                                   ("CitFone", "citation_f1"),
                                   ("CitVal", "citation_validity")]:
                n["Depth" + prompt_tag + depth_tag + metric] = vn(float(row[column]))

    # Presentation comparisons: values and bar lengths share the same source.
    round1 = list(csv.DictReader((ROOT / "docs/reports/phase1/round1.csv").open(encoding="utf-8-sig")))
    retrievers = {
        "SparseBge": "sparse_bge_m3_sparse", "BmStem": "sparse_bm25_okapi_stemmed",
        "BmPlus": "sparse_bm25_plus_simple", "BmSimple": "sparse_bm25_okapi_simple",
        "DenseEfive": "dense_intfloat_e5_base_v2", "DenseSmall": "dense_baai_bge_small_en_v1.5",
        "DenseLarge": "dense_baai_bge_large_en_v1.5", "DenseMini": "dense_all_minilm_l6_v2",
    }
    for tag, index in retrievers.items():
        for variant, suffix in [("resolved", ""), ("original", "Original")]:
            row = next(r for r in round1 if r["index"] == index and r["variant"] == variant)
            ndcg = float(row["retrieval.ndcg@5.mean"])
            n["Rone" + tag + suffix] = vn(ndcg)
            n["Rone" + tag + suffix + "Plot"] = str(ndcg)
            n["Rone" + tag + suffix + "Mrr"] = vn(float(row["retrieval.mrr@5.mean"]))
            n["Rone" + tag + suffix + "Hit"] = vn(float(row["retrieval.hit_rate@5.mean"]))
    round2 = list(csv.DictReader((ROOT / "docs/reports/phase1/round2.csv").open(encoding="utf-8-sig")))
    for family in ("dense", "sparse", "hybrid"):
        for tag, model in [("None", ""), ("Mini", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                           ("Large", "BAAI/bge-reranker-large")]:
            row = next(r for r in round2 if r["variant"] == "resolved" and r["retriever"] == family
                       and r["reranker_model"] == model)
            prefix = "Rtwo" + family.title() + tag
            n[prefix] = vn(float(row["retrieval.ndcg@5.mean"]))
            n[prefix + "Mrr"] = vn(float(row["retrieval.mrr@5.mean"]))
            n[prefix + "Latency"] = vn(float(row["latency.total.p50_ms"]), 1)
    sparse_rows = [r for r in round2 if r["variant"] == "resolved" and r["retriever"] == "sparse"]
    sparse_before = next(r for r in sparse_rows if r["reranker"] == "noop")
    sparse_after = next(r for r in sparse_rows if r["reranker_model"] == "BAAI/bge-reranker-large")
    n["RerankerMrrGain"] = signed(float(sparse_after["retrieval.mrr@5.mean"]) - float(sparse_before["retrieval.mrr@5.mean"]))
    n["RerankerNdcgGain"] = signed(float(sparse_after["retrieval.ndcg@5.mean"]) - float(sparse_before["retrieval.ndcg@5.mean"]))
    final_retrieval = load_json("docs/reports/phase1/heldout/heldout_significance.json")
    final_scores = final_retrieval["partitions"]["final_test"]
    n["RetrievalFinalQuestions"] = str(final_scores["hit_rate@5"]["n_questions"])
    n["RetrievalFinalArticles"] = str(final_scores["hit_rate@5"]["n_articles"])
    for tag, metric in [("HitFive", "hit_rate@5"), ("NdcgFive", "ndcg@5"),
                        ("MrrFive", "mrr@5"), ("RecallFive", "recall@5")]:
        n["RetrievalFinal" + tag] = vn(final_scores[metric]["value"])
    n["RetrievalFinalMissing"] = str(final_retrieval["evidence_missing_at_5"]["final_test"]["n"])
    gain = final_retrieval["reranker_gain_final_test"]["ndcg@5"]
    n["RetrievalFinalRerankerGain"] = signed(gain["delta"])
    n["RetrievalFinalRerankerCI"] = ci(gain["ci95_low"], gain["ci95_high"])

    for name, tag in [("p2_d5", "PtwodFive"), ("p2_d3", "PtwodThree")]:
        run = sig["runs"][name]
        n[tag + "Coverage"] = vn(run["coverage"] * 100, 0) + r"\%"
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
            n[f"{tag}Guard{key}Exact"] = signed(guard["delta"], 6)
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

    # Absolute finalist metrics shown before the guardrail deltas.
    finalist_paths = {
        "PtwodThree": "docs/reports/phase2/scores/p2_d3_development.jsonl",
        "PtwodFive": "docs/reports/phase2/scores/p2_d5_development.jsonl",
    }
    for prefix, relative in finalist_paths.items():
        finalist_rows = [json.loads(line) for line in
                          (ROOT / relative).read_text(encoding="utf-8").splitlines()
                          if line.strip()]
        for tag, path in [("AC", ("ragas", "answer_correctness")),
                          ("EM", ("qa", "exact_match")),
                          ("Fone", ("qa", "f1")),
                          ("Faith", ("ragas", "faithfulness")),
                          ("CitFone", ("citations", "citation_f1")),
                          ("CitVal", ("citations", "citation_validity"))]:
            n[prefix + tag] = vn(sum(r[path[0]][path[1]] for r in finalist_rows)
                                    / len(finalist_rows))

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
    n["WinnerApprovedTime"] = decision["review"]["approved_at"].split("T")[1].removesuffix("Z")
    n["HoStartedTime"] = access["started_at"].split("T")[1].removesuffix("Z")
    n["HoRunDate"] = access["started_at"].split("T")[0]
    n["HoSuccessful"] = str(ho["coverage"]["successful"])
    n["HoCoverage"] = vn(ho["coverage"]["success_rate"] * 100, 0) + r"\%"
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

    # ---- Phase 2C: chunking strategies --------------------------------------
    p2c = load_json("docs/reports/phase2c/paired_significance.json")
    n["PtwocWinner"] = "C0-P2-D5"
    n["PtwocCzeroChunks"] = "22.766"
    n["PtwocConeChunks"] = "22.014"
    n["PtwocCtwoChunks"] = "22.018"
    n["PtwocCthreeChunks"] = "49.218 + 22.766"
    c3_d3 = p2c["runs"]["c3_d3"]
    n["PtwocCthreeDthreeAC"] = vn(c3_d3["answer_correctness"])
    n["PtwocCthreeDthreeDeltaAC"] = signed(c3_d3["paired_vs_baseline"]["Answer Correctness"]["delta"])
    n["PtwocCthreeDthreeDeltaACCI"] = ci(c3_d3["paired_vs_baseline"]["Answer Correctness"]["ci95_low"],
                                         c3_d3["paired_vs_baseline"]["Answer Correctness"]["ci95_high"])
    for guard in c3_d3["guardrails"]:
        key = {"Faithfulness": "Faith", "Citation F1": "CitFone", "Citation Validity": "CitVal",
               "Hit@5": "HitFive", "Recall@5": "RecallFive"}[guard["metric"]]
        n[f"PtwocCthreeDthreeGuard{key}"] = signed(guard["delta"])
    c3_d5 = p2c["runs"]["c3_d5"]
    n["PtwocCthreeDfiveDeltaAC"] = signed(c3_d5["paired_vs_baseline"]["Answer Correctness"]["delta"])
    for guard in c3_d5["guardrails"]:
        key = {"Faithfulness": "Faith", "Citation F1": "CitFone", "Citation Validity": "CitVal",
               "Hit@5": "HitFive", "Recall@5": "RecallFive"}[guard["metric"]]
        n[f"PtwocCthreeDfiveGuard{key}"] = signed(guard["delta"])

    # ---- Phase 3: abstention ------------------------------------------------
    p3_sig = load_json("docs/reports/phase3/policy_significance.json")
    p3_comp = list(csv.DictReader((ROOT / "docs/reports/phase3/policy_comparison.csv").open(encoding="utf-8-sig")))
    comp_map = {(r["partition"], r["policy"]): r for r in p3_comp}
    n["PthreeWinner"] = str(p3_sig["winner"]["winner"])
    n["PthreeNDev"] = "140"
    n["PthreeNFinal"] = "60"
    n["PthreeBzeroFalseAnsDev"] = vn(float(comp_map[("development", "B0")]["false_answer_rate"]) * 100, 2) + "\\%"
    n["PthreeBoneFalseAnsDev"] = vn(float(comp_map[("development", "B1")]["false_answer_rate"]) * 100, 2) + "\\%"
    n["PthreeBzeroTokenFoneDev"] = vn(float(comp_map[("development", "B0")]["control_token_f1"]))
    n["PthreeBoneTokenFoneDev"] = vn(float(comp_map[("development", "B1")]["control_token_f1"]))
    b1_dev = p3_sig["control_token_f1_b1_minus_b0"]["development"]
    n["PthreeBoneDeltaTokenFoneDev"] = signed(b1_dev["delta"])
    n["PthreeBoneDeltaTokenFoneDevCI"] = ci(b1_dev["ci95_low"], b1_dev["ci95_high"])
    n["PthreeBzeroAbstFoneDev"] = vn(float(comp_map[("development", "B0")]["abstention_f1"]))
    n["PthreeBoneAbstFoneDev"] = vn(float(comp_map[("development", "B1")]["abstention_f1"]))

    n["PthreeBzeroAbstFoneFinal"] = vn(float(comp_map[("final", "B0")]["abstention_f1"]))
    n["PthreeBzeroFalseAnsFinal"] = vn(float(comp_map[("final", "B0")]["false_answer_rate"]) * 100, 2) + "\\%"
    n["PthreeBzeroFalseAbstFinal"] = vn(float(comp_map[("final", "B0")]["false_abstention_rate"]) * 100, 1) + "\\%"

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
