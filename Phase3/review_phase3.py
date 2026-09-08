"""Rebuild one reviewed draft from the actual uploaded Phase3 bundle (offline).

No testset/reserve input, historical corpus identity gate or fabricated approval.
Original files are never overwritten. Generated files live under reviewed/.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from phase3_api import atomic_json, stable_hash

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "phase3_full_review_bundle/results/phase3"
OUTPUT = BUNDLE / "reviewed"
TYPES = {"answerable_control": (61, 26), "natural_retrieval_miss": (2, 1),
         "controlled_context_ablation": (16, 6), "removed_article": (16, 6),
         "counterfactual": (15, 7), "external_unanswerable": (15, 7),
         "partial_weak_evidence": (15, 7)}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def validate_cases(cases):
    if len(cases) != 200 or len({c["case_id"] for c in cases}) != 200:
        raise ValueError("Expected 200 distinct slots; do not silently run a subset")
    counts = Counter((c["case_type"], c["partition"]) for c in cases)
    expected = {(t, p): n for t, ns in TYPES.items() for p, n in zip(("development", "final_test"), ns)}
    if dict(counts) != expected:
        raise ValueError(f"Type/partition quota changed: {counts}")
    partitions = defaultdict(set)
    for c in cases:
        for key in ("base_question_id", "source_article_id"):
            partitions[(key, c[key])].add(c["partition"])
        if not c["question"].strip() or not c["review"]["notes"]:
            raise ValueError("Missing question or per-case review")
        expected_label = "answerable" if c["case_type"] == "answerable_control" else "insufficient_evidence"
        if c["answerability_label"] != expected_label:
            raise ValueError("Unexpected label")
        if c["scope"] == "provided_context" and c["case_type"] != "natural_retrieval_miss" and not 1 <= len(c["contexts"]) <= 5:
            raise ValueError("Provided-context case must contain 1–5 exact text passages")
        if c["scope"] == "provided_context" and c["case_type"] not in {"controlled_context_ablation", "partial_weak_evidence", "natural_retrieval_miss"}:
            raise ValueError("A full-corpus case was silently converted to provided-context")
    if any(len(ps) != 1 for ps in partitions.values()):
        raise ValueError("Base-question/source-article leakage across development/final")


def prepare(bundle=BUNDLE, decisions_path=HERE / "review_decisions.json", output=OUTPUT, corpus_path=HERE / "chunks.jsonl"):
    bundle, output = Path(bundle), Path(output)
    queue = read_jsonl(bundle / "review_queue.jsonl")
    authored = read_jsonl(bundle / "authored_cases.jsonl")
    readable = json.loads((bundle / "review_queue_readable.json").read_text(encoding="utf-8"))
    material = {c["case_id"]: c for group in readable["groups"] for c in group["cases"]}
    overlays = {c["case_id"]: c for c in read_jsonl(bundle / "corpus_overlays.jsonl")}
    if len(queue) != 156 or len(authored) != 44 or set(material) != {c["case_id"] for c in queue} or set(overlays) != set(material):
        raise ValueError("Original queue/readable/overlay membership disagrees")
    texts = {}
    for original in queue:
        readable_case = material[original["case_id"]]
        if {k: v for k, v in readable_case.items() if k != "review_material"} != original:
            raise ValueError("Readable copy differs from original case")
        overlay = overlays[original["case_id"]]
        if any(original.get(k) != v for k, v in overlay.items()):
            raise ValueError("Overlay disagrees with original case")
        for passages in readable_case["review_material"].values():
            for passage in passages:
                ident, text = passage["chunk_id"], passage["text"]
                if ident in texts and texts[ident] != text:
                    raise ValueError("Conflicting embedded texts under one ID")
                texts[ident] = text
    decisions = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    source_path = HERE / "external_source_evidence.jsonl"
    external_sources = {r["authored_row"]: r for r in read_jsonl(source_path)} if source_path.exists() else {}
    corpus = read_jsonl(corpus_path)
    uploaded = {c["id"]: c for c in corpus}
    cases = copy.deepcopy(queue)
    for row, c in enumerate(cases, 1):
        c["origin"] = {"file": "review_queue.jsonl", "row": row, "case_id": c["case_id"]}
        c["human_review"] = {"decision": "not_performed", "notes": "AI-only school experiment; no human sign-off required."}
        c["review"] = {"method": "AI source-text review, not human approval", "content_status": "reviewed",
                       "notes": [], "changes": [], "runtime_status": "pending_retrieval" if c["scope"] == "full_corpus" else "provided_context_reviewed"}
        c["source_evidence"] = copy.deepcopy(material[c["case_id"]]["review_material"]["source_gold_chunks"])
        c["contexts"] = [{"id": i, "text": texts[i]} for i in c["provided_context_chunk_ids"]] if c["scope"] == "provided_context" else []
        if c["case_type"] == "natural_retrieval_miss":
            c["historical_contexts"] = c["contexts"]
            c["contexts"] = []
            c["review"]["runtime_status"] = "pending_retrieval"

    def evidence(action):
        ident, quote = action.get("evidence_chunk_id"), action.get("evidence_quote")
        if ident and quote and normalized(quote) not in normalized(texts.get(ident, "")):
            raise ValueError(f"Repair evidence not found: {ident}: {quote}")

    def change(c, action, patch):
        evidence(action)
        before = {k: c.get(k) for k in patch}
        c.update(copy.deepcopy(patch))
        c["review"]["changes"].append({"reason": action["reason"], "before": before, "after": patch,
                                          "evidence_chunk_id": action.get("evidence_chunk_id"), "quote": action.get("evidence_quote", action.get("source_quote"))})

    for action in decisions["base_repairs"]:
        for c in cases:
            if c["base_question_id"] == action["base_question_id"]:
                change(c, action, action["patch"])
    for action in decisions["case_repairs"]:
        change(cases[action["row"] - 1], action, action["patch"])
    for action in decisions["ablation_repairs"]:
        c = cases[action["row"] - 1] if "row" in action else next(c for c in cases if c["case_id"] == action["case_id"])
        if c["case_type"] != "controlled_context_ablation":
            raise ValueError("Ablation repair applied to wrong type")
        removed = set(action["remove_chunk_ids"])
        if not removed <= set(c["provided_context_chunk_ids"]):
            raise ValueError("Removed context is not in the original ablation")
        change(c, action, {"contexts": [v for v in c["contexts"] if v["id"] not in removed],
                           "provided_context_chunk_ids": [v for v in c["provided_context_chunk_ids"] if v not in removed],
                           "excluded_chunk_ids": sorted(set(c["excluded_chunk_ids"]) | removed)})
    for action in decisions["weak_replacements"]:
        evidence(action)
        c = next(c for c in cases if c["case_id"] == action["retired_case_id"])
        control = next(c for c in cases if c["case_id"] == action["control_case_id"])
        if c["partition"] != control["partition"] or c["case_type"] != "partial_weak_evidence":
            raise ValueError("Replacement changes subtype/partition")
        ident = action["provided_chunk_id"]
        if normalized(action["provided_quote"]) not in normalized(texts[ident]):
            raise ValueError("Weak replacement evidence not found")
        patch = {k: copy.deepcopy(control[k]) for k in ("base_question_id", "source_question_id", "source_article_id", "source_gold_chunk_ids", "source_evidence", "question", "ground_truth", "accepted_answers")}
        patch.update(contexts=[{"id": ident, "text": texts[ident]}], provided_context_chunk_ids=[ident],
                     gold_relevant_chunk_ids=[], excluded_chunk_ids=list(control["source_gold_chunk_ids"]),
                     case_id="p3_weak_" + stable_hash({"base": control["base_question_id"], "context": ident})[:20])
        change(c, action, patch)

    for action in decisions["removed_replacements"]:
        c, control = cases[action["row"] - 1], cases[action["control_row"] - 1]
        if c["case_type"] != "removed_article" or control["case_type"] != "answerable_control" or c["partition"] != control["partition"]:
            raise ValueError("Removed-article replacement changes the experiment design")
        patch = {k: copy.deepcopy(control[k]) for k in ("base_question_id", "source_question_id", "source_article_id", "source_gold_chunk_ids", "source_evidence", "question", "ground_truth", "accepted_answers")}
        patch.update(case_id="p3_removed_" + stable_hash({"base": control["base_question_id"]})[:20],
                     excluded_article_ids=[control["source_article_id"], *sorted({i.split("_chunk_")[0] for i in control["source_gold_chunk_ids"]})],
                     excluded_chunk_ids=[], provided_context_chunk_ids=[], gold_relevant_chunk_ids=[])
        change(c, action, patch)
        hits = [v["id"] for v in corpus if re.search(action["scan_pattern"], v["text"], re.I)]
        c["review"]["local_anchor_scan"] = {"pattern": action["scan_pattern"], "hit_chunk_ids": hits,
                                                "notice": "AI-inspected candidate screen, not a proof that no semantic paraphrase exists."}

    controls = {c["base_question_id"]: c for c in cases if c["case_type"] == "answerable_control"}
    for row, proposal in enumerate(authored, 1):
        proposal = copy.deepcopy(proposal)
        cf = proposal["case_type"] == "counterfactual"
        source = controls[proposal["base_question_id"]] if cf else proposal
        c = {"schema_version": 1, "case_id": "p3_authored_" + stable_hash({"row": row, "question": proposal["question"]})[:20],
             "base_question_id": proposal["base_question_id"], "source_question_id": proposal["base_question_id"],
             "case_type": proposal["case_type"], "partition": source["partition"], "source_article_id": source["source_article_id"],
             "question": proposal["question"], "answerability_label": "insufficient_evidence", "scope": "full_corpus",
             "ground_truth": "", "accepted_answers": [], "gold_relevant_chunk_ids": [],
             "source_gold_chunk_ids": source.get("source_gold_chunk_ids", []), "source_evidence": copy.deepcopy(source.get("source_evidence", [])),
             "construction": proposal["construction"], "excluded_article_ids": [], "excluded_chunk_ids": [],
             "provided_context_chunk_ids": [], "contexts": [], "human_review": {"decision": "not_performed"},
             "origin": {"file": "authored_cases.jsonl", "row": row},
             "review": {"method": "AI source-text review, not human approval", "content_status": "reviewed" if cf else "provisional_source_span_only",
                        "runtime_status": "pending_retrieval", "notes": ["Original proposal assessment: " + decisions["authored_notes"][row - 1]["finding"]], "changes": []}}
        c["construction"]["original_proposal_status"] = c["construction"].get("proposal_status")
        c["construction"]["proposal_status"] = "AI_reviewed_pending_retrieval"
        if not cf and row in external_sources:
            recovered = external_sources[row]
            text = recovered["text"]
            if (hashlib.sha256(text.encode()).hexdigest() != proposal["construction"]["source_context_sha256"]
                    or recovered["source_article_id"] != c["source_article_id"]
                    or any(text[s["start"]:s["end"]] != s["text"] for s in proposal["construction"]["source_evidence_spans"])):
                raise ValueError("Recovered external passage does not match original source evidence")
            c["source_evidence"] = [{"chunk_id": c["source_article_id"] + "_source", "text": text,
                                      "source_url": recovered["source_url"], "sha256": recovered["sha256"]}]
            c["review"]["content_status"] = "reviewed"
            c["review"]["notes"].append("Full retained source passage recovered and AI-reviewed; exact source digest and all original answer-span offsets verified. This passage is provenance only, never added to generation contexts.")
        actions = decisions["counterfactual_repairs"] if cf else decisions["external_repairs"]
        for action in actions:
            if action.get("row") == row or action.get("base_question_id") == c["base_question_id"]:
                change(c, action, {"question": action["question"]})
                if cf:
                    c["construction"]["changed_field"] = action["changed_field"]
                    c["construction"]["change_summary"] = action["reason"]
                else:
                    c["construction"]["source_expected_answer"] = action["source_expected_answer"]
                    if action.get("source_quote") and not any(normalized(action["source_quote"]) in normalized(s["text"]) for s in c["source_evidence"]):
                        raise ValueError("External wording repair requires its inspected full source quote")
        cases.append(c)

    withheld_articles = set()
    for action in decisions["external_replacements"]:
        c = next(c for c in cases if c["origin"] == {"file": "authored_cases.jsonl", "row": action["row"]})
        source = uploaded[action["chunk_id"]]
        physical = source["metadata"]["article_id"]
        canonical = source["metadata"]["canonical_article_id"]
        if normalized(action["evidence_quote"]) not in normalized(source["text"]):
            raise ValueError("Replacement external source quote not found")
        if any(i.split("_chunk_")[0] == physical for i in texts):
            raise ValueError("Newly withheld external source also appears in an embedded context")
        if any(x["source_article_id"] == canonical for x in cases):
            raise ValueError("New external source is already a benchmark source")
        article_chunks = [v for v in corpus if v["metadata"].get("canonical_article_id") == canonical or v["metadata"].get("article_id") == physical]
        # Preserve every available chunk of the withheld article as evidence,
        # never as generation context. No new source is downloaded.
        patch = {"case_id": "p3_external_" + stable_hash({"source": canonical, "question": action["question"]})[:20],
                 "base_question_id": "p3_new_" + stable_hash(action["question"])[:24],
                 "source_question_id": None, "source_article_id": canonical, "question": action["question"],
                 "source_gold_chunk_ids": [action["chunk_id"]],
                 "source_evidence": [{"chunk_id": v["id"], "text": v["text"]} for v in article_chunks],
                 "construction": {"source": "locally_withheld_Phase3_article", "source_expected_answer": action["source_expected_answer"],
                                  "source_evidence_spans": [{"text": action["evidence_quote"]}],
                                  "withheld_physical_article_id": physical, "retired_proposal": copy.deepcopy(c["construction"])}}
        change(c, action, patch)
        c["review"]["content_status"] = "reviewed"
        c["review"]["notes"] = ["Original proposal retired: " + action["reason"], "Replacement question and full supplied source reviewed; global withholding is applied before every policy's retrieval."]
        c["review"]["local_anchor_scan"] = {"pattern": action["scan_pattern"], "hit_chunk_ids": [v["id"] for v in corpus if re.search(action["scan_pattern"], v["text"], re.I)],
                                               "notice": "Specific entity/relation screen of every supplied chunk; retrieval still needs checking."}
        withheld_articles.update((physical, canonical))

    for action in decisions.get("retrieval_reclassifications", []):
        c = cases[action["row"] - 1]
        if c["case_type"] != "natural_retrieval_miss":
            raise ValueError("Only observed natural-miss candidates may be reclassified here")
        change(c, action, {"case_id": "p3_retrieved_control_" + stable_hash({"origin": c["origin"], "question": c["question"]})[:20],
                           "case_type": "answerable_control", "answerability_label": "answerable", "scope": "full_corpus",
                           "provided_context_chunk_ids": [], "contexts": [], "gold_relevant_chunk_ids": c["source_gold_chunk_ids"]})
        c["review"]["notes"].append("Observed retrieval contains the answer; retain the question as a control rather than manufacturing a miss to satisfy the original target quota.")

    for action in decisions.get("retrieved_control_misses", []):
        c = cases[action["row"] - 1]
        source = uploaded[action["source_chunk_id"]]
        if c["case_type"] != "answerable_control" or normalized(action["source_quote"]) not in normalized(source["text"]):
            raise ValueError("Control-to-miss correction needs an actual answering source passage")
        # A name in the retrieved passage does not prove the requested relation.
        # Keep the ID/question unchanged; no evidence is injected or removed.
        change(c, action, {"case_type": "natural_retrieval_miss", "answerability_label": "insufficient_evidence",
                           "scope": "provided_context", "gold_relevant_chunk_ids": [],
                           "source_gold_chunk_ids": [source["id"]],
                           "source_evidence": [{"chunk_id": source["id"], "text": source["text"]}]})
        c["review"]["notes"].append(action["reason"])

    for c in cases:
        replacement_rows = {a["row"] for a in (decisions["removed_replacements"] if c["origin"]["file"] == "review_queue.jsonl" else decisions["external_replacements"])}
        for hold in ([] if c["origin"]["row"] in replacement_rows else decisions["holds"] if c["origin"]["file"] == "review_queue.jsonl" else decisions["external_holds"]):
            if c["origin"]["row"] == hold["row"]:
                c["review"]["content_status"] = "hold"
                c["review"]["notes"].append(hold["reason"])
        if not c["review"]["notes"]:
            if c["case_type"] == "answerable_control":
                note = f"Source answer reviewed: {c['ground_truth']!r}. Actual generation contexts must still support it."
            elif c["case_type"] == "natural_retrieval_miss":
                note = "Source contains the intended fact. Recheck the current question with genuine selected retrieval; edited/old traces do not establish a natural miss."
            elif c["scope"] == "provided_context":
                note = "Compare the exact embedded passages with the withheld source evidence; topic overlap is intentional, but the requested fact must be absent."
            else:
                note = "Article removal is a construction operation, not proof of fact absence. Check both canonical and physical IDs and alternative-source answers."
            c["review"]["notes"].append(note)
        c["revision"] = stable_hash({k: c[k] for k in ("question", "ground_truth", "contexts", "source_article_id", "excluded_article_ids", "excluded_chunk_ids")})
    validate_cases(cases)

    index = {}
    for c in corpus:
        if not isinstance(c.get("id"), str) or not c["id"] or c["id"] in index or not isinstance(c.get("text"), str) or not c["text"].strip():
            raise ValueError("Empty/duplicate/invalid corpus row")
        if not c.get("metadata", {}).get("article_id"):
            raise ValueError("Missing corpus article identity")
        index[c["id"]] = c
    missing = sorted(set(texts) - set(index))
    different = sorted(i for i in set(texts) & set(index) if texts[i] != index[i]["text"])
    report = {"cases": len(cases), "content_status": dict(Counter(c["review"]["content_status"] for c in cases)),
              "changed_cases": sum(bool(c["review"]["changes"]) for c in cases), "corpus_rows": len(corpus),
              "global_withheld_article_ids": sorted(withheld_articles),
              "active_corpus_rows": sum(not (withheld_articles & {v["metadata"].get("article_id"), v["metadata"].get("canonical_article_id")}) for v in corpus),
              "embedded_texts": len(texts), "embedded_ids_absent_from_uploaded_corpus": missing,
              "embedded_ids_with_different_uploaded_text": different,
              "notice": "Structural validation passed; NOT semantic approval or permission to generate. Provided contexts retain embedded texts. Full-corpus cases require fresh retrieval; no historical testset/corpus requirement.",
              "ready_for_generation": False}
    inputs = {name: file_hash(bundle / name) for name in ("review_queue.jsonl", "review_queue_readable.json", "corpus_overlays.jsonl", "authored_cases.jsonl")}
    inputs.update(decisions=file_hash(decisions_path), corpus=file_hash(corpus_path))
    if source_path.exists():
        inputs["external_source_evidence"] = file_hash(source_path)
    write_jsonl(output / "cases.jsonl", cases)
    atomic_json(output / "manifest.json", {"schema_version": 1, "inputs": inputs, "cases_sha256": file_hash(output / "cases.jsonl"), "reviewer": "AI; no human review", **report})
    audit = ["# Phase3 — one-by-one data review", "", report["notice"], "",
             "Original source answers on negatives are construction metadata, not model targets. Each record below retains its own evidence and findings in cases.jsonl.", ""]
    for c in cases:
        o = c["origin"]
        audit += [f"## {o['file']} row {o['row']} — {c['case_type']} ({c['partition']})", "",
                  f"ID: `{c['case_id']}`", "", c["question"], "",
                  f"Content: {c['review']['content_status']}; runtime: {c['review']['runtime_status']}.", "",
                  *[n + "\n" for n in c["review"]["notes"]],
                  *["Repair: " + a["reason"] + "\n" for a in c["review"]["changes"]],
                  "Source: " + ", ".join(c["source_gold_chunk_ids"] or ["retained external answer spans only"]), "",
                  "Generation context: " + (", ".join(v["id"] for v in c["contexts"]) or "fresh retrieval required"), ""]
    (output / "CASE_REVIEW.md").write_text("\n".join(audit), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
