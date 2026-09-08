"""Small, deterministic Phase3 comparisons. No RAGAS, LLM judge or numpy."""
from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter

ABSTENTION = "I cannot find this information in the provided context."
B0_PROMPT = " ".join("""Answer using only the numbered context. Give one short answer sentence
containing only the information type requested by the question, such as
a person, place, date, year, number, object, event, or explanation.
Include multiple items only when the question explicitly asks for them.
Do not repeat the question, list alternative answers, or add background
details. Place a supporting citation [n] immediately after the answer.
If the answer is not in the context, say: 'I cannot find this information
in the provided context.'""".split())
B1_PROMPT = """You answer only from the numbered contexts. Return exactly one JSON object.
If the contexts support one defensible answer, return:
{"answerability":"answerable","answer":"concise answer","citations":[1]}
If they do not, return:
{"answerability":"insufficient_evidence","answer":null,"citations":[]}
Do not use outside knowledge. Do not repair a false premise. Citations are one-based context numbers."""


def parse_response(text, policy, context_count):
    if policy == "B0":
        from newsqa_rag.response_parsing import split_citation_indices
        citations, invalid = split_citation_indices(text, context_count)
        if text.strip().strip("\"'").strip() == ABSTENTION:
            return {"answerability": "insufficient_evidence", "answer": None, "citations": [], "raw_text": text}
        return {"answerability": "answerable", "answer": text.strip(), "citations": citations, "invalid_citations": invalid, "raw_text": text}
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.S | re.I)
    value = json.loads(fenced.group(1) if fenced else cleaned)
    if not isinstance(value, dict) or value.get("answerability") not in {"answerable", "insufficient_evidence"}:
        raise ValueError("Invalid answerability schema")
    citations = value.get("citations")
    if not isinstance(citations, list) or any(type(n) is not int or not 1 <= n <= context_count for n in citations):
        raise ValueError("Invalid citations")
    if value["answerability"] == "insufficient_evidence":
        if value.get("answer") is not None or citations:
            raise ValueError("Abstention requires null answer and empty citations")
    elif not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise ValueError("Empty answer")
    return {"answerability": value["answerability"], "answer": value.get("answer"), "citations": citations, "raw_text": text}


def normalize(text):
    # Same punctuation/whitespace convention as the selected project's QA metric.
    return " ".join(re.sub(r"[^\w\s]", "", text.lower().strip()).split())


def qa_scores(prediction, answers):
    from newsqa_rag.response_parsing import extract_answer_text
    predicted = normalize(extract_answer_text(prediction or ""))
    em, f1 = 0.0, 0.0
    for answer in answers:
        gold = normalize(answer)
        em = max(em, float(predicted == gold))
        p, g = Counter(predicted.split()), Counter(gold.split())
        common = sum((p & g).values())
        score = 2 * common / (sum(p.values()) + sum(g.values())) if p and g else 0.0
        f1 = max(f1, score)
    return em, f1


def per_case(case, prediction):
    ok = prediction.get("status") == "success"
    negative = case["answerability_label"] == "insufficient_evidence"
    abstained = ok and prediction.get("answerability") == "insufficient_evidence"
    # Failed requests never receive a free correct-abstention label.
    error = not ok or (not abstained if negative else abstained)
    em, f1 = qa_scores(prediction.get("answer") if ok and not abstained else "", case["accepted_answers"]) if not negative else (None, None)
    citations = prediction.get("citations") or []
    total_refs = len(set(citations)) + len(prediction.get("invalid_citations") or [])
    valid = len(set(citations)) / total_refs if ok and not abstained and total_refs else 0.0
    flags = []
    if not ok:
        flags.append("generation_failed")
    elif negative and not abstained:
        flags.append("counterfactual_non_abstention_review_correction" if case["case_type"] == "counterfactual" else "answered_negative")
    elif not negative and abstained:
        flags.append("false_abstention")
    if ok and not abstained and valid < 1:
        flags.append("missing_or_invalid_citation")
    if not negative and f1 == 0:
        flags.append("no_reference_token_overlap")
    return {"case_id": case["case_id"], "case_type": case["case_type"], "question": case["question"],
            "negative": negative, "success": ok, "abstained": abstained, "decision_error": error,
            "em": em, "f1": f1, "citation_valid": valid, "flags": flags,
            "answer": prediction.get("answer"), "raw_text": prediction.get("raw_text"),
            "notice": "Citation validity checks index syntax, not factual entailment. Flags are not hallucination judgments."}


def summarize(cases, predictions):
    if set(predictions) != {c["case_id"] for c in cases}:
        raise ValueError("Scoring needs exactly one result (including failures) per case")
    rows = [per_case(c, predictions[c["case_id"]]) for c in cases]
    controls = [r for r in rows if not r["negative"]]
    negatives = [r for r in rows if r["negative"]]
    tp = sum(r["success"] and r["abstained"] for r in negatives)
    fp = sum(r["decision_error"] for r in controls)
    fn = len(negatives) - tp
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"n": len(rows), "generation_success_rate": avg([r["success"] for r in rows]),
            "false_answer_rate": avg([r["decision_error"] for r in negatives]),
            "false_abstention_rate": avg([r["decision_error"] for r in controls]),
            "abstention_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "control_em": avg([r["em"] for r in controls]), "control_f1": avg([r["f1"] for r in controls]),
            "citation_validity": avg([r["citation_valid"] for r in controls]),
            "by_type": {t: {"n": len(rs), "decision_errors": sum(r["decision_error"] for r in rs),
                            "generation_failures": sum(not r["success"] for r in rs)}
                        for t in sorted({r["case_type"] for r in rows}) if (rs := [r for r in rows if r["case_type"] == t])},
            "failure_convention": "A failure counts as a decision error for its gold class; separate failure counts are also reported."}


def apply_gate(cases, predictions, traces, threshold):
    result = copy.deepcopy(predictions)
    for case in cases:
        row, trace = result[case["case_id"]], traces[case["case_id"]]
        score = trace.get("top1_reranker_score")
        applicable = case["scope"] == "full_corpus" and type(score) in (float, int) and math.isfinite(score)
        rejected = applicable and score < threshold and row.get("status") == "success"
        row["gate"] = {"applicable": applicable, "rejected": rejected, "score": score, "threshold": threshold}
        if rejected:
            row.update(answerability="insufficient_evidence", answer=None, citations=[], invalid_citations=[])
    return result


def calibrate(cases, predictions, traces):
    if any(c["partition"] != "development" for c in cases):
        raise ValueError("Threshold selection must use development only")
    scores = sorted({float(traces[c["case_id"]]["top1_reranker_score"]) for c in cases
                     if c["scope"] == "full_corpus" and type(traces[c["case_id"]].get("top1_reranker_score")) in (float, int)})
    if not scores or not all(math.isfinite(s) for s in scores):
        return {"status": "infeasible", "threshold": None, "curve": []}
    candidates = [math.nextafter(scores[0], -math.inf), *scores, math.nextafter(scores[-1], math.inf)]
    curve = [{"threshold": t, **summarize(cases, apply_gate(cases, predictions, traces, t))} for t in candidates]
    eligible = [r for r in curve if r["false_abstention_rate"] <= .10]
    best = min(eligible, key=lambda r: (r["false_answer_rate"], -r["abstention_f1"], r["threshold"])) if eligible else None
    return {"status": "selected" if best else "infeasible", "threshold": best["threshold"] if best else None, "curve": curve}


def select_policy(summaries):
    b0 = summaries["B0"]
    reasons = {}
    for name, s in summaries.items():
        reasons[name] = [reason for failed, reason in (
            (s["generation_success_rate"] < .98, "generation_success_below_98_percent"),
            (s["false_abstention_rate"] > .10, "false_abstention_above_10_percent"),
            (s["control_f1"] < b0["control_f1"] - .02, "control_f1_drop"),
            (s["citation_validity"] < b0["citation_validity"] - .01, "citation_validity_drop")) if failed]
    eligible = [name for name in summaries if not reasons[name]]
    best = min((summaries[n]["false_answer_rate"] for n in eligible), default=None)
    winner = min((n for n in eligible if summaries[n]["false_answer_rate"] <= best + .02), default=None)
    return {"winner": winner, "status": "selected" if winner else "infeasible", "ineligibility_reasons": reasons}
