"""One bounded post-review amendment; original Phase3 artifacts remain intact.

Repair the wildfire control/ablation/weak question together. Reuse historical
outcomes only for identical requests, including the four genuine failures.
New control retrieval still requires an actual evidence review before freezing.
This is not a fresh held-out test and never retunes from final responses.
"""
from __future__ import annotations

import argparse
import copy
import json
from getpass import getpass
from pathlib import Path

import phase3_run as runner
from phase3_api import QuotaLedger, RunPaused, atomic_json, request_payload, stable_hash
from review_phase3 import file_hash, read_jsonl, validate_cases, write_jsonl

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "revision_config.json"
ROWS = {13, 63, 87}
QUESTION = "What was found in a canyon east of San Diego in the path of the Harris Fire?"
REASON = ("Post-review ambiguity correction: identify the canyon/Harris Fire finding, "
          "not any wildfire damage or shelter amenity. Apply the same question to the "
          "control, ablation and weak-evidence variants. Source answer and labels stay unchanged.")


def amended_cases(original):
    cases = copy.deepcopy(original)
    changed = []
    for case in cases:
        if case["origin"]["file"] != "review_queue.jsonl" or case["origin"]["row"] not in ROWS:
            continue
        source = next(v["text"] for v in case["source_evidence"] if v["chunk_id"] == "0a39cd0d1263_chunk_0")
        if not all(phrase in source for phrase in ("Harris Fire", "four people", "canyon east of San Diego")):
            raise ValueError("Wildfire amendment needs the retained source evidence")
        before = case["question"]
        if before != "What was found near San Diego following the California wildfires?":
            raise ValueError("Unexpected original wildfire question")
        case["question"] = QUESTION
        case["review"]["changes"].append({"reason": REASON, "before": {"question": before},
            "after": {"question": QUESTION}, "evidence_chunk_id": "0a39cd0d1263_chunk_0",
            "quote": "Four bodies were found in a canyon in the path of the blaze Thursday."})
        case["review"]["notes"].append(REASON)
        case["revision"] = stable_hash({k: case[k] for k in (
            "question", "ground_truth", "contexts", "source_article_id", "excluded_article_ids", "excluded_chunk_ids")})
        changed.append(case["case_id"])
    if len(changed) != 3:
        raise ValueError("Expected exactly three linked wildfire cases")
    validate_cases(cases)
    return cases, changed


def keep_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RunPaused(f"Preserve existing revision artifact: {path}")
    else:
        atomic_json(path, value)


def prepare_revision():
    original_settings = runner.load_settings()
    original, old_traces, _ = runner.frozen_inputs(original_settings)
    settings = runner.load_settings(CONFIG)
    if settings["work"] == original_settings["work"] or settings["reviewed"] == original_settings["reviewed"]:
        raise ValueError("Revision must not overwrite original inputs or results")
    cases, changed = amended_cases(original)
    folder, work = settings["reviewed"], settings["work"]
    folder.mkdir(parents=True, exist_ok=True)
    case_path = folder / "cases.jsonl"
    if case_path.exists():
        if read_jsonl(case_path) != cases:
            raise RunPaused("Revised cases changed; do not overwrite a frozen experiment")
    else:
        write_jsonl(case_path, cases)
    manifest = json.loads((original_settings["reviewed"] / "manifest.json").read_text())
    manifest.update(cases_sha256=file_hash(case_path), changed_cases=sum(bool(c["review"]["changes"]) for c in cases),
        amendment={"parent_cases_sha256": file_hash(original_settings["reviewed"] / "cases.jsonl"),
                   "changed_case_ids": changed, "reason": REASON,
                   "evaluation_status": "post-review revision, not a new untouched final test"})
    keep_json(folder / "manifest.json", manifest)
    notes = ["# Phase3 revised case review", "", REASON, "",
             "AI evidence review, not independent human certification. Original reviews and source evidence remain in cases.jsonl.", ""]
    for c in cases:
        notes += [f"## {c['origin']['file']} row {c['origin']['row']} — {c['case_type']}", "",
                  c["question"], "", *[n + "\n" for n in c["review"]["notes"]]]
    text = "\n".join(notes)
    review_path = folder / "CASE_REVIEW.md"
    if review_path.exists() and review_path.read_text() != text:
        raise RunPaused("Existing case review differs")
    if not review_path.exists():
        review_path.write_text(text, encoding="utf-8")
    # The selected corpus/index are unchanged. Share the verified read-only index.
    for name in ("bge_m3.pkl", "index_manifest.json"):
        path = work / name
        if not path.exists():
            path.symlink_to(Path("..") / name)
        if file_hash(path) != file_hash(original_settings["work"] / name):
            raise ValueError("Revision index does not match original corpus")
    trace_path = work / "retrievals.json"
    identity = runner.retrieval_identity(settings)
    if trace_path.exists():
        if json.loads(trace_path.read_text())["identity"] != identity:
            raise RunPaused("Revision retrieval identity changed")
    else:
        traces = {}
        for c in cases:
            if c["case_id"] in changed and c["scope"] == "full_corpus":
                continue  # Changed query must receive fresh selected retrieval.
            traces[c["case_id"]] = {**old_traces[c["case_id"]], "case_revision": c["revision"]}
        atomic_json(trace_path, {"identity": identity, "traces": traces})
    review_path = work / "retrieval_review.jsonl"
    if not review_path.exists():
        write_jsonl(review_path, [v for v in read_jsonl(original_settings["work"] / "retrieval_review.jsonl")
                                 if v["case_id"] not in changed])
    return {"cases": 200, "changed_cases": len(changed), "content_status": {"reviewed": 200},
            "corpus_rows": manifest["corpus_rows"], "ready_for_generation": (work / "frozen.json").exists(),
            "notice": "Changed control needs fresh retrieval and evidence review; original failures are retained."}


def generation_identity(settings, frozen):
    return stable_hash({"frozen": frozen, "generator": runner.GENERATOR,
        "B0": runner.baseline_prompt(settings["prompt_registry"]), "B1": runner.B1_PROMPT,
        "prompt_registry_sha256": file_hash(settings["prompt_registry"]),
        "code": {p.name: file_hash(p) for p in (HERE / "phase3_metrics.py", HERE / "phase3_api.py", HERE / "phase3_run.py")}})


def payload(case, trace, prompt, settings):
    numbered = "\n\n".join(f"[{i}] {v['text']}" for i, v in enumerate(trace["contexts"], 1))
    return request_payload(settings, prompt, f"Context:\n{numbered}\n\nQuestion: {case['question']}")


def reusable_requests(cases, traces, old_cases, old_traces, settings):
    records = []
    for c in cases:
        cid = c["case_id"]
        for policy, prompt in (("B0", runner.B0_PROMPT), ("B1", runner.B1_PROMPT)):
            old = payload(old_cases[cid], old_traces[cid], prompt, settings)
            new = payload(c, traces[cid], prompt, settings)
            records.append({"case_id": cid, "policy": policy, "reused": old == new,
                            "original_payload_sha256": stable_hash(old), "payload_sha256": stable_hash(new)})
    return records


def seed_unchanged_predictions():
    settings = runner.load_settings(CONFIG)
    cases, traces, frozen = runner.frozen_inputs(settings)
    original_settings = runner.load_settings()
    original, old_traces, _ = runner.frozen_inputs(original_settings)
    old_cases = {c["case_id"]: c for c in original}
    old_results = [json.loads((original_settings["work"] / stage / "results.json").read_text())
                   for stage in ("development", "final")]
    old_run = old_results[0]["identity"]
    if old_results[1]["identity"] != old_run:
        raise ValueError("Original stage identities differ")
    audit = json.loads((original_settings["work"] / "label_audit/reuse_audit.json").read_text())
    known_hashes = {(v["case_id"], v["policy"]): v["payload_sha256"] for v in audit["request_payloads_checked"]}
    if audit["new_run"] != old_run:
        raise ValueError("Original generation provenance does not match")
    records = reusable_requests(cases, traces, old_cases, old_traces, settings)
    if sum(v["reused"] for v in records) != 394:
        raise ValueError("Expected 394 identical requests and six changed requests")
    for v in records:
        if known_hashes[v["case_id"], v["policy"]] != v["original_payload_sha256"]:
            raise ValueError("Original request hash changed")
    new_run = generation_identity(settings, frozen)
    ledger = QuotaLedger(HERE / "api_usage.sqlite", rpm=settings["requests_per_minute"],
                         daily=settings["requests_per_day"], reserve=settings["daily_safety_reserve"])
    predictions = {(cid, pol): pred for r in old_results for pol in ("B0", "B1")
                   for cid, pred in r["predictions"][pol].items()}
    for v in records:
        if not v["reused"]:
            continue
        cid, pol = v["case_id"], v["policy"]
        pred = predictions[cid, pol]
        if ledger.prediction(old_run, cid, pol) != pred:
            raise ValueError("Original prediction/ledger mismatch")
        existing = ledger.prediction(new_run, cid, pol)
        if existing is None:
            ledger.save_prediction(new_run, cid, pol, pred)
        elif existing != pred:
            raise RunPaused("Conflicting revised prediction; do not overwrite")
    result = {"original_run": old_run, "revision_run": new_run, "reused_outcomes": 394,
              "changed_requests": 6, "retained_failures": 4, "request_checks": records,
              "notice": "Exact-payload reuse includes genuine failures; no historical requests were reset or fabricated. Post-review comparison, not a new untouched final test."}
    keep_json(settings["work"] / "reuse_audit.json", result)
    return {k: v for k, v in result.items() if k != "request_checks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "seed", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare_revision(), indent=2))
    elif args.action == "seed":
        print(json.dumps(seed_unchanged_predictions(), indent=2))
    else:
        seed_unchanged_predictions()
        key = getpass("Current Gemini API key (hidden): ")
        try:
            for stage in ("smoke", "development", "final"):
                print(json.dumps(runner.run(runner.load_settings(CONFIG), stage, key), indent=2))
        finally:
            key = None


if __name__ == "__main__":
    main()
