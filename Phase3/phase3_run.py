"""Local 14a/14b entry points. All networking is explicit Gemini generation.

Selected BGE retrieval runs offline on the supplied Phase3 corpus. A repaired
draft is not an approved benchmark; freeze requires case-specific AI evidence
decisions on the resulting traces. No old testset, reserve or human signature.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))

from phase3_api import QuotaLedger, RunPaused, atomic_json, generate_one, request_payload, stable_hash
from phase3_metrics import B0_PROMPT, B1_PROMPT, apply_gate, calibrate, parse_response, per_case, select_policy, summarize
from review_phase3 import file_hash, read_jsonl, validate_cases, write_jsonl

MODELS = ("BAAI/bge-m3", "BAAI/bge-reranker-large")
GENERATOR = {"model": "gemini-3.1-flash-lite", "temperature": 0.0, "max_tokens": 512, "reasoning_effort": "minimal"}


def load_settings(path=HERE / "local_config.json"):
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    for key in ("corpus", "reviewed", "work", "prompt_registry"):
        settings[key] = (path.parent / settings[key]).resolve()
    if any(settings.get(k) != v for k, v in GENERATOR.items()) or settings.get("max_attempts") != 3:
        raise ValueError("Preserve the selected generator and three total attempts")
    for key in ("encoder_batch_size", "reranker_batch_size"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError("Batch sizes must be positive integers")
    if settings.get("device") not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Choose cpu, cuda or cuda:0")
    if settings.get("matmul_precision") not in ("high", "highest"):
        raise ValueError("Choose high (CUDA TensorFloat32) or highest internal matmul precision")
    baseline_prompt(settings["prompt_registry"])
    rpm, daily, reserve = (settings[k] for k in ("requests_per_minute", "requests_per_day", "daily_safety_reserve"))
    if type(rpm) is not int or not 1 <= rpm <= 15 or type(daily) is not int or not 1 <= daily <= 500 or type(reserve) is not int or not 0 <= reserve < daily:
        raise ValueError("Invalid local request ceilings/reserve")
    return settings


def baseline_prompt(path):
    # Deliberately read only the supplied registry's simple folded p2 block.
    # This avoids downloading PyYAML into the local kernel and fails closed on
    # another YAML representation instead of pretending to be a general parser.
    text = Path(path).read_text(encoding="utf-8")
    sections = re.findall(r"^  p2:\s*\n((?:(?!^  \S).|\n)*)", text, re.M)
    if len(sections) != 1:
        raise ValueError("Expected exactly one p2 entry in the supplied Phase2 prompt registry")
    blocks = re.findall(r"^    system_prompt: >-\s*\n((?:^      [^\n]*\n?)+)", sections[0], re.M)
    if len(blocks) != 1:
        raise ValueError("Expected the supplied six-space-indented folded p2 system_prompt")
    prompt = " ".join(blocks[0].split())
    if prompt != B0_PROMPT:
        raise ValueError("p2 differs from the selected Phase2 baseline; do not silently change B0")
    return prompt


def model_cached(model):
    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub")))
    snapshots = cache / ("models--" + model.replace("/", "--")) / "snapshots"
    for snapshot in snapshots.glob("*"):
        weights = any(p.is_file() and p.stat().st_size > 0 for name in ("model.safetensors", "pytorch_model.bin") if (p := snapshot / name))
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = snapshot / name
            if index.is_file():
                try:
                    shards = set(json.loads(index.read_text())["weight_map"].values())
                    weights = weights or bool(shards) and all((snapshot / s).is_file() and (snapshot / s).stat().st_size > 0 for s in shards)
                except (ValueError, KeyError):
                    pass
        # Dense-only BGE-M3 caches are insufficient: FlagEmbedding otherwise
        # initializes missing sparse/ColBERT heads, changing the retriever.
        heads = model != MODELS[0] or all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
                                        for name in ("sparse_linear.pt", "colbert_linear.pt"))
        if weights and heads and (snapshot / "config.json").is_file() and (snapshot / "tokenizer_config.json").is_file():
            return True
    return False


def draft(settings):
    folder = settings["reviewed"]
    cases = read_jsonl(folder / "cases.jsonl")
    manifest = json.loads((folder / "manifest.json").read_text())
    validate_cases(cases)
    if manifest["cases_sha256"] != file_hash(folder / "cases.jsonl") or manifest["inputs"]["corpus"] != file_hash(settings["corpus"]):
        raise ValueError("Reviewed draft/corpus changed: rerun 14a review preparation")
    if manifest["inputs"]["decisions"] != file_hash(HERE / "review_decisions.json"):
        raise ValueError("Repair decisions changed: rerun 14a review preparation")
    if manifest["inputs"].get("external_source_evidence") != (file_hash(HERE / "external_source_evidence.jsonl") if (HERE / "external_source_evidence.jsonl").exists() else None):
        raise ValueError("External source evidence changed: rerun 14a review preparation")
    # These are current input identities, not a hardcoded historical-corpus gate.
    return cases


def active_corpus(settings):
    manifest = json.loads((settings["reviewed"] / "manifest.json").read_text())
    withheld = set(manifest.get("global_withheld_article_ids", []))
    return [row for row in read_jsonl(settings["corpus"])
            if not withheld & {row.get("metadata", {}).get("article_id"), row.get("metadata", {}).get("canonical_article_id")}]


def retrieval_identity(settings):
    manifest = json.loads((settings["reviewed"] / "manifest.json").read_text())
    return {"corpus": file_hash(settings["corpus"]), "cases": file_hash(settings["reviewed"] / "cases.jsonl"),
            "global_withheld_article_ids": manifest.get("global_withheld_article_ids", []),
            "models": list(MODELS), "top_k": 20, "depth": 5, "device": settings["device"],
            "matmul_precision": settings["matmul_precision"],
            "code": {str(p.relative_to(ROOT)): file_hash(p) for p in (HERE / "phase3_run.py", ROOT / "common/newsqa_rag/indexing/learned_sparse_index.py", ROOT / "common/newsqa_rag/retrieval/reranker.py")}}


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=0)
    try:
        try:
            db.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError:
            raise RunPaused("Another local Phase3 process is active; wait for it to finish") from None
        yield
    finally:
        db.rollback()
        db.close()


def excluded(case, row):
    meta = row.get("metadata") or {}
    excluded_articles = set(case.get("excluded_article_ids") or [])
    if case["case_type"] == "removed_article":
        excluded_articles.add(case["source_article_id"])
        excluded_articles.update(i.split("_chunk_")[0] for i in case["source_gold_chunk_ids"])
    return row["id"] in set(case.get("excluded_chunk_ids") or []) or bool(excluded_articles & {meta.get("article_id"), meta.get("canonical_article_id")})


def collect_retrievals(settings):
    problems = preflight(settings, "retrieve")["errors"]
    if problems:
        raise RunPaused("; ".join(problems))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from newsqa_rag.indexing.learned_sparse_index import BGEM3SparseEncoder, LearnedSparseIndex
    from newsqa_rag.retrieval.reranker import BGESequenceClassificationReranker
    torch.set_float32_matmul_precision(settings["matmul_precision"])
    class LocalBatchedEncoder(BGEM3SparseEncoder):
        # Keep the selected encoder and GPU mini-batch, but feed 128 documents
        # per call. FlagEmbedding otherwise repeats its sizing forward pass
        # for every four documents in the shared index's build loop.
        def encode(self, texts, batch_size=32):
            result = super().encode(texts, min(len(texts), settings["encoder_batch_size"]))
            print(f"Encoded {len(texts)} chunks", flush=True)
            return result
    work = settings["work"]
    with exclusive(work / "process.lock.sqlite"):
        cases = draft(settings)
        rows = active_corpus(settings)
        chunks = {c["id"]: c for c in rows}
        identity = retrieval_identity(settings)
        index_path = work / "bge_m3.pkl"
        index_manifest = work / "index_manifest.json"
        binding = {"corpus": identity["corpus"], "global_withheld_article_ids": identity["global_withheld_article_ids"], "model": MODELS[0],
                   "matmul_precision": settings["matmul_precision"]}
        if index_path.exists():
            metadata = json.loads(index_manifest.read_text()) if index_manifest.exists() else {}
            if metadata.get("binding") != binding or metadata.get("sha256") != file_hash(index_path):
                raise RunPaused("Cached index belongs to other inputs or is damaged; preserve it and choose a fresh work directory")
            index = LearnedSparseIndex.load(str(index_path), device=settings["device"])
            if index.model_name != MODELS[0] or index.size != len(rows):
                raise ValueError("Wrong selected sparse index")
        else:
            index = LearnedSparseIndex(MODELS[0], device=settings["device"], encoder=LocalBatchedEncoder(MODELS[0], settings["device"]))
            print(f"Building selected BGE-M3 index for {len(rows)} actual Phase3 chunks (offline).", flush=True)
            index.build(rows, batch_size=128)
            index.save(str(index_path), metadata=binding)
            atomic_json(index_manifest, {"binding": binding, "sha256": file_hash(index_path)})
        reranker = BGESequenceClassificationReranker(MODELS[1], device=settings["device"], batch_size=settings["reranker_batch_size"])
        trace_path = work / "retrievals.json"
        stored = json.loads(trace_path.read_text()) if trace_path.exists() else {"identity": identity, "traces": {}}
        if stored["identity"] != identity:
            raise RunPaused("Retrieval inputs changed; preserve previous runs and choose a fresh work directory")
        for number, case in enumerate(cases, 1):
            ident = case["case_id"]
            if ident in stored["traces"]:
                continue
            if case["scope"] == "provided_context" and case["case_type"] != "natural_retrieval_miss":
                trace = {"mode": "provided_context", "contexts": case["contexts"], "top1_reranker_score": None, "retrieved_chunks": []}
            else:
                # Remove articles BEFORE the top-20 cut. Asking for all scored hits
                # prevents an excluded source monopolizing the candidate window.
                hits = index.query(case["question"], top_k=len(rows) if case["case_type"] == "removed_article" else 20)
                candidates = [{**chunks[h["id"]], "score": h["score"]} for h in hits if not excluded(case, chunks[h["id"]])][:20]
                ranked = reranker.rerank(case["question"], candidates, top_n=5)
                if not ranked:
                    raise RunPaused(f"No retrieval for {ident}; do not fabricate an empty-context negative")
                trace = {"mode": "selected_retrieval", "contexts": ranked, "retrieved_chunks": candidates,
                         "top1_reranker_score": ranked[0]["reranker_score"],
                         "source_article_present": any(c.get("metadata", {}).get("canonical_article_id") == case["source_article_id"] for c in rows)}
            stored["traces"][ident] = {**trace, "case_revision": case["revision"]}
            atomic_json(trace_path, stored)
            print(f"Retrieval {number}/{len(cases)}: {ident}", flush=True)
        template = work / "retrieval_review_template.jsonl"
        write_jsonl(template, [{"case_id": c["case_id"], "trace_sha256": stable_hash(stored["traces"][c["case_id"]]),
                                "decision": "pending", "evidence_notes": "", "review_method": "AI inspection; no human review",
                                "evidence_basis_checked": False, "scope_verified": False}
                               for c in cases if c["review"]["runtime_status"] == "pending_retrieval"])
        return {"status": "retrieval_complete_pending_AI_evidence_review", "traces": len(stored["traces"])}


def checked_traces(settings, cases):
    stored = json.loads((settings["work"] / "retrievals.json").read_text())
    if stored["identity"] != retrieval_identity(settings):
        raise ValueError("Stale retrieval identity")
    traces = stored["traces"]
    if set(traces) != {c["case_id"] for c in cases}:
        raise ValueError("Expected exactly 200 case-specific retrievals")
    corpus = {c["id"]: c for c in active_corpus(settings)}
    for c in cases:
        t = traces[c["case_id"]]
        if t["case_revision"] != c["revision"] or not 1 <= len(t["contexts"]) <= 5:
            raise ValueError("Stale/empty case contexts")
        if len({v["id"] for v in t["contexts"]}) != len(t["contexts"]):
            raise ValueError("Duplicate generation context")
        if c["scope"] == "provided_context" and c["case_type"] != "natural_retrieval_miss":
            if t["mode"] != "provided_context" or t["contexts"] != c["contexts"] or t["top1_reranker_score"] is not None:
                raise ValueError("Provided contexts were substituted or given invented scores")
        else:
            if t["mode"] != "selected_retrieval" or type(t["top1_reranker_score"]) not in (int, float) or not math.isfinite(t["top1_reranker_score"]):
                raise ValueError("Missing actual finite reranker score")
            candidates = t["retrieved_chunks"]
            if not 1 <= len(candidates) <= 20 or len({v["id"] for v in candidates}) != len(candidates):
                raise ValueError("Invalid top-20 candidate set")
            if len(t["contexts"]) != min(5, len(candidates)) or not {v["id"] for v in t["contexts"]} <= {v["id"] for v in candidates}:
                raise ValueError("Generation contexts do not match the reranked candidate set")
            scores = [v.get("reranker_score") for v in t["contexts"]]
            if any(type(s) not in (int, float) or not math.isfinite(s) for s in scores) or scores != sorted(scores, reverse=True) or t["top1_reranker_score"] != scores[0]:
                raise ValueError("Invalid or inconsistent raw reranker scores")
            if c["case_type"] == "natural_retrieval_miss" and not any(v.get("metadata", {}).get("canonical_article_id") == c["source_article_id"] or v.get("metadata", {}).get("article_id") in {i.split("_chunk_")[0] for i in c["source_gold_chunk_ids"]} for v in corpus.values()):
                raise ValueError("A natural retrieval miss requires its source to exist in the active corpus")
            if c["case_type"] == "removed_article" and not any(excluded(c, v) for v in corpus.values()):
                raise ValueError("Removed-article case removes nothing from the active corpus")
            for v in t["contexts"] + t["retrieved_chunks"]:
                if v["id"] not in corpus or v["text"] != corpus[v["id"]]["text"] or excluded(c, corpus[v["id"]]):
                    raise ValueError("Retrieval contains missing/substituted/excluded corpus text")
    return traces


def freeze(settings):
    cases = draft(settings)
    holds = [c["case_id"] for c in cases if c["review"]["content_status"] == "hold"]
    if holds:
        raise RunPaused(f"{len(holds)} content/label holds remain; repair them with evidence before freezing")
    traces = checked_traces(settings, cases)
    review_path = settings["work"] / "retrieval_review.jsonl"
    decisions = read_jsonl(review_path)
    by_id = {d["case_id"]: d for d in decisions}
    pending = {c["case_id"] for c in cases if c["review"]["runtime_status"] == "pending_retrieval"}
    if set(by_id) != pending or len(decisions) != len(by_id):
        raise ValueError("Need one actual AI evidence assessment per pending case, not a copied approval flag")
    for ident, d in by_id.items():
        if d.get("decision") != "accept" or d.get("trace_sha256") != stable_hash(traces[ident]) or not d.get("evidence_notes", "").strip() or d.get("evidence_basis_checked") is not True or d.get("scope_verified") is not True:
            raise ValueError(f"Unresolved or stale source/scope evidence: {ident}")
    frozen = {"identity": retrieval_identity(settings), "traces_sha256": stable_hash(traces),
              "review_sha256": file_hash(review_path), "case_count": len(cases), "review_method": "AI, no human review"}
    path = settings["work"] / "frozen.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise RunPaused("Frozen experiment changed; preserve it and choose a new work directory")
    atomic_json(path, frozen)
    return frozen


def frozen_inputs(settings):
    cases = draft(settings)
    if any(c["review"]["content_status"] == "hold" for c in cases):
        raise ValueError("Content holds remain")
    traces = checked_traces(settings, cases)
    frozen = json.loads((settings["work"] / "frozen.json").read_text())
    if frozen["identity"] != retrieval_identity(settings) or frozen["traces_sha256"] != stable_hash(traces) or frozen["review_sha256"] != file_hash(settings["work"] / "retrieval_review.jsonl"):
        raise ValueError("Frozen inputs/review were changed")
    return cases, traces, frozen


def preflight(settings, stage="smoke"):
    errors, notes = [], []
    try:
        cases = draft(settings)
        held = [f"{c['origin']['file']}:{c['origin']['row']}" for c in cases if c["review"]["content_status"] == "hold"]
        notes.append(f"Draft: 200 slots; {len(held)} content holds: {', '.join(held)}")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"Prepare the reviewed bundle first: {exc}")
    if stage == "retrieve":
        if sys.version_info < (3, 11):
            errors.append("Selected retrieval project requires Python >=3.11; choose the existing Python 3.12 kernel")
        missing = [m for m in ("numpy", "torch", "transformers", "FlagEmbedding", "accelerate") if importlib.util.find_spec(m) is None]
        if missing:
            errors.append("Missing local retrieval packages: " + ", ".join(missing))
        absent = [m for m in MODELS if not model_cached(m)]
        if absent:
            errors.append("Selected model weights not found in local HF cache: " + ", ".join(absent))
    else:
        try:
            frozen_inputs(settings)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"Generation is not ready: {exc}")
    notes += ["No downloads, API calls or key needed for this check.", "15 RPM / 500/day are local user ceilings, not verified provider entitlement; TPM and other clients can still cause 429.", "B2 costs no additional generation calls. Final cannot select its own threshold/policy."]
    return {"status": "blocked" if errors else "ready", "stage": stage, "errors": errors, "notes": notes}


def stage_cases(cases, stage):
    development = [c for c in cases if c["partition"] == "development"]
    if stage == "smoke":
        return [next(c for c in development if c["case_type"] == t) for t in sorted({c["case_type"] for c in cases})]
    return development if stage == "development" else [c for c in cases if c["partition"] == "final_test"]


def run(settings, stage, api_key, *, transport=None):
    if stage not in ("smoke", "development", "final"):
        raise ValueError("Choose smoke, development or final")
    if not api_key or not api_key.strip():
        raise RunPaused("Supply the existing key privately via GEMINI_API_KEY or the notebook's hidden prompt")
    with exclusive(settings["work"] / "process.lock.sqlite"):
        all_cases, traces, frozen = frozen_inputs(settings)
        selected_prompt = baseline_prompt(settings["prompt_registry"])
        identity = stable_hash({"frozen": frozen, "generator": GENERATOR, "B0": selected_prompt, "B1": B1_PROMPT,
                                "prompt_registry_sha256": file_hash(settings["prompt_registry"]),
                                "code": {p.name: file_hash(p) for p in (HERE / "phase3_metrics.py", HERE / "phase3_api.py", HERE / "phase3_run.py")}})
        cases = stage_cases(all_cases, stage)
        output = settings["work"] / stage
        lock_path = settings["work"] / "development" / "decision.json"
        existing_result = output / "results.json"
        if existing_result.exists():
            previous = json.loads(existing_result.read_text())
            if previous["identity"] != identity:
                raise RunPaused("Completed stage belongs to a different experiment; preserve it and choose a new work directory")
            if stage != "development" or lock_path.exists():
                return {"stage": stage, "status": "already_complete", "summaries": previous["summaries"], "result_file": str(existing_result)}
        lock = None
        if stage == "final":
            lock = json.loads(lock_path.read_text())
            dev_path = settings["work"] / "development" / "results.json"
            if lock["identity"] != identity or lock["development_results_sha256"] != file_hash(dev_path):
                raise RunPaused("Development decision is stale or unbound; final is locked before any API call")
        ledger = QuotaLedger(HERE / "api_usage.sqlite", rpm=settings["requests_per_minute"], daily=settings["requests_per_day"], reserve=settings["daily_safety_reserve"])
        predictions = {"B0": {}, "B1": {}}
        # Smoke/development share exact request identities, so the smoke's 14
        # successful calls are reused. Full clean experiment = 400, not 414.
        for case in cases:
            ident = case["case_id"]
            contexts = traces[ident]["contexts"]
            numbered = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(contexts, 1))
            user_prompt = f"Context:\n{numbered}\n\nQuestion: {case['question']}"
            for policy, system in (("B0", selected_prompt), ("B1", B1_PROMPT)):
                row = ledger.prediction(identity, ident, policy)
                if row is None:
                    row = generate_one(ledger, identity, ident, policy, request_payload(settings, system, user_prompt),
                                       lambda text, p=policy: parse_response(text, p, len(contexts)), api_key,
                                       max_attempts=settings["max_attempts"], **({"transport": transport} if transport else {}))
                    row = {**row, "case_id": ident, "policy": policy, "context_chunk_ids": [c["id"] for c in contexts]}
                    ledger.save_prediction(identity, ident, policy, row)
                predictions[policy][ident] = row
            print(f"{stage}: {len(predictions['B0'])}/{len(cases)} cases complete", flush=True)
            atomic_json(output / "progress.json", {"identity": identity, "completed": len(predictions["B0"]), "expected": len(cases), "usage": ledger.usage(identity)})
        threshold = calibrate(cases, predictions["B1"], traces) if stage == "development" else (lock.get("threshold") if lock else None)
        if threshold and threshold["status"] == "selected":
            predictions["B2"] = apply_gate(cases, predictions["B1"], traces, threshold["threshold"])
        summaries = {p: summarize(cases, rows) for p, rows in predictions.items()}
        result = {"identity": identity, "stage": stage, "summaries": summaries, "predictions": predictions,
                  "per_case": {p: [per_case(c, rows[c["case_id"]]) for c in cases] for p, rows in predictions.items()},
                  "usage": ledger.usage(identity), "selected_development_policy": lock["policy"] if lock else None}
        comparisons = []
        for case in cases:
            baseline = per_case(case, predictions["B0"][case["case_id"]])
            for policy in predictions.keys() - {"B0"}:
                candidate = per_case(case, predictions[policy][case["case_id"]])
                comparisons.append({"case_id": case["case_id"], "case_type": case["case_type"], "question": case["question"],
                                    "policy": policy, "B0_answer": baseline["answer"], "candidate_answer": candidate["answer"],
                                    "control_f1_delta": candidate["f1"] - baseline["f1"] if baseline["f1"] is not None else None,
                                    "new_decision_error": candidate["decision_error"] and not baseline["decision_error"],
                                    "new_generation_failure": baseline["success"] and not candidate["success"],
                                    "citation_validity_delta": candidate["citation_valid"] - baseline["citation_valid"],
                                    "flags": ";".join(candidate["flags"])})
        comparisons.sort(key=lambda r: (r["case_id"], r["policy"]))
        result["comparisons"] = comparisons
        output.mkdir(parents=True, exist_ok=True)
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        atomic_json(output / "results.json", result)
        if stage == "development":
            decision = {"identity": identity, "policy": select_policy(summaries), "threshold": threshold,
                        "development_results_sha256": file_hash(output / "results.json")}
            if lock_path.exists() and json.loads(lock_path.read_text()) != decision:
                raise RunPaused("Development decision already frozen; do not retune it in place")
            atomic_json(lock_path, decision)
        return {"stage": stage, "summaries": summaries, "result_file": str(output / "results.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["retrieve", "freeze", "smoke", "development", "final"])
    parser.add_argument("--config", type=Path, default=HERE / "local_config.json")
    parser.add_argument("--execute", action="store_true", help="Explicitly run local retrieval/freezing or Gemini requests; default is read-only preflight")
    args = parser.parse_args()
    settings = load_settings(args.config)
    try:
        if not args.execute:
            result = preflight(settings, "retrieve" if args.stage == "retrieve" else args.stage)
        elif args.stage == "retrieve":
            result = collect_retrievals(settings)
        elif args.stage == "freeze":
            result = freeze(settings)
        else:
            result = run(settings, args.stage, os.environ.get("GEMINI_API_KEY", ""))
        print(json.dumps(result, indent=2))
        return 2 if result.get("status") == "blocked" else 0
    except (RunPaused, OSError, ValueError, KeyError) as exc:
        print(f"Phase3 paused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
