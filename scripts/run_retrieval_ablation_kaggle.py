#!/usr/bin/env python3
"""Ablate what the retrievers are actually given, to test whether the sparse lock is fair.

Phase 1 compared retrievers, not *representations*. Every index in that
tournament was built over `chunk["text"]` alone and nothing else, so the
verdict "sparse beats dense here" is really "sparse beats dense **on this
representation**". This script varies the representation and re-measures.

Two axes, chosen because they are the two the tournament never touched:

  DOCUMENT SIDE   contextual chunking. Note first what is NOT on offer: this
                  dataset has no usable metadata to embed. `metadata.title` is
                  not a headline, it is the first 160 characters of the article
                  body (verified: a prefix of the text in 200/200 evaluation
                  articles), and url, author and publish_date are empty strings
                  the chunker fills in by default. Publisher is "CNN" for all
                  11,064. So "index the metadata" has nothing to index.

                  What the title field accidentally provides is an article-level
                  context blurb, and that IS worth testing: chunk 0 opens with
                  the article, but chunks 1..n arrive with no indication of what
                  story they belong to. 51.4% of chunks (11,702 of 22,766) are
                  such continuation chunks. Arm: prepend the article's opening
                  to every continuation chunk, so each one carries its own
                  context - a crude form of contextual retrieval.

  QUERY SIDE      each model already receives its own prescribed prefix -
                  "query: " for e5, "Represent this sentence for searching
                  relevant passages: " for bge-* (embeddings.py:131-144). So the
                  canonical instruction is NOT missing. What is untested is
                  whether extra framing on top of it helps. Arm: wrap the
                  question as a description of the passage being sought.

Both arms are pure data transforms feeding the existing scripts unchanged - no
retriever code is modified, so nothing here can flatter one side by accident.
Chunk IDs derive from article id and position, not text, so the gold labels stay
valid across arms and every comparison is paired on the same 281 questions.

    python scripts/run_retrieval_ablation_kaggle.py --work-root /kaggle/working/ablation

This does NOT re-open the Phase 1 lock. It is a fairness check whose result is
reported either way: if the gap survives, the lock is evidence-backed; if it
closes, that is a limitation the report must state.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.experiments import paired_comparison  # noqa: E402

DENSE_MODEL = "intfloat/e5-base-v2"
SPARSE_ID = "bge_m3_sparse"
QUERY_PROMPT = "A passage that answers the question: {question}"


def command(args: list[str | Path], *, device: int | None = None) -> None:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([str(v) for v in args], cwd=PROJECT_ROOT, env=env, check=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def contextual_chunks(source: Path, target: Path) -> Path:
    """Give every continuation chunk the article's opening as context.

    metadata.title holds the article's first 160 characters, not a headline, so
    this prepends an article-level context blurb rather than a title. Chunk 0
    already begins with that text and is skipped - duplicating it would distort
    term frequencies, and chunk 0 needs no context because it *is* the context.

    Chunk IDs are never touched: the gold labels reference them, and every
    comparison in this script is paired on them.
    """
    if target.exists():
        return target
    rows, changed = read_jsonl(source), 0
    for row in rows:
        context = str((row.get("metadata") or {}).get("title", "")).strip()
        if context and not row["text"].lstrip().startswith(context):
            row["text"] = f"{context}\n\n{row['text']}"
            changed += 1
    print(f"  context prepended to {changed:,} of {len(rows):,} chunks "
          f"({changed / max(1, len(rows)):.1%} - the continuation chunks)", flush=True)
    return write_jsonl(target, rows)


def prompted_testset(source: Path, target: Path) -> Path:
    """Wrap each question in passage-seeking framing, on top of the model prefix."""
    if target.exists():
        return target
    rows = read_jsonl(source)
    for row in rows:
        row["question"] = QUERY_PROMPT.format(question=row["question"])
    return write_jsonl(target, rows)


def build_index(chunks: Path, base_manifest: Path, root: Path, kind: str, identifier: str) -> Path:
    manifest = root / "index_manifest.json"
    if manifest.exists():
        print(f"  reuse {manifest}", flush=True)
        return manifest
    args = [sys.executable, "scripts/build_retrieval_models_index.py",
            "--chunks-path", chunks, "--base-variant-manifest", base_manifest,
            "--output-dir", root, "--overwrite"]
    if kind == "dense":
        args += ["--dense-models", identifier, "--skip-sparse", "--device", "cuda"]
    else:
        args += ["--sparse-ids", identifier, "--skip-dense", "--device", "cuda"]
    command(args)
    return manifest


def run_arm(arm: str, profile: dict, original: Path, resolved: Path,
            specs: Path, experiments: Path, cache: Path, n_eval: int | None) -> Path:
    experiment_id = re.sub(r"[^a-z0-9_-]+", "_", f"ablation-{arm}".lower()).strip("_-")
    spec = specs / f"{arm}.yaml"
    args = [sys.executable, "scripts/create_phase1_experiment_spec.py", "--stage", "round1",
            "--testset", original, "--resolved-testset", resolved, "--output", spec,
            "--experiment-id", experiment_id, "--runs-output-dir", experiments,
            "--shared-retrieval-cache", cache,
            "--profile", f"{arm},{profile['retriever']},{profile['config_path']},{profile['variant_manifest']}"]
    if n_eval:
        args += ["--n-eval", str(n_eval)]
    command(args)
    command([sys.executable, "scripts/run_experiment.py", spec])
    directory = experiments / experiment_id
    command([sys.executable, "scripts/summarize_experiments.py", directory])
    return directory


def scores_by_variant(directory: Path) -> dict[str, list[dict]]:
    out = {}
    for meta in sorted(directory.glob("*/experiment_run.json")):
        scores = meta.parent / "deterministic_scores.jsonl"
        if not scores.exists():
            continue
        blob = json.loads(meta.read_text(encoding="utf-8"))
        out[blob["parameters"]["variant"]] = read_jsonl(scores)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default="/kaggle/working/newsqa_ablation")
    parser.add_argument("--repo-id", default="MatchaMacchiato/newsqa_200_11064_v2.0.0")
    parser.add_argument("--revision", default="b81c8db6847a23272665946c0c43c72e9a212fd9")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--restore-checkpoint", default=None)
    parser.add_argument("--n-eval", type=int, default=None, help="Limit questions for a smoke run")
    parser.add_argument("--skip-query-axis", action="store_true")
    args = parser.parse_args()

    work = Path(args.work_root).resolve()
    data, indexes, specs = work / "data", work / "indexes", work / "specs"
    cache, experiments, results = work / "retrieval_cache", work / "experiments", work / "results"
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint_path).resolve() if args.checkpoint_path else work.parent / "ablation_checkpoint.tar"
    atexit.register(lambda: work.exists() and shutil.make_archive(str(checkpoint.with_suffix("")), "tar", work, base_dir="."))
    if args.restore_checkpoint:
        shutil.unpack_archive(args.restore_checkpoint, work)

    output_root = data / "newsqa_200_11064"
    command([sys.executable, "scripts/materialize_evaluation_dataset.py", "--repo-id", args.repo_id,
             "--revision", args.revision, "--output-root", output_root,
             "--db-path", data / "chroma", "--skip-vector-index"])
    final = output_root / "final_deduplicated"
    original, resolved = final / "testset_reviewed_original.jsonl", final / "testset_resolved.jsonl"
    base_manifest = output_root / "manifests/deduplicated.variant.json"
    plain_chunks = final / "chunks.jsonl"

    print("\n=== document axis: build the title-augmented corpus ===", flush=True)
    context_chunks = contextual_chunks(plain_chunks, data / "chunks_contextual.jsonl")

    arms: dict[str, dict] = {}
    for corpus_name, chunks in (("plain", plain_chunks), ("contextual", context_chunks)):
        for kind, identifier, key in (("dense", DENSE_MODEL, f"dense_{DENSE_MODEL.replace('/', '_')}"),
                                      ("sparse", SPARSE_ID, f"sparse_{SPARSE_ID}")):
            print(f"\n=== index {kind} / {corpus_name} ===", flush=True)
            root = indexes / f"{corpus_name}_{kind}"
            manifest = build_index(chunks, base_manifest, root, kind, identifier)
            blob = json.loads(manifest.read_text(encoding="utf-8"))
            section = "dense_indexes" if kind == "dense" else "sparse_indexes"
            index_key = next(iter(blob[section]))
            profile = {**blob[section][index_key], "retriever": kind}
            arms[f"{corpus_name}_{kind}"] = {"profile": profile, "original": original, "resolved": resolved}

    if not args.skip_query_axis:
        print("\n=== query axis: reuse the plain indexes, reword the questions ===", flush=True)
        p_original = prompted_testset(original, data / "testset_original_prompted.jsonl")
        p_resolved = prompted_testset(resolved, data / "testset_resolved_prompted.jsonl")
        for kind in ("dense", "sparse"):
            arms[f"prompted_{kind}"] = {"profile": arms[f"plain_{kind}"]["profile"],
                                        "original": p_original, "resolved": p_resolved}

    directories = {}
    for arm, spec in arms.items():
        print(f"\n=== evaluate {arm} ===", flush=True)
        directories[arm] = run_arm(arm, spec["profile"], spec["original"], spec["resolved"],
                                   specs, experiments, cache, args.n_eval)

    scores = {arm: scores_by_variant(d) for arm, d in directories.items()}
    comparisons = []
    for kind in ("dense", "sparse"):
        for treatment, label in (("contextual", "article opening prepended to continuation chunks"),
                                 ("prompted", "question wrapped in passage-seeking prompt")):
            baseline_arm, treated_arm = f"plain_{kind}", f"{treatment}_{kind}"
            if treated_arm not in scores:
                continue
            for variant in ("resolved", "original"):
                left = scores[baseline_arm].get(variant)
                right = scores[treated_arm].get(variant)
                if not left or not right:
                    continue
                record = {"retriever": kind, "treatment": treatment, "description": label,
                          "variant": variant, "metrics": {}}
                for metric in ("retrieval.ndcg@5", "retrieval.mrr@5",
                               "retrieval.hit_rate@1", "retrieval.hit_rate@5"):
                    out = paired_comparison(left, right, metric, 1000, 42)
                    if out:
                        record["metrics"][metric] = {
                            "delta_treated_minus_baseline": out["mean_delta_right_minus_left"],
                            "ci95_low": out["ci95_low"], "ci95_high": out["ci95_high"],
                            "n_pairs": out["n_pairs"],
                            "significant": out["ci95_low"] > 0 or out["ci95_high"] < 0,
                        }
                comparisons.append(record)

    summary = {"schema_version": 1, "method": "paired percentile bootstrap, seed 42, 1000 samples",
               "query_prompt_template": QUERY_PROMPT,
               "note": "positive delta means the treatment helped that retriever",
               "comparisons": comparisons}
    (results / "ablation_paired.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("REPRESENTATION ABLATION - does the sparse lock survive a fairer dense?")
    print("=" * 78)
    for record in comparisons:
        entry = record["metrics"].get("retrieval.ndcg@5")
        if not entry:
            continue
        mark = "SIG " if entry["significant"] else "    "
        print(f"{mark}{record['retriever']:7s} {record['treatment']:9s} {record['variant']:9s} "
              f"nDCG@5 {entry['delta_treated_minus_baseline']:+.4f} "
              f"[{entry['ci95_low']:+.4f}, {entry['ci95_high']:+.4f}]")
    print(f"\nWrote {results / 'ablation_paired.json'}")


if __name__ == "__main__":
    main()
