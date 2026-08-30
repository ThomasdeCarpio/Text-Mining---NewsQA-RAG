#!/usr/bin/env python3
"""Run the complete resumable Phase 1 tournament on one or two CUDA GPUs."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from newsqa_rag.evaluation.phase1 import load_comparison_rows, select_winner, write_rows_csv

def command(args: list[str | Path], *, device: int | None = None) -> None:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([str(value) for value in args], cwd=PROJECT_ROOT, env=env, check=True)


def build_one(chunks, base_manifest, root, kind, identifier, device=None):
    output = root / f"{kind}_{identifier.replace('/', '_')}"
    completed = output / "index_manifest.json"
    if completed.exists():
        return completed
    args = [sys.executable, "scripts/build_retrieval_models_index.py", "--chunks-path", chunks,
            "--base-variant-manifest", base_manifest, "--output-dir", output, "--overwrite"]
    if kind == "dense":
        args += ["--dense-models", identifier, "--skip-sparse", "--device", "cuda"]
    else:
        args += ["--sparse-ids", identifier, "--skip-dense"]
        if identifier == "bge_m3_sparse":
            args += ["--device", "cuda"]
    command(args, device=device)
    return completed


def index_build_schedule(fast=False, gpu_count=2):
    """Separate safe parallel jobs from high-host-memory model builds."""
    if gpu_count not in {1, 2}:
        raise ValueError("gpu_count must be 1 or 2")
    second_device = 1 if gpu_count > 1 else 0
    if fast:
        return {
            "light_dense": [("all-MiniLM-L6-v2", 0)],
            "cpu_sparse": ["bm25_okapi_simple"],
            "heavy": [],
        }
    return {
            "light_dense": [
            ("all-MiniLM-L6-v2", 0),
            ("BAAI/bge-small-en-v1.5", second_device),
        ],
        "cpu_sparse": [
            "bm25_okapi_simple",
            "bm25_plus_simple",
            "bm25_okapi_stemmed",
        ],
        # Run these in isolated subprocesses, one at a time. Concurrent model
        # materialization can exhaust Kaggle's host RAM before CUDA inference.
        "heavy": [
            ("dense", "intfloat/e5-base-v2", 0),
            ("dense", "BAAI/bge-large-en-v1.5", second_device),
            ("sparse", "bge_m3_sparse", 0),
        ],
    }


def parallel_build(chunks, base_manifest, root, fast=False, gpu_count=2):
    root.mkdir(parents=True, exist_ok=True)
    schedule = index_build_schedule(fast, gpu_count)
    manifests = []
    cpu_workers = min(2, max(1, (os.cpu_count() or 4) - 2))
    gpu_pools = [ThreadPoolExecutor(1) for _ in range(gpu_count)]
    with ThreadPoolExecutor(cpu_workers) as cpu:
        futures = []
        for model, device in schedule["light_dense"]:
            futures.append(gpu_pools[device].submit(
                build_one, chunks, base_manifest, root, "dense", model, device
            ))
        for sparse_id in schedule["cpu_sparse"]:
            futures.append(cpu.submit(build_one, chunks, base_manifest, root, "sparse", sparse_id, None))
        manifests.extend(future.result() for future in futures)
    for pool in gpu_pools:
        pool.shutdown()

    for kind, identifier, device in schedule["heavy"]:
        manifests.append(build_one(chunks, base_manifest, root, kind, identifier, device))

    merged = root / "index_manifest.json"
    command([sys.executable, "scripts/merge_phase1_index_manifests.py", *manifests, "--output", merged])
    return json.loads(merged.read_text(encoding="utf-8"))


def profiles_from_manifest(manifest):
    profiles = {}
    for _, item in manifest["dense_indexes"].items():
        profiles[f"dense_{item['slug']}"] = {**item, "retriever": "dense"}
    for name, item in manifest["sparse_indexes"].items():
        profiles[f"sparse_{name}"] = {**item, "retriever": "sparse"}
    return profiles


def phase1_experiment_id(stage: str, profile_name: str) -> str:
    """Return a schema-safe ID while retaining the profile label elsewhere."""

    raw = f"phase1-{stage}-{profile_name}".lower()
    return re.sub(r"[^a-z0-9_-]+", "_", raw).strip("_-")


def run_stage(
    stage, profiles, specs_root, experiments_root, cache,
    reranker_model=None, gpu_count=2, n_eval=None,
):
    specs_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for position, (name, profile) in enumerate(profiles.items()):
        spec = specs_root / f"{stage}_{name}.yaml"
        experiment_id = phase1_experiment_id(stage, name)
        args = [sys.executable, "scripts/create_phase1_experiment_spec.py", "--stage", stage,
                "--testset", profile["testset_original"], "--resolved-testset", profile["testset_resolved"],
                "--output", spec, "--experiment-id", experiment_id,
                "--runs-output-dir", experiments_root,
                "--shared-retrieval-cache", cache, "--profile",
                f"{name},{profile['retriever']},{profile['config_path']},{profile['variant_manifest']}"]
        if reranker_model:
            args += ["--reranker-model", reranker_model]
        if n_eval:
            args += ["--n-eval", str(n_eval)]
        command(args)
        jobs.append((spec, position % gpu_count))
    # XLM-R large has a high host-memory peak while loading. Running two copies
    # concurrently can cause Kaggle to kill both collectors and orphan parents.
    serial_large_reranker = stage == "round2" or str(reranker_model).startswith("BAAI/bge-reranker")
    workers = 1 if serial_large_reranker else min(gpu_count, len(jobs))
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(command, [sys.executable, "scripts/run_experiment.py", spec], device=device) for spec, device in jobs]
        for future in futures:
            future.result()
    comparisons = []
    for name in profiles:
        directory = experiments_root / phase1_experiment_id(stage, name)
        command([sys.executable, "scripts/summarize_experiments.py", directory])
        comparisons.append(directory / "comparison.json")
    return comparisons


def attach_testsets(profiles, original, resolved):
    return {name: {**value, "testset_original": str(original), "testset_resolved": str(resolved)} for name, value in profiles.items()}


def one_profile(manifest, kind, key):
    section = "dense_indexes" if kind == "dense" else "sparse_indexes"
    item = manifest[section][key]
    name = f"{kind}_{item.get('slug', key)}"
    return name, {**item, "retriever": kind}


def build_golden_chunk_profile(chunks, base_manifest, root, golden_method, dense_model, sparse_id, device):
    root.mkdir(parents=True, exist_ok=True)
    dense_value = sparse_value = None
    if golden_method in {"dense", "hybrid"}:
        dense_manifest_path = build_one(chunks, base_manifest, root, "dense", dense_model, device)
        dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
        _, dense_value = one_profile(dense_manifest, "dense", dense_model)
    if golden_method in {"sparse", "hybrid"}:
        sparse_manifest_path = build_one(chunks, base_manifest, root, "sparse", sparse_id,
                                         device if sparse_id == "bge_m3_sparse" else None)
        sparse_manifest = json.loads(sparse_manifest_path.read_text(encoding="utf-8"))
        _, sparse_value = one_profile(sparse_manifest, "sparse", sparse_id)
    if golden_method == "dense":
        return dense_value
    if golden_method == "sparse":
        return sparse_value
    command([sys.executable, "scripts/compose_phase1_retrieval_profile.py",
             "--dense-manifest", dense_value["variant_manifest"],
             "--sparse-manifest", sparse_value["variant_manifest"],
             "--output-dir", root, "--profile-id", "golden_hybrid"], device=device)
    return {"retriever": "hybrid", "config_path": str(root / "config_golden_hybrid.yaml"),
            "variant_manifest": str(root / "variant_golden_hybrid.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ThomasAnderson2009/newsqa-rag-evaluation")
    parser.add_argument("--revision", default="v1.0.0")
    parser.add_argument("--work-root", default="/kaggle/working/newsqa_phase1")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--smoke-questions",
        type=int,
        default=None,
        help="Evaluate only this many questions per method and variant",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of CUDA devices available to the runner",
    )
    parser.add_argument("--restore-checkpoint")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--stop-after", choices=["round1", "round2", "round3", "final"], default="final")
    args = parser.parse_args()
    if args.smoke_questions is not None and args.smoke_questions < 1:
        parser.error("--smoke-questions must be positive")
    work = Path(args.work_root).resolve()
    data = work / "data"
    indexes = work / "indexes"
    specs = work / "specs"
    cache = work / "retrieval_cache"
    results = work / "results"
    experiments = work / "experiments"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint_file = Path(args.checkpoint_path).resolve() if args.checkpoint_path else work.parent / "phase1_checkpoint.tar"
    checkpoint_base = checkpoint_file.with_suffix("")
    def save_checkpoint():
        if work.exists():
            shutil.make_archive(str(checkpoint_base), "tar", work, base_dir=".")
    atexit.register(save_checkpoint)
    if args.restore_checkpoint:
        shutil.unpack_archive(args.restore_checkpoint, work)
    if os.environ.get("HF_TOKEN") is None:
        raise SystemExit("HF_TOKEN must be configured through Kaggle Secrets")

    output_root = data / "newsqa_200_11064"
    command([sys.executable, "scripts/materialize_evaluation_dataset.py", "--repo-id", args.repo_id,
             "--revision", args.revision, "--output-root", output_root,
             "--db-path", data / "chroma_512_64", "--skip-vector-index"])
    final = output_root / "final_deduplicated"
    original, resolved = final / "testset_reviewed_original.jsonl", final / "testset_resolved.jsonl"
    base_manifest = output_root / "manifests/deduplicated.variant.json"

    round1_manifest_path = indexes / "round1/index_manifest.json"
    if round1_manifest_path.exists():
        round1_manifest = json.loads(round1_manifest_path.read_text(encoding="utf-8"))
    else:
        round1_manifest = parallel_build(
            final / "chunks.jsonl", base_manifest, indexes / "round1", args.fast, args.gpu_count
        )
    round1_profiles = attach_testsets(profiles_from_manifest(round1_manifest), original, resolved)
    round1_cmp = run_stage(
        "round1", round1_profiles, specs, experiments, cache,
        gpu_count=args.gpu_count, n_eval=args.smoke_questions,
    )
    round1_rows = load_comparison_rows(round1_cmp)
    write_rows_csv(round1_rows, results / "round1.csv")
    dense_winner = select_winner([row for row in round1_rows if str(row["index"]).startswith("dense_")])
    sparse_winner = select_winner([row for row in round1_rows if str(row["index"]).startswith("sparse_")])
    dense_profile = round1_profiles[dense_winner["index"]]
    sparse_profile = round1_profiles[sparse_winner["index"]]
    if args.stop_after == "round1":
        (results / "round1_winners.json").write_text(
            json.dumps({"dense": dense_winner, "sparse": sparse_winner}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "round1_complete", "results": str(results)}, indent=2))
        return

    hybrid_dir = indexes / "round2"
    hybrid_id = "hybrid_best"
    command([sys.executable, "scripts/compose_phase1_retrieval_profile.py",
             "--dense-manifest", dense_profile["variant_manifest"],
             "--sparse-manifest", sparse_profile["variant_manifest"],
             "--output-dir", hybrid_dir, "--profile-id", hybrid_id])
    round2_profiles = {
        "dense_best": dense_profile,
        "sparse_best": sparse_profile,
        "hybrid_best": {"retriever": "hybrid", "config_path": str(hybrid_dir / f"config_{hybrid_id}.yaml"),
                         "variant_manifest": str(hybrid_dir / f"variant_{hybrid_id}.json"),
                         "testset_original": str(original), "testset_resolved": str(resolved)},
    }
    round2_cmp = run_stage(
        "round2", round2_profiles, specs, experiments, cache,
        gpu_count=args.gpu_count, n_eval=args.smoke_questions,
    )
    round2_rows = load_comparison_rows(round2_cmp)
    write_rows_csv(round2_rows, results / "round2.csv")
    golden = select_winner(round2_rows)
    cross_rows = [row for row in round2_rows if row.get("reranker") == "cross-encoder"]
    best_cross = select_winner(cross_rows)

    winner_lock = {"round1_dense": dense_winner, "round1_sparse": sparse_winner,
                   "round2_golden": golden, "cross_encoder": best_cross}
    results.mkdir(parents=True, exist_ok=True)
    (results / "winner_lock.json").write_text(json.dumps(winner_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.stop_after == "round2":
        print(json.dumps({"status": "round2_complete", "results": str(results)}, indent=2))
        return
    ablation_command = [sys.executable, "scripts/build_ablation_datasets.py", "--repo-id", args.repo_id,
                        "--revision", args.revision, "--output-base", data / "ablation",
                        "--db-path-base", data / "ablation_chroma", "--skip-vector-index"]
    if args.fast:
        ablation_command += ["--chunk-sizes", "512"]
    command(ablation_command)
    golden_method = round2_profiles[golden["index"]]["retriever"]
    dense_model = dense_profile["model_name"]
    sparse_id = sparse_profile["sparse_id"]
    variants = [(256, 32), (512, 64), (1024, 128)] if not args.fast else [(512, 64)]
    def build_round3_variant(position_variant):
        position, (size, overlap) = position_variant
        variant_root = data / "ablation" / f"chunk_{size}_{overlap}_recursive"
        profile = build_golden_chunk_profile(
            variant_root / "final_deduplicated/chunks.jsonl",
            variant_root / "manifests/deduplicated.variant.json",
            indexes / f"round3_{size}_{overlap}", golden_method, dense_model, sparse_id,
            position % args.gpu_count,
        )
        profile.update({
            "testset_original": str(variant_root / "final_deduplicated/testset_reviewed_original.jsonl"),
            "testset_resolved": str(variant_root / "final_deduplicated/testset_resolved.jsonl"),
        })
        return f"chunk_{size}_{overlap}", profile

    large_chunk_profile = (
        dense_model in {"intfloat/e5-base-v2", "BAAI/bge-large-en-v1.5"}
        or sparse_id == "bge_m3_sparse"
    )
    with ThreadPoolExecutor(
        1 if large_chunk_profile else min(args.gpu_count, len(variants))
    ) as pool:
        built_profiles = list(pool.map(build_round3_variant, enumerate(variants)))
    round3_profiles = dict(built_profiles)
    round3_cmp = run_stage(
        "round3", round3_profiles, specs, experiments, cache,
        best_cross["reranker_model"], args.gpu_count, args.smoke_questions,
    )
    round3_rows = load_comparison_rows(round3_cmp)
    write_rows_csv(round3_rows, results / "round3.csv")
    final_winner = select_winner(round3_rows)
    final_profile = round3_profiles[final_winner["index"]]
    winner_lock["round3_final"] = final_winner
    (results / "winner_lock.json").write_text(json.dumps(winner_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.stop_after == "round3":
        print(json.dumps({"status": "round3_complete", "results": str(results)}, indent=2))
        return

    final_spec = specs / "final.yaml"
    final_args = [sys.executable, "scripts/create_phase1_experiment_spec.py", "--stage", "final",
                  "--testset", final_profile["testset_original"], "--resolved-testset", final_profile["testset_resolved"],
                  "--output", final_spec, "--experiment-id", "phase1-final-locked",
                  "--runs-output-dir", experiments,
                  "--shared-retrieval-cache", cache, "--profile",
                  f"locked,{final_profile['retriever']},{final_profile['config_path']},{final_profile['variant_manifest']}",
                  "--final-reranker", final_winner["reranker"]]
    if final_winner.get("reranker_model"):
        final_args += ["--reranker-model", final_winner["reranker_model"]]
    if args.smoke_questions:
        final_args += ["--n-eval", str(args.smoke_questions)]
    command(final_args)
    command([sys.executable, "scripts/run_experiment.py", final_spec], device=0)
    final_dir = experiments / "phase1-final-locked"
    command([sys.executable, "scripts/summarize_experiments.py", final_dir])
    final_rows = load_comparison_rows([final_dir / "comparison.json"])
    write_rows_csv(final_rows, results / "final_test.csv")
    command([sys.executable, "scripts/export_phase1_results.py", "--experiments-root", experiments,
             "--output-dir", results])
    print(json.dumps({"status": "complete", "results": str(results),
                      "checkpoint": str(checkpoint_file)}, indent=2))


if __name__ == "__main__":
    main()
