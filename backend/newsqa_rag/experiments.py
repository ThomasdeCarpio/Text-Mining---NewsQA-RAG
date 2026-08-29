"""Small experiment orchestrator built on the existing benchmark CLIs."""

from __future__ import annotations

import csv
import importlib.metadata
import itertools
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from newsqa_rag.evaluation.benchmark_io import atomic_write_json, stable_hash, utc_now
from newsqa_rag.evaluation.testset import load_testset, sha256_file
from newsqa_rag.model_gateway import PROJECT_ROOT


class ExperimentSpecError(ValueError):
    """Raised when an experiment specification is incomplete or inconsistent."""


_MATRIX_KEYS = {
    "generator_model",
    "index",
    "partition",
    "rerank_top_n",
    "reranker",
    "reranker_model",
    "retrieval_only",
    "retriever",
    "top_k",
    "variant",
}
_DEFAULTS = {
    "partition": "development",
    "retrieval_only": True,
    "retriever": "hybrid",
    "reranker": "noop",
    "top_k": 10,
    "rerank_top_n": 5,
}
_DEFAULT_METRICS = (
    "retrieval.mrr@5",
    "retrieval.ndcg@5",
    "retrieval.recall@5",
    "qa.f1",
    "citations.citation_f1",
)
_PACKAGE_NAMES = (
    "newsqa-rag",
    "chromadb",
    "fastapi",
    "numpy",
    "openai",
    "ragas",
    "sentence-transformers",
    "torch",
)


def load_experiment_spec(path: str | Path) -> dict:
    """Load and validate one YAML experiment specification."""

    with Path(path).open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    validate_experiment_spec(spec)
    return spec


def validate_experiment_spec(spec: dict) -> None:
    """Validate the small schema consumed by :func:`expand_run_matrix`."""

    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ExperimentSpecError("schema_version must be 1")
    experiment = spec.get("experiment")
    if not isinstance(experiment, dict):
        raise ExperimentSpecError("experiment must be a mapping")
    experiment_id = str(experiment.get("id", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment_id):
        raise ExperimentSpecError("experiment.id must contain lowercase letters, digits, _ or -")

    dataset = spec.get("dataset")
    indexes = dataset.get("indexes") if isinstance(dataset, dict) else None
    if not isinstance(indexes, dict) or not indexes:
        raise ExperimentSpecError("dataset.indexes must contain at least one index")
    for name, index in indexes.items():
        if not isinstance(index, dict):
            raise ExperimentSpecError(f"dataset.indexes.{name} must be a mapping")
        missing = {"config", "variant_manifest", "testsets"} - set(index)
        if missing:
            raise ExperimentSpecError(f"dataset.indexes.{name} is missing {sorted(missing)}")
        if not isinstance(index["testsets"], dict) or not index["testsets"]:
            raise ExperimentSpecError(f"dataset.indexes.{name}.testsets must be a mapping")
        paths = [index["config"], index["variant_manifest"], *index["testsets"].values()]
        if not all(isinstance(path, str) and path for path in paths):
            raise ExperimentSpecError(f"dataset.indexes.{name} paths must be strings")

    for section in ("runtime", "judge", "summary", "pricing"):
        if section in spec and not isinstance(spec[section], dict):
            raise ExperimentSpecError(f"{section} must be a mapping")

    fixed = spec.get("fixed", {})
    matrix = spec.get("matrix", {})
    explicit_runs = spec.get("runs")
    if not isinstance(fixed, dict) or not isinstance(matrix, dict):
        raise ExperimentSpecError("fixed and matrix must be mappings")
    unknown = (set(fixed) | set(matrix)) - _MATRIX_KEYS
    if unknown:
        raise ExperimentSpecError(f"unsupported experiment parameters: {sorted(unknown)}")
    overlap = set(fixed) & set(matrix)
    if overlap:
        raise ExperimentSpecError(f"parameters cannot appear in fixed and matrix: {sorted(overlap)}")
    for key, values in matrix.items():
        if not isinstance(values, list) or not values:
            raise ExperimentSpecError(f"matrix.{key} must be a non-empty list")
    if explicit_runs is not None:
        if matrix:
            raise ExperimentSpecError("runs and matrix are mutually exclusive")
        if not isinstance(explicit_runs, list) or not explicit_runs:
            raise ExperimentSpecError("runs must be a non-empty list")
        for position, run in enumerate(explicit_runs):
            if not isinstance(run, dict):
                raise ExperimentSpecError(f"runs[{position}] must be a mapping")
            unknown_run = set(run) - _MATRIX_KEYS
            if unknown_run:
                raise ExperimentSpecError(f"runs[{position}] has unsupported parameters: {sorted(unknown_run)}")

    development_articles = int(dataset.get("development_articles", 50))
    if development_articles < 1:
        raise ExperimentSpecError("dataset.development_articles must be positive")


def expand_run_matrix(spec: dict) -> list[dict]:
    """Expand the declared Cartesian matrix into stable, validated run records."""

    validate_experiment_spec(spec)
    fixed = {**_DEFAULTS, **spec.get("fixed", {})}
    matrix = spec.get("matrix", {})
    keys = sorted(matrix)
    explicit_runs = spec.get("runs")
    combinations: Iterable[dict[str, Any]]
    if explicit_runs is not None:
        combinations = explicit_runs
    else:
        combinations = [dict(zip(keys, values)) for values in (
            itertools.product(*(matrix[key] for key in keys)) if keys else [()]
        )]
    indexes = spec["dataset"]["indexes"]
    runs = []
    for values in combinations:
        parameters = {**fixed, **values}
        index_name = parameters.get("index")
        if not index_name:
            if len(indexes) != 1:
                raise ExperimentSpecError("index is required when more than one index is registered")
            index_name = next(iter(indexes))
            parameters["index"] = index_name
        if index_name not in indexes:
            raise ExperimentSpecError(f"unknown index: {index_name}")
        variant = parameters.get("variant")
        if not variant:
            variants = indexes[index_name]["testsets"]
            if len(variants) != 1:
                raise ExperimentSpecError("variant is required when an index has multiple testsets")
            variant = next(iter(variants))
            parameters["variant"] = variant
        if variant not in indexes[index_name]["testsets"]:
            raise ExperimentSpecError(f"index {index_name!r} has no {variant!r} testset")
        _validate_run(parameters, spec.get("judge", {}))
        identity = {"spec": spec, "parameters": parameters}
        readable = "-".join(
            str(parameters[key])
            for key in ("variant", "partition", "retriever", "reranker")
        )
        run_id = f"{_slug(readable)}-{stable_hash(identity)[:10]}"
        runs.append({"run_id": run_id, "parameters": parameters})
    return runs


def _validate_run(parameters: dict, judge: dict) -> None:
    if parameters["partition"] not in {"development", "final_test", "all"}:
        raise ExperimentSpecError("partition must be development, final_test or all")
    if parameters["retriever"] not in {"dense", "bm25", "sparse", "hybrid"}:
        raise ExperimentSpecError(f"unsupported retriever: {parameters['retriever']}")
    if parameters["reranker"] not in {"noop", "cross-encoder"}:
        raise ExperimentSpecError(f"unsupported reranker: {parameters['reranker']}")
    if int(parameters["top_k"]) < 1 or int(parameters["rerank_top_n"]) < 1:
        raise ExperimentSpecError("top_k and rerank_top_n must be positive")
    if int(parameters["rerank_top_n"]) > int(parameters["top_k"]):
        raise ExperimentSpecError("rerank_top_n cannot exceed top_k")
    if parameters.get("retrieval_only") and parameters.get("generator_model"):
        raise ExperimentSpecError("generator_model cannot be set for a retrieval-only run")
    if judge.get("enabled") and not parameters.get("retrieval_only"):
        judge_model = judge.get("model")
        if not judge_model:
            raise ExperimentSpecError("judge.model is required when judging is enabled")
        if judge_model == parameters.get("generator_model") and not judge.get("allow_same_model"):
            raise ExperimentSpecError("judge and generator models must differ")


def build_article_partitions(
    testsets: dict[str, str | Path],
    development_articles: int,
    seed: int = 42,
    article_field: str = "article_key",
) -> dict:
    """Split paired variants by article and return locked question IDs."""

    records_by_variant = {
        variant: load_testset(path)
        for variant, path in testsets.items()
    }
    article_sets = {
        variant: {str(row.get(article_field) or "") for row in rows}
        for variant, rows in records_by_variant.items()
    }
    if any("" in articles for articles in article_sets.values()):
        raise ExperimentSpecError(f"every test row must contain {article_field}")
    first_articles = next(iter(article_sets.values()))
    if any(articles != first_articles for articles in article_sets.values()):
        raise ExperimentSpecError("question variants must contain the same article set")
    if development_articles >= len(first_articles):
        raise ExperimentSpecError("development_articles must be smaller than the article count")

    article_ids = sorted(first_articles)
    random.Random(seed).shuffle(article_ids)
    development = sorted(article_ids[:development_articles])
    final_test = sorted(article_ids[development_articles:])
    partitions = {}
    for partition, selected in (
        ("development", development),
        ("final_test", final_test),
        ("all", sorted(article_ids)),
    ):
        selected_set = set(selected)
        questions = {
            variant: [
                row["question_id"]
                for row in rows
                if str(row.get(article_field) or "") in selected_set
                and row.get("relevant_chunk_ids")
            ]
            for variant, rows in records_by_variant.items()
        }
        partitions[partition] = {"article_ids": selected, "question_ids": questions}

    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "seed": seed,
        "article_field": article_field,
        "development_articles": development_articles,
        "total_articles": len(article_ids),
        "sources": {
            variant: {"path": str(Path(path)), "sha256": sha256_file(path)}
            for variant, path in testsets.items()
        },
        "partitions": partitions,
    }


def environment_record() -> dict:
    """Capture cheap, stable runtime provenance without importing heavy ML packages."""

    packages = {}
    for name in _PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = _run_output(["git", "rev-parse", "HEAD"])
    dirty = _run_output(["git", "status", "--porcelain"])
    return {
        "captured_at": utc_now(),
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "threads": {
            name: os.getenv(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")
        },
        "gpu": {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "peak_memory_mb": None,
        },
    }


def run_experiment(spec_path: str | Path, *, dry_run: bool = False) -> Path:
    """Execute or resume every run by invoking the existing benchmark scripts."""

    spec_path = Path(spec_path).resolve()
    spec = load_experiment_spec(spec_path)
    runs = expand_run_matrix(spec)
    seed = int(spec.get("seed", 42))
    experiment_id = spec["experiment"]["id"]
    output_root = _project_path(spec.get("output_dir", "outputs/experiments"))
    experiment_root = output_root / experiment_id
    partition_payloads = {
        name: build_article_partitions(
            {variant: _project_path(path) for variant, path in index["testsets"].items()},
            int(spec["dataset"].get("development_articles", 50)),
            seed,
            str(spec["dataset"].get("article_field", "article_key")),
        )
        for name, index in spec["dataset"]["indexes"].items()
    }

    if dry_run:
        for run in runs:
            command = _collect_command(spec, run, experiment_root)
            print(_format_command(command))
        return experiment_root

    experiment_root.mkdir(parents=True, exist_ok=True)
    _copy_locked_spec(spec_path, experiment_root / "experiment_spec.yaml")
    question_files = _write_partition_artifacts(experiment_root, partition_payloads)
    registry_path = experiment_root / "registry.json"
    registry = _load_registry(registry_path, experiment_id, runs)

    for run in runs:
        run_id = run["run_id"]
        run_dir = experiment_root / run_id
        entry = next(item for item in registry["runs"] if item["run_id"] == run_id)
        if is_run_complete(run_dir, spec.get("judge", {}), run["parameters"]):
            entry.update({"status": "complete", "updated_at": utc_now()})
            atomic_write_json(registry_path, registry)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        _copy_locked_spec(spec_path, run_dir / "experiment_spec.yaml")
        atomic_write_json(run_dir / "environment.json", environment_record())
        atomic_write_json(
            run_dir / "experiment_run.json",
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "parameters": run["parameters"],
                "pricing": spec.get("pricing", {}),
            },
        )
        entry.update({"status": "running", "started_at": utc_now()})
        atomic_write_json(registry_path, registry)
        started = time.perf_counter()
        try:
            _execute(
                _collect_command(spec, run, experiment_root, question_files),
                dry_run=False,
            )
            _annotate_run_manifest(run_dir, experiment_id, run, spec.get("pricing", {}))
            _execute([sys.executable, "scripts/score_benchmark_predictions.py", "--run-dir", str(run_dir)])
            judge = spec.get("judge", {})
            if judge.get("enabled") and not run["parameters"].get("retrieval_only"):
                _execute(_judge_command(judge, run_dir, seed))
                _execute([sys.executable, "scripts/score_benchmark_predictions.py", "--run-dir", str(run_dir)])
        except (Exception, KeyboardInterrupt) as exc:
            entry.update(
                {
                    "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "updated_at": utc_now(),
                    "wall_time_seconds": round(time.perf_counter() - started, 3),
                    "error": exc.__class__.__name__,
                }
            )
            atomic_write_json(registry_path, registry)
            raise
        entry.update(
            {
                "status": "complete",
                "updated_at": utc_now(),
                "wall_time_seconds": round(time.perf_counter() - started, 3),
            }
        )
        atomic_write_json(registry_path, registry)
    return experiment_root


def is_run_complete(run_dir: str | Path, judge: dict, parameters: dict) -> bool:
    """Return whether a run already has every requested cached artifact."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    report_path = run_dir / "report.json"
    if not manifest_path.exists() or not report_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return False
    if judge.get("enabled") and not parameters.get("retrieval_only"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return (run_dir / "judge_results.jsonl").exists() and "ragas" in report
    return True


def summarize_experiment(
    experiment_dir: str | Path,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Build comparison artifacts, confidence intervals and paired deltas."""

    experiment_dir = Path(experiment_dir)
    spec = load_experiment_spec(experiment_dir / "experiment_spec.yaml")
    metric_names = tuple(spec.get("summary", {}).get("metrics", _DEFAULT_METRICS))
    rows = []
    per_run_scores: dict[str, list[dict]] = {}
    for metadata_path in sorted(experiment_dir.glob("*/experiment_run.json")):
        run_dir = metadata_path.parent
        report_path = run_dir / "report.json"
        scores_path = run_dir / "deterministic_scores.jsonl"
        if not report_path.exists() or not scores_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scores = _load_jsonl(scores_path)
        run_id = metadata["run_id"]
        per_run_scores[run_id] = scores
        row = {"run_id": run_id, **metadata["parameters"]}
        row.update(_report_columns(report))
        for metric in metric_names:
            values = [float(value) for score in scores if (value := _nested(score, metric)) is not None]
            article_values = _article_macro_values(scores, metric)
            if values:
                low, high = bootstrap_ci(values, bootstrap_samples, seed)
                row[f"{metric}.mean"] = round(statistics.fmean(values), 6)
                row[f"{metric}.ci95_low"] = low
                row[f"{metric}.ci95_high"] = high
            if article_values:
                row[f"{metric}.article_macro"] = round(statistics.fmean(article_values), 6)
        row["estimated_generation_cost_usd"] = _estimated_cost(report, metadata.get("pricing", {}))
        rows.append(row)

    paired_metric = str(spec.get("summary", {}).get("paired_metric", "retrieval.mrr@5"))
    paired = []
    for left, right in itertools.combinations(rows, 2):
        comparison = paired_comparison(
            per_run_scores[left["run_id"]],
            per_run_scores[right["run_id"]],
            paired_metric,
            bootstrap_samples,
            seed,
        )
        if comparison:
            paired.append({"left": left["run_id"], "right": right["run_id"], **comparison})

    quality_metric = str(spec.get("summary", {}).get("quality_metric", "retrieval.mrr@5.mean"))
    latency_metric = str(spec.get("summary", {}).get("latency_metric", "latency.total.p95_ms"))
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "experiment_id": spec["experiment"]["id"],
        "runs": rows,
        "paired_metric": paired_metric,
        "paired_comparisons": paired,
        "pareto_run_ids": pareto_frontier(rows, quality_metric, latency_metric),
    }
    atomic_write_json(experiment_dir / "comparison.json", summary)
    _write_csv(experiment_dir / "comparison.csv", rows)
    return summary


def bootstrap_ci(values: list[float], samples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap confidence interval."""

    if not values:
        raise ValueError("values cannot be empty")
    if samples < 1:
        raise ValueError("samples must be positive")
    if len(values) == 1:
        value = round(values[0], 6)
        return value, value
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return round(_percentile(means, 0.025), 6), round(_percentile(means, 0.975), 6)


def paired_comparison(
    left: list[dict],
    right: list[dict],
    metric: str,
    samples: int = 1000,
    seed: int = 42,
) -> dict | None:
    """Compare two runs on shared source-question IDs using paired bootstrap."""

    def keyed(rows: list[dict]) -> dict[str, float]:
        return {
            str(row.get("source_question_id") or row.get("question_id")): float(value)
            for row in rows
            if (value := _nested(row, metric)) is not None
        }

    left_values, right_values = keyed(left), keyed(right)
    shared = sorted(set(left_values) & set(right_values))
    if not shared:
        return None
    deltas = [right_values[key] - left_values[key] for key in shared]
    low, high = bootstrap_ci(deltas, samples, seed)
    return {
        "n_pairs": len(deltas),
        "mean_delta_right_minus_left": round(statistics.fmean(deltas), 6),
        "ci95_low": low,
        "ci95_high": high,
    }


def pareto_frontier(rows: list[dict], quality_metric: str, latency_metric: str) -> list[str]:
    """Return runs not dominated on higher quality and lower latency."""

    candidates = [
        row for row in rows
        if isinstance(row.get(quality_metric), (int, float))
        and isinstance(row.get(latency_metric), (int, float))
    ]
    return [
        row["run_id"]
        for row in candidates
        if not any(
            other[quality_metric] >= row[quality_metric]
            and other[latency_metric] <= row[latency_metric]
            and (
                other[quality_metric] > row[quality_metric]
                or other[latency_metric] < row[latency_metric]
            )
            for other in candidates
        )
    ]


def _collect_command(
    spec: dict,
    run: dict,
    experiment_root: Path,
    question_files: dict | None = None,
) -> list[str]:
    parameters = run["parameters"]
    index = spec["dataset"]["indexes"][parameters["index"]]
    run_dir = experiment_root / run["run_id"]
    key = (parameters["index"], parameters["partition"], parameters["variant"])
    question_path = (
        question_files[key]
        if question_files
        else experiment_root / "partitions" / parameters["index"] / f"{parameters['partition']}_{parameters['variant']}.json"
    )
    command = [
        sys.executable,
        "scripts/collect_benchmark_predictions.py",
        "--retriever", str(parameters["retriever"]),
        "--reranker", str(parameters["reranker"]),
        "--testset", str(_project_path(index["testsets"][parameters["variant"]])),
        "--variant-manifest", str(_project_path(index["variant_manifest"])),
        "--config", str(index["config"]),
        "--run-dir", str(run_dir),
        "--question-ids-file", str(question_path),
        "--top-k", str(parameters["top_k"]),
        "--rerank-top-n", str(parameters["rerank_top_n"]),
        "--seed", str(spec.get("seed", 42)),
        "--max-attempts", str(spec.get("runtime", {}).get("max_attempts", 3)),
    ]
    if parameters.get("retrieval_only"):
        command.append("--retrieval-only")
    if parameters.get("generator_model"):
        command.extend(["--generator-model", str(parameters["generator_model"])])
    if parameters.get("reranker_model"):
        command.extend(["--reranker-model", str(parameters["reranker_model"])])
    runtime = spec.get("runtime", {})
    if runtime.get("n_eval"):
        command.extend(["--n-eval", str(runtime["n_eval"])])
    if runtime.get("retry_failed"):
        command.append("--retry-failed")
    if runtime.get("progress", True):
        command.append("--progress")
    return command


def _judge_command(judge: dict, run_dir: Path, seed: int) -> list[str]:
    command = [
        sys.executable,
        "scripts/judge_benchmark_predictions.py",
        "--run-dir", str(run_dir),
        "--judge-provider", str(judge.get("provider", "openai")),
        "--judge-model", str(judge["model"]),
        "--batch-size", str(judge.get("batch_size", 10)),
        "--max-workers", str(judge.get("max_workers", 4)),
        "--seed", str(seed),
        "--max-attempts", str(judge.get("max_attempts", 3)),
    ]
    if judge.get("pilot_questions"):
        command.extend(["--n-eval", str(judge["pilot_questions"])])
    if judge.get("allow_same_model"):
        command.append("--allow-same-judge")
    if judge.get("progress", True):
        command.append("--progress")
    return command


def _write_partition_artifacts(experiment_root: Path, payloads: dict) -> dict:
    paths = {}
    for index_name, payload in payloads.items():
        directory = experiment_root / "partitions" / index_name
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "manifest.json", payload)
        for partition, partition_record in payload["partitions"].items():
            for variant, question_ids in partition_record["question_ids"].items():
                path = directory / f"{partition}_{variant}.json"
                atomic_write_json(path, question_ids)
                paths[(index_name, partition, variant)] = path
    return paths


def _load_registry(path: Path, experiment_id: str, runs: list[dict]) -> dict:
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
        known = {entry["run_id"] for entry in registry.get("runs", [])}
        registry.setdefault("runs", []).extend(
            {"run_id": run["run_id"], "parameters": run["parameters"], "status": "pending"}
            for run in runs if run["run_id"] not in known
        )
        return registry
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "runs": [
            {"run_id": run["run_id"], "parameters": run["parameters"], "status": "pending"}
            for run in runs
        ],
    }


def _annotate_run_manifest(run_dir: Path, experiment_id: str, run: dict, pricing: dict) -> None:
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["experiment"] = {
        "experiment_id": experiment_id,
        "run_id": run["run_id"],
        "parameters": run["parameters"],
        "pricing": pricing,
    }
    environment_path = run_dir / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        manifest["code"] = {"git_commit": environment.get("git_commit")}
    atomic_write_json(path, manifest)


def _copy_locked_spec(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise ExperimentSpecError(f"existing experiment spec differs: {destination}")
        return
    shutil.copy2(source, destination)


def _execute(command: list[str], dry_run: bool = False) -> None:
    print(_format_command(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _run_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:64]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _nested(value: dict, path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _article_macro_values(rows: list[dict], metric: str) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        article = row.get("article_key")
        value = _nested(row, metric)
        if article and value is not None:
            grouped.setdefault(str(article), []).append(float(value))
    return [statistics.fmean(values) for values in grouped.values()]


def _percentile(values: list[float], fraction: float) -> float:
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _report_columns(report: dict) -> dict:
    columns = {}
    for section in ("coverage", "retrieval_initial", "retrieval", "reranker_delta", "qa", "citations", "ragas", "usage"):
        value = report.get(section)
        if isinstance(value, dict):
            _flatten(columns, section, value)
    latency = report.get("latency")
    if isinstance(latency, dict):
        _flatten(columns, "latency", latency)
    return columns


def _flatten(output: dict, prefix: str, value: dict) -> None:
    for key, item in value.items():
        name = f"{prefix}.{key}"
        if isinstance(item, dict):
            _flatten(output, name, item)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            output[name] = item


def _estimated_cost(report: dict, pricing: dict) -> float | None:
    usage = report.get("usage", {})
    input_price = pricing.get("input_per_million")
    output_price = pricing.get("output_per_million")
    if input_price is None or output_price is None or not usage:
        return None
    cost = (
        float(usage.get("input_tokens", 0)) * float(input_price)
        + float(usage.get("output_tokens", 0)) * float(output_price)
    ) / 1_000_000
    return round(cost, 6)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
