"""Local experiment control used by both the admin API and dashboard."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from newsqa_rag.experiments import (
    build_article_partitions,
    expand_run_matrix,
    load_experiment_spec,
    run_experiment,
    summarize_experiment,
)
from newsqa_rag.model_gateway import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
_running: set[str] = set()
_errors: dict[str, str] = {}
_lock = threading.Lock()


def _spec_path(filename: str) -> Path:
    if Path(filename).name != filename or Path(filename).suffix not in {".yaml", ".yml"}:
        raise ValueError("Invalid experiment filename")
    path = CONFIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(filename)
    return path


def _output_dir(spec: dict) -> Path:
    root = Path(spec.get("output_dir", "outputs/experiments"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if not root.is_relative_to(PROJECT_ROOT):
        raise ValueError("Experiment output_dir must stay inside the project")
    return root / spec["experiment"]["id"]


def _status(filename: str, output_dir: Path) -> tuple[str, dict[str, int]]:
    with _lock:
        running = filename in _running
        error = _errors.get(filename)
    registry_path = output_dir / "registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {"runs": []}
    )
    counts: dict[str, int] = {}
    for run in registry.get("runs", []):
        state = str(run.get("status", "pending"))
        counts[state] = counts.get(state, 0) + 1
    if running:
        return "running", counts
    if error or counts.get("failed") or counts.get("interrupted"):
        return "failed", counts
    if counts and counts.get("complete") == sum(counts.values()):
        return "complete", counts
    return "pending", counts


def _partition_counts(payload: dict) -> dict:
    return {
        name: {
            "articles": len(partition["article_ids"]),
            "questions": {
                variant: len(question_ids)
                for variant, question_ids in partition["question_ids"].items()
            },
        }
        for name, partition in payload["partitions"].items()
    }


def describe_experiment(filename: str, *, include_runs: bool = False) -> dict:
    spec = load_experiment_spec(_spec_path(filename))
    runs = expand_run_matrix(spec)
    output_dir = _output_dir(spec)
    status, counts = _status(filename, output_dir)
    description = {
        "filename": filename,
        "id": spec["experiment"]["id"],
        "name": spec["experiment"].get("name", spec["experiment"]["id"]),
        "description": spec["experiment"].get("description", ""),
        "run_count": len(runs),
        "status": status,
        "status_counts": counts,
        "result_ready": (output_dir / "comparison.json").exists(),
        "error": _errors.get(filename),
    }
    if include_runs:
        description["runs"] = runs
        description["partitions"] = {
            name: _partition_counts(
                build_article_partitions(
                    {
                        variant: path if Path(path).is_absolute() else PROJECT_ROOT / path
                        for variant, path in index["testsets"].items()
                    },
                    int(spec["dataset"].get("development_articles", 50)),
                    int(spec.get("seed", 42)),
                    str(spec["dataset"].get("article_field", "article_key")),
                )
            )
            for name, index in spec["dataset"]["indexes"].items()
        }
    return description


def list_experiments() -> list[dict]:
    if not CONFIG_DIR.exists():
        return []
    return [
        describe_experiment(path.name)
        for path in sorted([*CONFIG_DIR.glob("*.yaml"), *CONFIG_DIR.glob("*.yml")])
    ]


def start_experiment(filename: str) -> dict:
    path = _spec_path(filename)
    describe_experiment(filename, include_runs=True)
    with _lock:
        if filename in _running:
            raise RuntimeError("Experiment is already running")
        _running.add(filename)
        _errors.pop(filename, None)

    def work() -> None:
        try:
            output_dir = run_experiment(path)
            summarize_experiment(output_dir)
        except Exception as exc:  # surfaced by the status endpoint
            with _lock:
                _errors[filename] = f"{exc.__class__.__name__}: {exc}"
        finally:
            with _lock:
                _running.discard(filename)

    threading.Thread(target=work, name=f"experiment-{path.stem}", daemon=True).start()
    return describe_experiment(filename)


def get_results(filename: str) -> dict:
    spec = load_experiment_spec(_spec_path(filename))
    output_dir = _output_dir(spec)
    path = output_dir / "comparison.json"
    if not path.exists():
        raise FileNotFoundError("Experiment results are not ready")
    result = json.loads(path.read_text(encoding="utf-8"))
    registry_path = output_dir / "registry.json"
    result["history"] = (
        json.loads(registry_path.read_text(encoding="utf-8")).get("runs", [])
        if registry_path.exists()
        else []
    )
    return result


def get_run_detail(filename: str, run_id: str) -> dict:
    spec = load_experiment_spec(_spec_path(filename))
    runs = {run["run_id"]: run for run in expand_run_matrix(spec)}
    if run_id not in runs:
        raise FileNotFoundError("Unknown experiment run")
    report_path = _output_dir(spec) / run_id / "report.json"
    if not report_path.exists():
        raise FileNotFoundError("Run report is not ready")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "parameters": runs[run_id]["parameters"],
        "generated_at": report.get("generated_at"),
        "coverage": report.get("coverage", {}),
        "failures": report.get("failures", []),
    }
