#!/usr/bin/env python3
"""Run serial, cache-free latency calibration from completed Phase 1 specs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))
from newsqa_rag.evaluation.phase1 import load_comparison_rows, write_rows_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-eval", type=int, default=100)
    args = parser.parse_args()
    generated, comparisons = [], []
    for repeat in range(args.repeats):
        for spec_path in map(Path, args.specs):
            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            spec["experiment"]["id"] = f"latency-r{repeat + 1}-{spec['experiment']['id']}"
            spec["runs"] = [row for row in spec["runs"] if row.get("variant") == "original"]
            spec["runtime"].update({"n_eval": args.n_eval, "warmup_queries": 10})
            spec["runtime"].pop("shared_retrieval_cache", None)
            target = Path(args.output).parent / "latency_specs" / f"r{repeat + 1}_{spec_path.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            generated.append(target)
            subprocess.run([sys.executable, "scripts/run_experiment.py", str(target)], cwd=PROJECT_ROOT, check=True)
            directory = PROJECT_ROOT / spec.get("output_dir", "outputs/experiments") / spec["experiment"]["id"]
            subprocess.run([sys.executable, "scripts/summarize_experiments.py", str(directory)], cwd=PROJECT_ROOT, check=True)
            comparisons.append(directory / "comparison.json")
    rows = load_comparison_rows(comparisons)
    write_rows_csv(rows, args.output)
    print(json.dumps({"calibration_runs": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
