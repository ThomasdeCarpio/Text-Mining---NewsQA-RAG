#!/usr/bin/env python3
"""Collect Phase 1 comparisons, create CSVs/figures, and archive Kaggle outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from newsqa_rag.evaluation.phase1 import load_comparison_rows, write_rows_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, output = Path(args.experiments_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparisons = sorted(root.glob("phase1-*/comparison.json"))
    rows = load_comparison_rows(comparisons)
    write_rows_csv(rows, output / "combined_leaderboard.csv")
    for stage in ("round1", "round2", "round3", "final"):
        subset = [row for row in rows if f"phase1-{stage}" in str(row.get("run_id", ""))]
        if not subset:
            stage_paths = [path for path in comparisons if f"phase1-{stage}" in str(path)]
            subset = load_comparison_rows(stage_paths) if stage_paths else []
        if subset:
            write_rows_csv(subset, output / f"{stage}.csv")

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        frame = pd.DataFrame(rows)
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        if not frame.empty and "retrieval.mrr@5.mean" in frame:
            plot = frame[frame["variant"] == "original"].copy()
            plot["configuration"] = plot["index"].astype(str) + " / " + plot["reranker"].astype(str)
            plt.figure(figsize=(12, max(5, len(plot) * 0.25)))
            sns.barplot(data=plot, y="configuration", x="retrieval.mrr@5.mean", hue="partition")
            plt.tight_layout()
            plt.savefig(figures / "mrr_leaderboard.png", dpi=300)
            plt.close()
        if not frame.empty and {"latency.total.p50_ms", "retrieval.mrr@5.mean"} <= set(frame):
            plt.figure(figsize=(9, 6))
            sns.scatterplot(data=frame[frame["variant"] == "original"], x="latency.total.p50_ms", y="retrieval.mrr@5.mean", hue="reranker", style="partition", s=100)
            plt.tight_layout()
            plt.savefig(figures / "quality_latency.png", dpi=300)
            plt.close()
    except ImportError:
        pass

    (output / "comparison_sources.json").write_text(
        json.dumps([str(path) for path in comparisons], indent=2) + "\n", encoding="utf-8"
    )
    archive = shutil.make_archive(str(output.parent / "phase1_results_bundle"), "zip", output)
    print(json.dumps({"rows": len(rows), "archive": archive}, indent=2))


if __name__ == "__main__":
    main()
