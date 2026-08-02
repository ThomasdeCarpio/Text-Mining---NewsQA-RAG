#!/usr/bin/env python3
"""Run or preview an experiment matrix."""

import argparse

from newsqa_rag.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="Experiment YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only")
    args = parser.parse_args()
    print(f"Experiment directory: {run_experiment(args.spec, dry_run=args.dry_run)}")


if __name__ == "__main__":
    main()
