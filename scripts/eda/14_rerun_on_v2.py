"""Re-run the corpus-dependent EDA analyses against the restored v2.0.0 corpus.

Same scripts, same methodology - only common.py's paths are redirected, so any
difference in the output is a difference in the data, not in the method.
"""
import sys, pathlib, runpy

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs" / "eda_v2"
sys.path.insert(0, str(ROOT / "scripts" / "eda"))

import common as C

C.FINAL = HERE / "final"
C.STAGING = ROOT / "data/evaluation/newsqa_200_11064_restored/staging"
C.OUT = HERE
C.OUT.mkdir(parents=True, exist_ok=True)
C.VARIANTS = {
    "original": C.FINAL / "testset_original.jsonl",
    "reviewed_original": C.FINAL / "testset_reviewed_original.jsonl",
    "resolved": C.FINAL / "testset_resolved.jsonl",
    "clarified": C.FINAL / "testset_clarified.jsonl",
}

for name in sys.argv[1:]:
    print("\n" + "#" * 74)
    print(f"# {name}  ON RESTORED v2.0.0")
    print("#" * 74)
    runpy.run_path(str(ROOT / "scripts" / "eda" / f"{name}.py"), run_name="__main__")
