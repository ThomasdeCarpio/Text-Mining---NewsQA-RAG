#!/usr/bin/env python3
"""Collect the indexes a Kaggle run built into one clean, checksummed dataset.

Building the Phase 1 indexes costs real GPU time - four dense Chroma
collections and four sparse indexes over 22,766 chunks - and a Kaggle session
throws all of it away. This packages what a run left behind so the next session
attaches it as an input instead of rebuilding.

It discovers artifacts rather than hardcoding paths: any directory holding an
`index_manifest.json` is an index bundle, wherever the run happened to put it.
So it works on `/kaggle/working/newsqa_phase1`, on an attached input, or on an
unpacked checkpoint, without being told the layout.

    # see what is there and how big, without writing anything
    python scripts/package_index_artifacts.py --source /kaggle/working/newsqa_phase1 --dry-run

    # package everything
    python scripts/package_index_artifacts.py \
        --source /kaggle/working/newsqa_phase1 \
        --output /kaggle/working/phase1-indexes

    # just the locked retriever, and zip it
    python scripts/package_index_artifacts.py \
        --source /kaggle/working/newsqa_phase1 \
        --output /kaggle/working/phase1-indexes \
        --only bge_m3 --zip

Every file is recorded in manifest.json with its SHA-256, so a later run can
prove the index it loaded is the index that was built.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.testset import sha256_file  # noqa: E402

# Directories that are caches or scratch, never artifacts worth shipping.
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "hf_cache", "retrieval_cache"}
SKIP_SUFFIXES = {".tmp", ".lock", ".pyc"}

README = """# {name}

Retrieval indexes built by the NewsQA RAG Phase 1 pipeline, packaged so a later
session can attach them instead of spending GPU time rebuilding.

Generated {generated} from `{source}`.

## Contents

{contents}

Total: **{total_files} files, {total_size}**.

## Verifying

`manifest.json` records every file with its SHA-256 and byte size. Check before
use - an index that does not match the manifest was not built by this run:

```python
import json, hashlib
from pathlib import Path

manifest = json.loads(Path("manifest.json").read_text())
for record in manifest["files"]:
    path = Path(record["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == record["sha256"], path
print(f"{{len(manifest['files'])}} files verified")
```

## Loading

Point the pipeline at the bundle directory. Each index bundle keeps its own
`index_manifest.json`, `config_*.yaml` and `variant_*.json`, so the paths it
records stay internally consistent.

## Security note

Sparse indexes are Python **pickle** files (`*.pkl`). Unpickling executes
arbitrary code. Load them only after the checksum above matches, and only from
a source you control. Never load a `.pkl` from an untrusted copy of this
dataset.
"""


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def interesting(path: Path) -> bool:
    if path.suffix in SKIP_SUFFIXES:
        return False
    return not any(part in SKIP_DIRS for part in path.parts)


def find_bundles(source: Path) -> list[Path]:
    """An index bundle is any directory holding an index_manifest.json."""
    seen = {manifest.parent for manifest in source.rglob("index_manifest.json")
            if interesting(manifest)}
    # A nested bundle would be copied twice; keep only the outermost.
    return sorted(d for d in seen if not any(p in seen for p in d.parents))


def bundle_files(bundle: Path) -> list[Path]:
    return sorted(p for p in bundle.rglob("*") if p.is_file() and interesting(p))


def describe(bundle: Path) -> str:
    """Read the bundle's own manifest for a human-readable label."""
    try:
        blob = json.loads((bundle / "index_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable index_manifest.json"
    parts = []
    for key, label in (("dense_indexes", "dense"), ("sparse_indexes", "sparse")):
        for name in (blob.get(key) or {}):
            parts.append(f"{label}: {name}")
    return ", ".join(parts) or "no indexes listed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="Run root to scan, e.g. /kaggle/working/newsqa_phase1")
    parser.add_argument("--output", help="Directory to write the dataset into")
    parser.add_argument("--name", default=None, help="Dataset name (default: output dir name)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Keep only bundles whose path contains one of these substrings")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="Additional files or directories to include verbatim")
    parser.add_argument("--zip", action="store_true", help="Also write <output>.zip")
    parser.add_argument("--dry-run", action="store_true", help="List what would be packaged")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        parser.error(f"source does not exist: {source}")

    bundles = find_bundles(source)
    if args.only:
        bundles = [b for b in bundles if any(token in str(b) for token in args.only)]
    if not bundles and not args.extra:
        parser.error(f"no index_manifest.json found under {source} - is that the run root?")

    print(f"Scanning {source}\n")
    plan, total_bytes = [], 0
    for bundle in bundles:
        files = bundle_files(bundle)
        size = sum(p.stat().st_size for p in files)
        total_bytes += size
        plan.append((bundle, files))
        print(f"  {bundle.relative_to(source)}")
        print(f"      {describe(bundle)}")
        print(f"      {len(files)} files, {human(size)}")

    extras = []
    for raw in args.extra:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"  [skip] missing extra: {path}")
            continue
        files = [path] if path.is_file() else bundle_files(path)
        size = sum(p.stat().st_size for p in files)
        total_bytes += size
        extras.append((path, files))
        print(f"  {path.name} (extra): {len(files)} files, {human(size)}")

    print(f"\nTotal: {human(total_bytes)} across {len(plan) + len(extras)} bundles")
    if args.dry_run:
        print("\nDry run - nothing written.")
        return
    if not args.output:
        parser.error("--output is required unless --dry-run")

    output = Path(args.output).expanduser().resolve()
    name = args.name or output.name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    records, contents = [], []
    for root, files in plan + extras:
        # A single-file extra is copied to the top level under its own name; a
        # directory keeps its internal layout under a folder of the same name.
        base = root.parent if root.is_file() else root
        size = 0
        for path in files:
            target = output / root.name / path.relative_to(base) if root.is_dir() else output / root.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            record = {"path": target.relative_to(output).as_posix(),
                      "bytes": target.stat().st_size,
                      "sha256": sha256_file(target)}
            records.append(record)
            size += record["bytes"]
        label = describe(root) if root.is_dir() and (root / "index_manifest.json").exists() else "extra"
        suffix = "/" if root.is_dir() else ""
        contents.append(f"- `{root.name}{suffix}` — {label} ({human(size)})")

    manifest = {
        "schema_version": 1,
        "name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "file_count": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "files": sorted(records, key=lambda r: r["path"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(README.format(
        name=name, generated=manifest["generated_at"], source=source,
        contents="\n".join(contents), total_files=len(records),
        total_size=human(manifest["total_bytes"])), encoding="utf-8")

    print(f"\nWrote {output}")
    print(f"  {len(records)} files, {human(manifest['total_bytes'])}, manifest.json + README.md")

    if args.zip:
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(p for p in output.rglob("*") if p.is_file()):
                handle.write(path, path.relative_to(output))
        print(f"  {archive} ({human(archive.stat().st_size)})")

    print(f"""
Upload
------
Hugging Face (private; needs HF_TOKEN):

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo("<user>/{name}", repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path="{output}", repo_id="<user>/{name}", repo_type="dataset")

Kaggle: "Save Version" keeps /kaggle/working as this notebook's output, and a
later notebook attaches it with Add Input. For a standalone dataset instead:

    kaggle datasets init -p {output}
    # edit dataset-metadata.json, then
    kaggle datasets create -p {output} --dir-mode zip
""")


if __name__ == "__main__":
    main()
