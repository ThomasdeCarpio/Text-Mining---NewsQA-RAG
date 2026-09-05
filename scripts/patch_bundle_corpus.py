"""Put the restored article text into the published bundle, changing nothing else.

The alternative - re-exporting a bundle from scratch with
publish_evaluation_dataset.py - re-runs validate_source_dataset, which re-checks
the human dedup approval seal. That seal is currently broken in this repository
(the approval records a hash of semantic_clusters.json that no longer matches the
file, for both the published corpus and this one), and re-signing it is a human
decision, not a build step.

So this does not rebuild anything. It takes the bundle already published on
Hugging Face, replaces the two corpus files with the restored ones, and re-seals
the manifest. Questions, review, dedup decisions and the human approval are
carried across byte for byte - whatever state that seal is in, it stays exactly
as published.

    python scripts/patch_bundle_corpus.py --bundle <downloaded v1.0.0 dir or .zip>

Chunks are not in the bundle and never were, so nothing here touches chunk IDs.
Re-indexing happens the usual way, when the new revision is materialized:
materialize_evaluation_dataset.py re-chunks the restored articles and re-maps
every evidence span onto fresh chunk IDs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "common"))

from newsqa_rag.evaluation.cloud_dataset import (  # noqa: E402
    _artifact_record,
    verify_canonical_bundle,
)
from newsqa_rag.evaluation.testset import (  # noqa: E402
    DatasetBuildError,
    canonical_json,
    sha256_file,
    sha256_text,
)

RESTORED = PROJECT / "data" / "evaluation" / "newsqa_200_11064_restored"
PUBLISHED = PROJECT / "data" / "evaluation" / "newsqa_200_11064"
CORPUS = {
    "evaluation_articles": "corpus/evaluation_articles.jsonl",
    "distractor_articles": "corpus/distractor_articles.jsonl",
}
CHUNK = 1 << 20


def unpack(bundle: Path, workdir: Path) -> Path:
    if bundle.is_dir():
        return bundle
    if bundle.suffix != ".zip":
        raise DatasetBuildError(f"--bundle must be a directory or a .zip: {bundle}")
    target = workdir / "downloaded"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(target)
    # A zip made from the bundle root has cloud_manifest.json at the top; one
    # made from its parent nests it a level down.
    if (target / "cloud_manifest.json").exists():
        return target
    nested = [p.parent for p in target.rglob("cloud_manifest.json")]
    if len(nested) != 1:
        raise DatasetBuildError(f"Could not find cloud_manifest.json inside {bundle}")
    return nested[0]


def copy_streamed(source: Path, destination: Path) -> None:
    """Byte-for-byte, a megabyte at a time - these files are tens of MB."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, CHUNK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="The published bundle downloaded from Hugging Face")
    parser.add_argument("--restored-root", type=Path, default=RESTORED)
    parser.add_argument("--version", default="v2.0.0")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "data" / "evaluation_exports")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    workdir = args.output / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    source_bundle = unpack(args.bundle.resolve(), workdir)

    print(f"Verifying the published bundle: {source_bundle}")
    published = verify_canonical_bundle(source_bundle)
    print(f"  version {published['dataset_version']}  "
          f"dataset_sha256 {published['dataset_sha256'][:16]}...")

    # The restored files were produced by appending to the LOCAL staging corpus.
    # If the published bundle holds different text, appending would silently
    # replace the published articles with local ones instead of extending them.
    for key, relative in CORPUS.items():
        local = PUBLISHED / "staging" / relative
        if sha256_file(source_bundle / relative) != sha256_file(local):
            raise DatasetBuildError(
                f"{relative} in the bundle differs from {local}. The restored text "
                "was built by appending to the local copy, so it cannot be applied "
                "to a different base. Re-run scripts/restore_corpus.py against the "
                "downloaded bundle first."
            )
    print("  base corpus matches the local copy the restoration was built on")

    target = args.output / "newsqa_200_11064" / args.version
    if target.exists():
        shutil.rmtree(target)
    # .cache is huggingface_hub's local download bookkeeping, created by the
    # download and no part of the dataset. Uploading it would publish stale
    # pointers to the previous revision's blobs.
    shutil.copytree(source_bundle, target,
                    ignore=shutil.ignore_patterns(".cache"))

    manifest = json.loads((target / "cloud_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    appended = 0
    for key, relative in CORPUS.items():
        restored = args.restored_root / "staging" / relative
        if not restored.exists():
            raise DatasetBuildError(f"No restored corpus at {restored}")
        before = artifacts[key]["bytes"]
        copy_streamed(restored, target / relative)
        artifacts[key] = _artifact_record(target / relative, relative)
        appended += artifacts[key]["bytes"] - before
        print(f"  {relative}: {before:,} -> {artifacts[key]['bytes']:,} bytes "
              f"({artifacts[key]['records']:,} records)")

    report = args.restored_root / "restore_report.json"
    manifest["dataset_version"] = args.version
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["derived_from"] = {
        "dataset_version": published["dataset_version"],
        "dataset_sha256": published["dataset_sha256"],
        "change": "article text restored from the archived CNN pages; append-only, "
                  "so every evidence character offset is unchanged",
        "tool": "scripts/restore_corpus.py",
        "restoration": json.loads(report.read_text(encoding="utf-8")) if report.exists() else None,
    }
    manifest["dataset_sha256"] = sha256_text(canonical_json(artifacts))
    (target / "cloud_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print(f"\nRe-verifying the patched bundle: {target}")
    verify_canonical_bundle(target)
    print(f"  passed  dataset_sha256 {manifest['dataset_sha256'][:16]}...")
    print(f"  corpus grew by {appended:,} bytes")

    if not args.no_zip:
        archive = shutil.make_archive(
            str(args.output / f"newsqa_200_11064_{args.version}"), "zip", root_dir=str(target))
        size = Path(archive).stat().st_size
        print(f"\nUPLOAD THIS:\n  {archive}\n  {size:,} bytes  "
              f"sha256 {sha256_file(Path(archive))[:16]}...")
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
