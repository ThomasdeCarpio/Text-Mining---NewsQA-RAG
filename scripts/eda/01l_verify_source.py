"""Re-download the source dataset at the pinned revision and settle where the
truncation came from.

Two possibilities:
  A. HuggingFace already ships truncated contexts -> inherited, nothing to fix
  B. This repo's selection/staging step truncated them -> our bug, fixable
"""

from __future__ import annotations

import hashlib
import os
import statistics as st

import common as C

REPO = "lucadiliello/newsqa"
REVISION = "728e52920b8e4ffcfaad93fa47556f26a1d82546"

# Articles this session verified as truncated against the independent crawl.
SUSPECTS = {
    "newsqa_8db4b769ce5b0df1": "Gadhafi autopsy (evaluation, live-confirmed)",
    "newsqa_15c64f1bceed8c8f": "Iran nuclear",
    "newsqa_49d1fc683839f0a5": "Cairo taxi drivers",
    "newsqa_116a1e6e64da286c": "Syria unrest",
    "newsqa_4ba9b96b1a3b1a57": "Loire Valley hotels (6,173 chars missing)",
    "newsqa_8fdcd8874101869e": "5 ways to stay motivated",
}


def article_id(context: str) -> str:
    """Reproduce the pipeline's article id: newsqa_ + first 16 hex of sha256."""
    return "newsqa_" + hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    from datasets import load_dataset

    print(f"Downloading {REPO} @ {REVISION[:12]} ...")
    data = load_dataset(REPO, revision=REVISION)
    print(f"  splits: { {k: len(v) for k, v in data.items()} }\n")

    # Verify the id scheme reproduces on a known article before trusting matches.
    staged = {a["article_id"]: a for a in C.articles("evaluation") + C.articles("distractor")}
    probe = next(iter(staged.values()))
    print("  id scheme check:", article_id(probe["context"]) == probe["article_id"])

    hf_by_id: dict[str, str] = {}
    lengths = []
    for split in data:
        for row in data[split]:
            context = row["context"]
            lengths.append(len(context))
            key = article_id(context)
            if key not in hf_by_id:
                hf_by_id[key] = context

    print(f"\n  unique contexts on HuggingFace: {len(hf_by_id):,}")
    ordered = sorted(lengths)
    print(f"  HF context length (chars): median {int(st.median(ordered)):,}  "
          f"p95 {ordered[int(.95*len(ordered))]:,}  max {ordered[-1]:,}")

    staged_lengths = sorted(len(a["context"]) for a in staged.values())
    print(f"  staged corpus     (chars): median {int(st.median(staged_lengths)):,}  "
          f"p95 {staged_lengths[int(.95*len(staged_lengths))]:,}  max {staged_lengths[-1]:,}")

    print()
    print("=" * 78)
    print("DO THE SUSPECT ARTICLES DIFFER BETWEEN HUGGINGFACE AND OUR STAGED COPY?")
    print("=" * 78)
    verdicts = {}
    for aid, label in SUSPECTS.items():
        ours = staged.get(aid)
        theirs = hf_by_id.get(aid)
        print(f"\n  {label}")
        print(f"    article_id : {aid}")
        if ours is None:
            print("    NOT in our staged corpus")
            continue
        if theirs is None:
            print("    NOT found on HuggingFace by id  (id is a hash of the exact text,")
            print("    so a mismatch here means the text differs somewhere)")
            verdicts[aid] = "id not found on HF"
            continue
        same = ours["context"] == theirs
        print(f"    staged : {len(ours['context']):,} chars")
        print(f"    HF     : {len(theirs):,} chars")
        print(f"    identical: {same}")
        verdicts[aid] = "identical" if same else f"differs ({len(theirs)-len(ours['context']):+,} chars)"

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    identical = sum(1 for v in verdicts.values() if v == "identical")
    print(f"  suspects checked : {len(verdicts)}")
    print(f"  identical to HF  : {identical}")
    if identical == len(verdicts) and verdicts:
        print("""
  Every suspect article is byte-identical to what HuggingFace ships.
  The truncation was NOT introduced by this repo. It is inherited from the
  source dataset, and no amount of re-downloading or re-unzipping will
  recover the missing text.""")
    else:
        print("""
  At least one article differs from the HuggingFace source. That points at
  the download/unzip/staging step rather than the dataset - worth chasing.""")

    C.save("01l_verify_source", {
        "hf_unique_contexts": len(hf_by_id),
        "hf_max_chars": ordered[-1],
        "staged_max_chars": staged_lengths[-1],
        "verdicts": verdicts,
    })
    print("\nSaved -> out/01l_verify_source.json")


if __name__ == "__main__":
    main()
