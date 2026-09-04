"""Restore the text that was cut off the end of each benchmark article.

The corpus was built from a CNN scrape that cropped stories around 640-680
words. For most articles the archived page is still in data/cnn_downloads.tgz,
so the missing tail can be put back. Articles with no usable page are kept
exactly as they are - this pass never removes or rewrites existing text.

APPEND-ONLY, AND WHY THAT MATTERS
---------------------------------
Truncation means our text is a PREFIX of the page's text. So restoration is a
pure append:

    restored = our text  +  the page's tail

Every character offset in the original text therefore still points at the same
character. The evidence spans in the testsets need no remapping at all, which
is what keeps the ground truth trustworthy. When the page has extra text BEFORE
our article (a dateline or promo box the scrape dropped), that leading text is
deliberately discarded rather than prepended - keeping it would shift every
offset in the article for no retrieval benefit.

PAIRING
-------
Two anchors, union of both, same as the EDA:
  head   the first 60 characters of the article text
  mid    any 100-character window from offsets 200..900 of the article

Both are EXACT matching on whitespace-normalised, lower-cased text - a hash
lookup then a substring check. No similarity threshold anywhere.

Two passes, because NewsCleaner costs ~90 ms/page and there are 92,579 pages:
  pass 1  cheap regex extraction, high recall, finds candidate pages
  pass 2  NewsCleaner on the candidates only; every reported number comes from
          this pass, which re-matches from scratch and drops pass 1's guesses

This script only rewrites article text. It contains no chunking and no chunk-ID
logic on purpose: the repository already relabels chunks when a dataset is
rebuilt. derive_chunked_testsets() re-chunks each article, chunk_char_ranges()
locates every chunk back in the article text, and map_spans_to_chunks() re-maps
the evidence spans onto the new chunk IDs (backend/newsqa_rag/evaluation/
testset.py), all driven by prepare_evaluation_dataset.py build-baseline. Since
the published bundle ships no chunks at all, restored text picks up correct
chunk IDs automatically the next time the dataset is materialized.

Output: data/evaluation/newsqa_200_11064_restored/ - a copy of the staging tree
with the corpus JSONL rewritten. Publish it with:

    python scripts/publish_evaluation_dataset.py \\
      --source-root data/evaluation/newsqa_200_11064_restored \\
      --version v2.0.0 --dry-run --zip
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
import statistics as st
import sys
import tarfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "backend"))

ARCHIVE = PROJECT / "data" / "cnn_downloads.tgz"
SOURCE = PROJECT / "data" / "evaluation" / "newsqa_200_11064"
TARGET = PROJECT / "data" / "evaluation" / "newsqa_200_11064_restored"

FURNITURE = 40      # a tail this short is page chrome, not a sentence
KEY = 60            # characters of an article head used as its signature
WINDOW = 100        # mid-anchor window length (calibrated in the EDA)
INDEX_LO, INDEX_HI = 200, 900
PAGE_STRIDE = 150
# NewsCleaner costs ~90 ms a page and is pure-Python CPU work, so the pass is
# embarrassingly parallel. One core of fourteen turned a 9-minute job into 51.
WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))
# Worst to best: an article keeps the best verdict any of its pages produced.
RANK = {"failed": 0, "diverged": 1, "intact": 2, "restored": 3}

PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S)
TAG = re.compile(r"<[^>]+>")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def norm_with_map(raw: str) -> tuple[str, list[int]]:
    """norm(raw), plus normalised index -> raw index, so a match can be cut.

    The last entry is a sentinel equal to len(raw), so the map can be indexed at
    len(normalised) to mean "everything after the match".
    """
    out: list[str] = []
    idx: list[int] = []
    i, n = 0, len(raw)
    while i < n:
        if raw[i].isspace():
            j = i
            while j < n and raw[j].isspace():
                j += 1
            if out and j < n:               # interior run collapses to one space
                out.append(" ")
                idx.append(i)
            i = j
        else:
            lowered = raw[i].lower()        # a few code points lower to >1 char
            out.append(lowered)
            idx.extend([i] * len(lowered))
            i += 1
    idx.append(n)
    return "".join(out), idx


def candidate_heads(html: str) -> list[str]:
    """Cheap head signatures for pass 1 only - never used for a reported number."""
    parts = [p for p in (norm(TAG.sub(" ", p)) for p in PARA.findall(html)) if p]
    heads = []
    for start in range(min(8, len(parts))):
        joined = " ".join(parts[start:start + 6])
        if len(joined) >= KEY:
            heads.append(joined[:KEY])
    return heads


def extract_cheap(html: str) -> str:
    """Boilerplate-inclusive text. Only ever adds prose, never drops it."""
    return norm(" ".join(TAG.sub(" ", p) for p in PARA.findall(html)))


def pages():
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".html"):
                handle = tar.extractfile(member)
                if handle is not None:
                    yield member.name, handle.read().decode("utf-8", "replace")


_CLEANER = None


def _init_worker() -> None:
    """One NewsCleaner per process; building it per page would dominate."""
    global _CLEANER
    from newsqa_rag.ingestion.cleaner import NewsCleaner
    _CLEANER = NewsCleaner()


def examine(job: tuple[str, str, str, str]) -> tuple[str, str, dict | None]:
    """Decide what one archived page offers one article. Runs in a worker.

    Returns (article_id, verdict, record). The verdict is why this page did or
    did not help, so the caller can report articles that were never truncated
    separately from articles it genuinely failed on.
    """
    name, html, aid, ours = job
    try:
        raw = _CLEANER.clean_html_string(html)["text"]
    except Exception:
        return aid, "failed", None
    theirs, index = norm_with_map(raw)
    if not theirs or theirs != norm(raw):   # the offset map must be exact
        return aid, "failed", None
    at = theirs.find(ours)
    if at < 0:
        return aid, "diverged", None
    end = at + len(ours)
    trailing = len(theirs) - end
    if trailing <= FURNITURE:
        return aid, "intact", None
    return aid, "restored", {
        "article_id": aid, "leading": at, "gap": trailing,
        "tail": raw[index[end]:].lstrip(), "page": name.split("/")[-1],
    }


def jobs(candidates: dict[str, str], ours_norm: dict[str, str]):
    """Stream candidate pages to the pool; never hold the archive in memory."""
    for name, html in pages():
        aid = candidates.get(name)
        if aid is not None:
            yield name, html, aid, ours_norm[aid]


def describe(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    o = sorted(values)
    n = len(o)
    return {"n": n, "mean": round(st.fmean(o), 1), "median": o[n // 2],
            "p25": o[n // 4], "p75": o[3 * n // 4], "p90": o[int(0.9 * n)], "max": o[-1]}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    corpus = {}
    for role in ("evaluation", "distractor"):
        for row in read_jsonl(SOURCE / "staging" / "corpus" / f"{role}_articles.jsonl"):
            corpus[row["article_id"]] = row
    ours_norm = {aid: norm(row["context"]) for aid, row in corpus.items()}
    print(f"benchmark articles : {len(corpus):,}")

    by_head: dict[str, str] = {}
    windows: dict[int, str] = {}
    for aid, text in ours_norm.items():
        by_head.setdefault(text[:KEY], aid)
        stop = min(INDEX_HI, max(INDEX_LO + 1, len(text) - WINDOW))
        for start in range(INDEX_LO, stop):
            windows.setdefault(hash(text[start:start + WINDOW]), aid)
    print(f"anchors            : {len(by_head):,} heads, {len(windows):,} mid-windows\n")

    print("PASS 1 - cheap scan for candidate pages")
    started = time.perf_counter()
    candidates: dict[str, str] = {}
    scanned = 0
    for name, html in pages():
        scanned += 1
        hit = None
        for head in candidate_heads(html):
            hit = by_head.get(head)
            if hit:
                break
        if hit is None:
            text = extract_cheap(html)
            for start in range(0, max(1, len(text) - WINDOW), PAGE_STRIDE):
                hit = windows.get(hash(text[start:start + WINDOW]))
                if hit:
                    break
        if hit is not None:
            candidates[name] = hit
        if scanned % 20000 == 0:
            print(f"  {scanned:,} pages, {len(candidates):,} candidates", flush=True)
    print(f"  scanned {scanned:,} pages in {time.perf_counter()-started:.0f}s")
    print(f"  candidates: {len(candidates):,} pages, "
          f"{len(set(candidates.values())):,} distinct articles\n")

    print(f"PASS 2 - NewsCleaner on the candidates, across {WORKERS} processes")
    started = time.perf_counter()
    best: dict[str, dict] = {}
    outcome: dict[str, str] = {}        # article_id -> best outcome seen
    tally = {"failed": 0, "diverged": 0, "intact": 0, "restored": 0}
    done = 0
    with mp.Pool(WORKERS, initializer=_init_worker) as pool:
        for result in pool.imap_unordered(examine, jobs(candidates, ours_norm),
                                          chunksize=8):
            done += 1
            aid, verdict, record = result
            tally[verdict] += 1
            if RANK[verdict] > RANK.get(outcome.get(aid, "failed"), -1):
                outcome[aid] = verdict
            if record is not None:
                prior = best.get(aid)
                if prior is None or record["gap"] > prior["gap"]:
                    best[aid] = record
            if done % 2000 == 0:
                print(f"  {done:,}/{len(candidates):,} cleaned, "
                      f"{len(best):,} restorable", flush=True)
    elapsed = (time.perf_counter() - started) / 60
    print(f"  cleaned {done:,} pages in {elapsed:.1f} min"
          f"  ({done/max(elapsed*60, 1e-9):.0f} pages/s)")
    print(f"  per page: {tally['failed']:,} failed to parse, "
          f"{tally['diverged']:,} text not contained, "
          f"{tally['intact']:,} nothing extra, {tally['restored']:,} had a tail")

    # Why each article ended where it did. Without this split, "unchanged" reads
    # as failure when most of it is articles that were never truncated.
    reasons = {"restored": 0, "intact": 0, "diverged": 0, "failed": 0, "unpaired": 0}
    for aid in ours_norm:
        reasons[outcome.get(aid, "unpaired")] += 1
    print("\n  WHY EACH ARTICLE ENDED WHERE IT DID")
    labels = {
        "restored": "restored - the page had more text, appended",
        "intact": "already complete - page had nothing extra",
        "diverged": "page text differs mid-article, unsafe to append",
        "failed": "page would not parse",
        "unpaired": "no archived page matched this article",
    }
    for key in ("restored", "intact", "diverged", "failed", "unpaired"):
        print(f"    {reasons[key]:>6,}  {reasons[key]/len(ours_norm):>5.1%}"
              f"  {labels[key]}")

    print("\n" + "=" * 74)
    print("RESTORATION")
    print("=" * 74)
    gaps = [r["gap"] for r in best.values()]
    stats = describe(gaps)
    print(f"  articles restored          : {len(best):,} of {len(corpus):,}"
          f"  ({len(best)/len(corpus):.1%})")
    print(f"  articles kept unchanged    : {len(corpus)-len(best):,}")
    print(f"  characters appended        : {sum(gaps):,}")
    if gaps:
        print(f"  per article: median {stats['median']:,}  mean {stats['mean']:,.0f}"
              f"  p90 {stats['p90']:,}  max {stats['max']:,}")
    dropped = [r for r in best.values() if r["leading"] > 0]
    print(f"  pages with text before ours: {len(dropped):,} (that lead is dropped,"
          f" so offsets never move)")

    # ---- write the restored staging tree ---------------------------------
    # Streamed line by line, source handle straight to destination handle. The
    # corpus is 40 MB of JSONL and the window index above is still holding about
    # a gigabyte, so neither file is worth materialising as a list of rows.
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(SOURCE / "staging", TARGET / "staging",
                    ignore=shutil.ignore_patterns("*_articles.jsonl"))
    by_role = {"evaluation": 0, "distractor": 0}
    for role in ("evaluation", "distractor"):
        name = f"{role}_articles.jsonl"
        source = SOURCE / "staging" / "corpus" / name
        target = TARGET / "staging" / "corpus" / name
        with source.open(encoding="utf-8") as reader, \
                target.open("w", encoding="utf-8", newline="\n") as writer:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                record = best.get(row["article_id"])
                if record is not None:
                    original = row["context"]
                    row["context"] = original + "\n\n" + record["tail"]
                    assert row["context"].startswith(original)   # offsets intact
                    row["restored_chars"] = record["gap"]
                    by_role[role] += 1
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    # scripts/publish_evaluation_dataset.py validates a bundle against the
    # resolved testset that the approved question deduplication was bound to.
    # That binding is over questions, which restoration does not touch, so the
    # original file is the correct one to carry across unchanged.
    resolved = TARGET / "final" / "testset_resolved.jsonl"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "final" / "testset_resolved.jsonl", resolved)
    print(f"\n  evaluation articles restored : {by_role['evaluation']:,}")
    print(f"  distractor articles restored : {by_role['distractor']:,}")
    print(f"  wrote {TARGET.name}/staging/corpus/")

    report = {
        "source": SOURCE.name,
        "target": TARGET.name,
        "corpus": len(corpus),
        "restored": len(best),
        "unchanged": len(corpus) - len(best),
        "by_role": by_role,
        "chars_appended": sum(gaps),
        "gap": stats,
        "leading_dropped": len(dropped),
        "pages_scanned": scanned,
        "pages_cleaned": done,
        "page_outcomes": tally,
        "article_outcomes": reasons,
        "workers": WORKERS,
        "pass2_minutes": round(elapsed, 1),
        "method": "append-only; evidence offsets unchanged by construction",
    }
    (TARGET / "restore_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {TARGET.name}/restore_report.json")


def selfcheck() -> None:
    """The offset map is the only piece that can silently corrupt spans."""
    for raw in ["  Hello   World\n\n\nAgain  ", "a\tb\nc", "One two three.",
                "\n\nLead\n\n\n\nBody text here.\n"]:
        text, index = norm_with_map(raw)
        assert text == norm(raw), (text, norm(raw))
        for i, ch in enumerate(text):
            if ch != " ":
                assert raw[index[i]].lower().startswith(ch), (i, ch, raw[index[i]])
        assert index[len(text)] == len(raw)
    # a prefix cut must leave the head byte-identical and recover the tail
    raw = "Lead para.\n\n\n\nSecond para.\n\n\n\nThird para."
    text, index = norm_with_map(raw)
    head = norm("Lead para.\n\n\n\nSecond para.")
    at = text.find(head)
    assert at == 0
    assert raw[index[at + len(head)]:].lstrip() == "Third para."
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
