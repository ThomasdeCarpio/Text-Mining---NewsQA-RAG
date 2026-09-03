"""Measure the overlap between our text and the page, then pick the window size.

10 anchors pairing on a 200-character window taken from offset 300. Those two
numbers were guessed. This script measures what they should be, by asking of
real pairs:

  1. How long is the longest run of text that our article and the page share
     character-for-character? That is the hard ceiling on any window size.
  2. For a given window length k, what fraction of an article's k-length
     windows actually appear in the page? That is the hit rate a pairing pass
     would get, and it falls as k grows.
  3. Where along the article do the matches break? If breaks cluster near the
     start, anchoring further in helps; if they are spread evenly, a shorter
     window helps instead.

Run on the 98 articles that have both a benchmark version and a cleaned page,
against BOTH extractions - the cheap regex one (what a first-stage filter sees)
and NewsCleaner's (what the measurement sees) - because they fail differently.

Writes docs/figures/eda/fig7_window_calibration.png.
"""

from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

sys.path.insert(0, str(C.PROJECT / "backend"))

LOOSE = C.PROJECT / "data" / "cnn_downloads" / "cnn" / "downloads"
FIGURES = C.PROJECT / "docs" / "figures" / "eda"
KS = [40, 60, 80, 100, 150, 200, 300, 400, 600]
STRIDE = 20

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"

PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S)
TAG = re.compile(r"<[^>]+>")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def extract_cheap(html: str) -> str:
    return norm(" ".join(TAG.sub(" ", p) for p in PARA.findall(html)))


def longest_run(a: str, b: str) -> int:
    """Longest common contiguous substring, via binary search on length."""
    lo, hi, best = 0, min(len(a), len(b)), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            break
        grams = {a[i:i + mid] for i in range(0, len(a) - mid + 1, 1)}
        if any(g in b for g in grams):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def coverage(a: str, b: str, k: int) -> float:
    """Share of a's k-length windows that appear anywhere in b."""
    spots = range(0, max(1, len(a) - k), STRIDE)
    grams = [a[i:i + k] for i in spots]
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in b) / len(grams)


def first_break(a: str, b: str, k: int = 100) -> float | None:
    """Position (0-1) of the first k-window of a that is NOT in b."""
    spots = list(range(0, max(1, len(a) - k), STRIDE))
    for i in spots:
        if a[i:i + k] not in b:
            return i / max(1, len(a))
    return None


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    from newsqa_rag.ingestion.cleaner import NewsCleaner

    corpus = C.articles("evaluation") + C.articles("distractor")
    by_head = {}
    for row in corpus:
        by_head.setdefault(norm(row["context"])[:120], row)

    cleaner = NewsCleaner()
    pairs = []
    for path in sorted(glob.glob(str(C.PROJECT / "data" / "processed" / "*_clean.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        clean = norm(data["text"])
        row = by_head.get(clean[:120])
        if row is None:
            continue
        html_path = LOOSE / (Path(path).name.replace("_clean.json", "") + ".html")
        if not html_path.exists():
            continue
        cheap = extract_cheap(html_path.read_text(encoding="utf-8", errors="replace"))
        pairs.append({"ours": norm(row["context"]), "clean": clean, "cheap": cheap})
    print(f"pairs available: {len(pairs)}\n")
    if not pairs:
        print("no pairs; nothing to calibrate")
        return

    print("=" * 74)
    print("1. LONGEST SHARED RUN OF TEXT  (the ceiling on any window size)")
    print("=" * 74)
    runs = {}
    for source in ("clean", "cheap"):
        values = [longest_run(p["ours"], p[source]) for p in pairs]
        runs[source] = values
        ordered = sorted(values)
        n = len(ordered)
        print(f"  vs {source:6s} extraction: median {ordered[n//2]:6,d}   "
              f"p25 {ordered[n//4]:6,d}   p10 {ordered[n//10]:6,d}   "
              f"min {ordered[0]:5,d}")
    under = sum(1 for v in runs["cheap"] if v < 200)
    print(f"\n  articles whose longest shared run is under 200 chars "
          f"(the window 10 used): {under} of {len(pairs)}"
          f"  -> {under/len(pairs):.0%} could never match")

    print("\n" + "=" * 74)
    print("2. HIT RATE BY WINDOW LENGTH")
    print("=" * 74)
    print(f"  {'k':>6s}{'clean: any match':>20s}{'cheap: any match':>20s}"
          f"{'cheap: median cover':>22s}")
    curves = {"clean": [], "cheap": []}
    anyhit = {"clean": [], "cheap": []}
    for k in KS:
        row = [f"  {k:>6,d}"]
        for source in ("clean", "cheap"):
            covs = [coverage(p["ours"], p[source], k) for p in pairs]
            curves[source].append(st.fmean(covs))
            hit = sum(1 for c in covs if c > 0) / len(covs)
            anyhit[source].append(hit)
        row.append(f"{anyhit['clean'][-1]:>19.1%}")
        row.append(f"{anyhit['cheap'][-1]:>19.1%}")
        row.append(f"{curves['cheap'][-1]:>21.1%}")
        print("".join(row))

    print("\n" + "=" * 74)
    print("3. WHERE DOES THE FIRST MISMATCH HAPPEN?")
    print("=" * 74)
    breaks = [b for p in pairs if (b := first_break(p["ours"], p["cheap"])) is not None]
    perfect = len(pairs) - len(breaks)
    print(f"  articles matching end to end : {perfect} of {len(pairs)}")
    if breaks:
        ordered = sorted(breaks)
        n = len(ordered)
        print(f"  first break position (0=start, 1=end): median "
              f"{ordered[n//2]:.0%}   p25 {ordered[n//4]:.0%}   p75 {ordered[3*n//4]:.0%}")
        early = sum(1 for b in breaks if b < 0.15)
        print(f"  breaks in the first 15% of the article: {early} of {len(breaks)}"
              f"  ({early/len(breaks):.0%})")

    # ---- figure ---------------------------------------------------------
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(15, 4))

    caps = sorted(runs["cheap"])
    left.hist([min(v, 6000) for v in runs["cheap"]], bins=40, color=BLUE)
    left.axvline(200, color=ORANGE, linewidth=1.8)
    left.annotate("200 = first attempt", xy=(200, 0), xytext=(14, 96),
                  textcoords="offset points", fontsize=8, color=ORANGE,
                  fontweight="bold")
    left.axvline(100, color=AQUA, linewidth=1.8)
    left.annotate("100 = chosen", xy=(100, 0), xytext=(14, 150),
                  textcoords="offset points", fontsize=8, color=AQUA,
                  fontweight="bold")
    style(left, "Longest run of text we share with the page",
          "characters (capped at 6,000)", "articles")
    left.text(0.97, 0.86, f"median {caps[len(caps)//2]:,}\n"
                          f"{under/len(pairs):.0%} below 200",
              transform=left.transAxes, ha="right", fontsize=8, color=MUTED)

    middle.plot(KS, [v * 100 for v in anyhit["clean"]], marker="o", color=BLUE,
                linewidth=2, markersize=6, label="NewsCleaner text")
    middle.plot(KS, [v * 100 for v in anyhit["cheap"]], marker="s", color=ORANGE,
                linewidth=2, markersize=6, label="cheap regex text")
    middle.axvline(200, color=MUTED, linewidth=1, linestyle="--")
    for k, v in zip(KS, anyhit["cheap"]):
        if k in (60, 200, 400):
            middle.annotate(f"{v:.0%}", xy=(k, v * 100), xytext=(4, -12),
                            textcoords="offset points", fontsize=8, color=INK,
                            fontweight="bold")
    middle.set_ylim(0, 105)
    style(middle, "Share of articles a window of length k can find",
          "window length k (characters)", "% of articles matched")
    middle.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower left")

    if breaks:
        right.hist([b * 100 for b in breaks], bins=20, color=BLUE)
        style(right, "Where the first mismatch falls",
              "position in the article (%)", "articles")
        right.text(0.97, 0.86,
                   f"{perfect} of {len(pairs)} match\nend to end",
                   transform=right.transAxes, ha="right", fontsize=8, color=MUTED)

    fig.tight_layout()
    fig.suptitle("Choosing the pairing window from the data, not by guess",
                 fontsize=12, fontweight="bold", color=INK, x=0.055, ha="left", y=1.02)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGURES / "fig7_window_calibration.png", dpi=300,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("\n  wrote docs/figures/eda/fig7_window_calibration.png")

    C.save("11_window_calibration", {
        "pairs": len(pairs),
        "longest_run_median": {s: sorted(v)[len(v) // 2] for s, v in runs.items()},
        "longest_run_p10": {s: sorted(v)[len(v) // 10] for s, v in runs.items()},
        "below_200_cheap": under,
        "k_values": KS,
        "any_match_rate": anyhit,
        "mean_coverage": curves,
        "first_break_median": sorted(breaks)[len(breaks) // 2] if breaks else None,
        "matched_end_to_end": perfect,
    })
    print("Saved -> out/11_window_calibration.json")


if __name__ == "__main__":
    main()
