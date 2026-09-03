"""Render the EDA figures used by docs/eda_report.md.

Reads the cached results in out/*.json, so it is cheap and cannot disagree with
the report. Run the analysis scripts first if out/ is empty.

Writes 300 DPI PNGs to docs/figures/eda/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

FIGURES = C.PROJECT / "docs" / "figures" / "eda"

# Categorical slots 1-3 of the validated default palette. Assigned by entity
# (original vs resolved), never by rank, and never cycled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SURFACE = "#fcfcfb"


def cached(name: str) -> dict:
    return json.loads((C.OUT / f"{name}.json").read_text(encoding="utf-8"))


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Recessive axes and grid; the data carries the ink."""
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote docs/figures/eda/{name}")


# --------------------------------------------------------------------------
def fig_truncation() -> None:
    """The three things that establish truncation, in order of strength."""
    articles = C.articles("evaluation") + C.articles("distractor")
    chars = [len(a["context"]) for a in articles]
    words = [len(a["context"].split()) for a in articles]
    truncation = cached("01d_truncation")
    n = len(articles)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))

    a.hist(chars, bins=70, color=BLUE)
    a.axvline(max(chars), color=MUTED, linewidth=1.1, linestyle="--")
    a.annotate(f"longest\n{max(chars):,} chars", xy=(max(chars), 0),
               xytext=(-6, 52), textcoords="offset points", ha="right",
               fontsize=8, color=MUTED)
    style(a, "Article length in characters", "characters", "articles")

    # The character view hides the cap because articles vary in word length.
    # In words the ceiling is unmistakable.
    counts = b.hist(words, bins=70, color=BLUE)[0]
    capped = sum(1 for w in words if 640 <= w <= 680)
    b.axvspan(640, 680, color=ORANGE, alpha=0.16)
    b.annotate(f"{capped/n:.0%} of the corpus\nends at 640-680 words",
               xy=(660, max(counts) * 0.92), xytext=(-150, -6),
               textcoords="offset points", fontsize=8, color=INK,
               fontweight="bold")
    over = sum(1 for w in words if w > 700)
    # Leave room on the right so the cliff annotation does not sit on the bars.
    b.set_xlim(0, 980)
    b.annotate(f"then a cliff -\nonly {over} articles\nexceed 700 words",
               xy=(715, max(counts) * 0.06), xytext=(790, max(counts) * 0.22),
               fontsize=8, color=MUTED,
               arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))
    style(b, "Article length in WORDS - the cap", "words", "articles")

    order = ["shortest 25%", "middle", "longest 25%", "within 200 of max"]
    values = [truncation["ends_cleanly"][k] * 100 for k in order]
    bars = c.bar(range(len(order)), values, color=BLUE, width=0.62)
    bars[-1].set_color(ORANGE)
    bars[-2].set_color(ORANGE)
    for bar, value in zip(bars, values):
        c.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%",
               ha="center", fontsize=9, color=INK, fontweight="bold")
    c.set_xticks(range(len(order)))
    c.set_xticklabels(["shortest\n25%", "middle\n50%", "longest\n25%",
                       "within 200\nof the max"], fontsize=8)
    c.set_ylim(0, 100)
    style(c, "Articles ending on sentence punctuation", "", "% of articles")
    c.annotate("long articles stop\nmid-sentence", xy=(2.6, 22),
               xytext=(1.3, 58), fontsize=8, color=MUTED,
               arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))

    fig.suptitle("Truncation: a word cap, and articles that stop mid-sentence",
                 fontsize=12, fontweight="bold", color=INK, x=0.055, ha="left",
                 y=1.02)
    fig.tight_layout()
    save(fig, "fig1_truncation.png")


def fig_evidence_position() -> None:
    """Why truncation costs less than the 41% headline suggests."""
    deciles = cached("01h_versions")["answer_position_deciles"]
    total = sum(deciles)
    shares = [d / total * 100 for d in deciles]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(range(10), shares, color=BLUE, width=0.72)
    for index in (8, 9):
        bars[index].set_color(ORANGE)
    for bar, share in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, share + 0.7, f"{share:.0f}%",
                ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{i*10}-{i*10+10}%" for i in range(10)], fontsize=8)
    ax.set_ylim(0, max(shares) * 1.18)
    style(ax, "Where the answer sits inside its article",
          "position in the article", "% of questions")
    ax.annotate(f"truncation removes the tail\n{(shares[8]+shares[9]):.0f}% of answers live here",
                xy=(8.5, shares[8] + 1), xytext=(5.6, max(shares) * 0.62),
                fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))
    fig.tight_layout()
    save(fig, "fig2_evidence_position.png")


def fig_question_repair() -> None:
    """What was wrong with the questions, and what repairing them bought."""
    codes = cached("05_reason_codes")["reason_codes"]
    overlap = cached("03_lexical_overlap")
    before = cached("06_near_duplicates_original")
    after = cached("06_near_duplicates_resolved")

    fig = plt.figure(figsize=(13, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.85], wspace=0.42)

    top = sorted(codes.items(), key=lambda kv: kv[1])[-9:]
    ax = fig.add_subplot(grid[0])
    names = [k.replace("_", " ") for k, _ in top]
    values = [v for _, v in top]
    bars = ax.barh(names, values, color=BLUE, height=0.66)
    bars[-1].set_color(ORANGE)
    for bar, value in zip(bars, values):
        ax.text(value + 8, bar.get_y() + bar.get_height() / 2,
                f"{value:,}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(values) * 1.18)
    style(ax, "Why a question could not stand alone", "questions (n=1,340)")
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(grid[1])
    labels = ["rare terms shared\nwith the gold chunk",
              "questions with\n>=1 rare locator"]
    original = [overlap["original"]["rare_gold_mean"],
                overlap["original"]["at_least_one_rare"]]
    resolved = [overlap["resolved"]["rare_gold_mean"],
                overlap["resolved"]["at_least_one_rare"]]
    # Two measures on different scales never share an axis: plot each panel
    # half against its own normalised height and label the real value.
    positions = [0, 1]
    width = 0.36
    for offset, series, colour, name in ((-width / 2, original, BLUE, "original"),
                                         (width / 2, resolved, ORANGE, "resolved")):
        heights = [series[0] / 1.0, series[1]]
        bars = ax.bar([p + offset for p in positions], heights, width=width,
                      color=colour, label=name)
        for bar, raw, index in zip(bars, series, range(2)):
            text = f"{raw:.2f}" if index == 0 else f"{raw:.0%}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    text, ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    style(ax, "What resolution added", "", "")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
    ax.grid(visible=False)

    ax = fig.add_subplot(grid[2])
    values = [before["cross_article_conflicts"], after["cross_article_conflicts"]]
    bars = ax.bar(["original", "resolved"], values, color=[BLUE, ORANGE], width=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, str(value),
                ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3)
    style(ax, "Unscoreable question pairs", "", "near-duplicate pairs across\ndifferent gold articles")
    ax.grid(axis="x", visible=False)

    fig.suptitle("The question set: the defect, the repair, and what the repair fixed",
                 fontsize=12, fontweight="bold", color=INK, x=0.055, ha="left",
                 y=1.02)
    fig.tight_layout()
    save(fig, "fig3_question_repair.png")


def fig_retrieval_difficulty() -> None:
    """How far first-stage retrieval gets, and the ground truth's error floor."""
    collision = cached("04_distractor_collision")
    counts = collision.get("competitor_counts")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4))

    if counts:
        capped = [min(c, 100) for c in counts]
        left.hist(capped, bins=50, color=BLUE)
        median = collision["competitors_median"]
        left.axvline(median, color=ORANGE, linewidth=1.8)
        left.annotate(f"median {median}", xy=(median, 0), xytext=(10, 62),
                      textcoords="offset points", fontsize=9, color=ORANGE,
                      fontweight="bold")
        left.set_xlim(0, 100)
        style(left, "Non-gold chunks sharing the question's rare terms",
              "competitors (capped at 100)", "questions")
        left.text(0.98, 0.86,
                  f"19,263 chunks narrowed to ~{median}\nonly "
                  f"{collision['narrowed_to_10']:.0%} reach 10 or fewer",
                  transform=left.transAxes, ha="right", fontsize=8, color=MUTED)

    strength = {int(k): v for k, v in (collision.get("match_strength") or {}).items()}
    checked = collision["checked"]
    none = checked - collision["distractor_has_answer"]
    labels = ["no answer-bearing\ndistractor", "1 shared rare term\n(coincidental)",
              "2 shared\n(same event)", "3+ shared\n(same event)"]
    values = [none, strength.get(1, 0), strength.get(2, 0), strength.get(3, 0)]
    colours = [GRID, MUTED, ORANGE, ORANGE]
    bars = right.bar(range(4), values, color=colours, width=0.62)
    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2, value + 6,
                   f"{value}\n{value/checked:.1%}", ha="center", fontsize=8,
                   color=INK)
    right.set_xticks(range(4))
    right.set_xticklabels(labels, fontsize=8)
    right.set_ylim(0, max(values) * 1.25)
    style(right, "Can a non-gold distractor answer the question?", "",
          f"questions (n={checked:,})")
    strong = collision.get("strong_candidates", 0)
    right.text(0.98, 0.72,
               f"strong candidates {strong/checked:.1%}\nupper bound "
               f"{collision['distractor_has_answer']/checked:.1%}",
               transform=right.transAxes, ha="right", fontsize=8, color=MUTED)

    fig.suptitle("Retrieval difficulty, and the benchmark's false-negative floor",
                 fontsize=12, fontweight="bold", color=INK, x=0.065, ha="left",
                 y=1.02)
    fig.tight_layout()
    save(fig, "fig4_retrieval_difficulty.png")


def fig_restoration() -> None:
    """How far the archived pages reach, from the corrected pairing."""
    gap = cached("08_truncation_gap")
    stages = [
        ("archived pages scanned", gap["pages_scanned"]),
        ("benchmark articles", gap["corpus"]),
        ("paired with a cleaned page", gap["matched"]),
        ("text missing off the end", gap["truncated"]),
        ("real content missing (>40 chars)", gap["real_content"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [n for n, _ in stages][::-1]
    values = [v for _, v in stages][::-1]
    bars = ax.barh(names, values, color=BLUE, height=0.62)
    bars[0].set_color(ORANGE)
    top = max(values)
    for bar, value in zip(bars, values):
        ax.text(value + top * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{value:,}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, top * 1.12)
    style(ax, "Restoration reach - no crawling required", "articles / pages", "")
    ax.grid(axis="y", visible=False)
    ax.text(0.97, 0.12,
            "extraction: the project's own NewsCleaner\n"
            "(newspaper3k), not a hand-rolled parser",
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig5_restoration.png")


def fig_truncation_gap() -> None:
    """How much text is missing, across every article we could pair up."""
    gap = cached("08_truncation_gap")

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(15, 4))

    # What happened to each paired article. Only one bucket is truncation.
    buckets = [
        ("intact\n(page no longer\nthan ours)", gap["intact"], GRID),
        ("TRUNCATED\n(ours is an exact\nprefix)", gap["truncated"], ORANGE),
        ("diverged\n(differs mid-text,\nnot countable)", gap["diverged"], MUTED),
    ]
    bars = left.bar(range(3), [v for _, v, _ in buckets],
                    color=[c for _, _, c in buckets], width=0.6)
    for bar, (_, value, _) in zip(bars, buckets):
        left.text(bar.get_x() + bar.get_width() / 2, value + gap["matched"] * 0.015,
                  f"{value:,}\n{value/gap['matched']:.1%}", ha="center",
                  fontsize=8, color=INK, fontweight="bold")
    left.set_xticks(range(3))
    left.set_xticklabels([n for n, _, _ in buckets], fontsize=8)
    left.set_ylim(0, max(v for _, v, _ in buckets) * 1.28)
    style(left, "Every article paired with its original page", "",
          f"articles (n={gap['matched']:,} matched)")

    # The distribution, not the average - the average hides the shape.
    gaps = gap["gaps_real"]
    capped = [min(g, 3000) for g in gaps]
    middle.hist(capped, bins=60, color=BLUE)
    median = gap["real_gap_chars"]["median"]
    middle.axvline(median, color=ORANGE, linewidth=1.8)
    middle.annotate(f"median {median:,}", xy=(median, 0), xytext=(8, 92),
                    textcoords="offset points", fontsize=9, color=ORANGE,
                    fontweight="bold")
    style(middle, "Characters missing (real content only)",
          "characters (capped at 3,000)", "articles")
    middle.text(0.97, 0.82,
                f"mean {gap['real_gap_chars']['mean']:,.0f}\n"
                f"p90 {gap['real_gap_chars']['p90']:,}\n"
                f"max {gap['real_gap_chars']['max']:,}\n\n"
                f"{gap['furniture_only']:,} furniture-only\ngaps excluded",
                transform=middle.transAxes, ha="right", fontsize=8, color=MUTED)

    # Characters are hard to feel. Share of the article is not.
    # gaps_truncated and gaps_percent are parallel, so the same >40 filter
    # that defines "real content" applies to both.
    percent = [p for g, p in zip(gap["gaps_truncated"], gap["gaps_percent"])
               if g > 40]
    right.hist(percent, bins=50, color=BLUE)
    pmedian = gap["real_gap_percent"]["median"]
    right.axvline(pmedian, color=ORANGE, linewidth=1.8)
    right.annotate(f"median {pmedian}%", xy=(pmedian, 0), xytext=(8, 92),
                   textcoords="offset points", fontsize=9, color=ORANGE,
                   fontweight="bold")
    style(right, "Share of the article that is missing",
          "% of the original article", "articles")

    fig.suptitle("How much text is actually missing, measured against the original pages",
                 fontsize=12, fontweight="bold", color=INK, x=0.055, ha="left",
                 y=1.02)
    fig.tight_layout()
    save(fig, "fig6_truncation_gap.png")


def main() -> None:
    print("Rendering EDA figures from cached results...")
    fig_truncation()
    fig_evidence_position()
    fig_question_repair()
    fig_retrieval_difficulty()
    fig_restoration()
    fig_truncation_gap()
    print(f"\nDone -> {FIGURES.relative_to(C.PROJECT)}")


if __name__ == "__main__":
    main()
