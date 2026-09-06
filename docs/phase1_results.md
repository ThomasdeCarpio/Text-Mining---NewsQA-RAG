# Phase 1 — choosing the retrieval configuration

**What this document is for.** It answers three questions in order: *what did we
try*, *what did we pick*, and *why is that the right pick*. The third question is
the one that matters, and we answer it twice — once from the tournament numbers,
and once from the dataset itself (`docs/eda/eda_report.md`). A configuration that
wins the tournament *and* wins for a reason the data predicted in advance is a
defensible choice. One that only wins the tournament is a coin flip we got lucky on.

Every number below is measured on the **281 resolved development questions**
(50 articles), searching a corpus of **22,766 chunks** from 11,064 articles
(dataset `newsqa_200_11064_v2.0.0`, after the truncated article tails were
restored). The EDA was written against the pre-restoration v1.0.0 corpus of
19,263 chunks, so where a figure differs between the two this document uses the
v2.0.0 one and says so.

Raw output: `docs/reports/phase1/round{1,2,3}.csv`, winners in
`docs/reports/phase1/winner_lock.jsonl`.

---

## 0. The one rule that governs every comparison

From EDA §7, re-measured on the v2.0.0 corpus: about **7.0% of questions have a
wrong-labelled chunk that plausibly answers them** — the same news event covered in a second article our
scoring does not mark as correct. The absolute upper bound is 24.5%. (The EDA
reported 6.5% / 23.1% on the smaller v1.0.0 corpus.)

> **Two configurations whose scores differ by less than that margin are not
> meaningfully different.**

We apply this rule everywhere below. It is what stops the report from claiming
precision the data cannot support — and it is also what caught a real mistake in
our own first run (§2).

---

## 1. What we tried — 23 configurations, staged

Testing every combination of 8 retrievers x 3 rerankers x 3 chunk sizes is 72
runs with a cross-encoder in the loop. Instead we ran a **staged tournament**:
settle one axis, carry the winner forward, settle the next.

| Round | Axis under test | Held fixed | Configs |
|---|---|---|---|
| 1 | retriever (4 dense, 4 sparse) | no reranker, 512/64 chunks | 8 |
| 2 | reranker (none / MiniLM / bge-large) x (dense / sparse / hybrid) | round-1 winners | 9 |
| 3 | chunk size (256/32, 512/64, 1024/128) x 2 rerankers | round-2 winner | 6 |

**23 configurations, each run on both question variants** (`original` and
`resolved`) — 46 evaluations. The staging is safe here precisely because the
axes turned out not to interact: the reranker helps every retriever (§3) and
chunk size moves nothing (§4).

---

## 2. Round 1 — which retriever?

nDCG@5, first-stage only, no reranker:

| Retriever | `original` (floor) | `resolved` (realistic) | p50 latency |
|---|---|---|---|
| **sparse — BGE-M3 learned sparse** | **0.4243** | **0.8317** | 77 ms |
| sparse — BM25 Okapi, stemmed | 0.3563 | 0.8123 | 56 ms |
| sparse — BM25+ simple | 0.2436 | 0.7206 | 101 ms |
| sparse — BM25 Okapi, simple | 0.2420 | 0.7087 | 112 ms |
| **dense — intfloat/e5-base-v2** | 0.2263 | **0.6684** | 16 ms |
| dense — BAAI/bge-small-en-v1.5 | 0.2249 | 0.6589 | 16 ms |
| dense — BAAI/bge-large-en-v1.5 | **0.2325** | 0.6536 | 27 ms |
| dense — all-MiniLM-L6-v2 | 0.1788 | 0.5165 | 11 ms |

> **The four dense rows do not reproduce.** Re-running this exact configuration
> on the same data moved every dense model (up to 0.0143) and reshuffled their
> ranking, while every sparse model returned identical figures to four decimal
> places. Do not read a dense ordering out of this table — see §2.3.

**Winners: BGE-M3 sparse, and e5-base-v2 dense.**

### Why sparse wins, and why that is not an artefact

EDA §6 measured what the question repair does to the vocabulary: rare terms
shared with the correct chunk go **0.34 → 0.93**, and the share of questions with
at least one rare term to anchor on goes **28.4% → 59.5%**. Rare-term anchors are
exactly what word-matching retrieval feeds on, so the EDA warned in advance:
*"comparing sparse against dense using only the resolved set tilts the
comparison toward sparse."*

That warning is why the table above has both columns. **Sparse wins on both** —
by 0.166 nDCG@5 on resolved and by 0.192 on original, where the tilt does not
apply. Both gaps are far outside the 7.0% ambiguity margin, and the CI95 intervals
are disjoint ([0.7952, 0.8668] vs [0.6191, 0.7108]). The verdict survives
the bias the data warned us about, which is the only reason we are allowed to
state it.

### Why BGE-M3 over BM25

The paired bootstrap settles this, and it splits by variant:

| Variant | delta nDCG@5 (BGE-M3 − BM25) | CI95 of the delta | Verdict |
|---|---|---|---|
| **`original`** | **+0.0680** | **[+0.0311, +0.1056]** | **significant** (4/4 metrics) |
| `resolved` | +0.0194 | [−0.0125, +0.0508] | not separated |

That split *is* the EDA mechanism, measured. `original` carries a rare-term
anchor on only **28.4%** of questions and **35.2% (470/1,336) carry none at
all**; `resolved` raises the anchored share to **59.5%**. BGE-M3's advantage
appears exactly where lexical matching runs out of handles and disappears
exactly where it does not. A model winning by luck does not produce that
clean a context dependence — so the "learned term weights degrade more
gracefully" claim now has a measured fingerprint, not just an assertion.

### Why the dense pick was wrong the first time — the noise rule in action

Our first run selected the dense winner on the `original` set, where
bge-small scored 0.2285 and e5-base 0.2325. **A 0.004 gap — deep inside the
7.0% margin.** We were reading noise as a result. `select_winner` now defaults
to `resolved` (`common/newsqa_rag/evaluation/phase1.py`) and rounds 1–2 were
re-run.

**But the deeper problem is that no dense ranking here is stable.** The paired
bootstrap cannot separate e5-base from bge-small on any metric or either
variant, and bge-small is slightly *ahead* at Hit@1 (−0.0107 for e5). Then
re-running the identical configuration showed why:

| | nDCG@5 run 1 | run 2 | drift |
|---|---|---|---|
| sparse — all four models | — | — | **0.0000** |
| dense — e5-base-v2 | 0.6661 | 0.6684 | +0.0023 |
| dense — bge-small | 0.6472 | 0.6589 | **+0.0117** |
| dense — bge-large | 0.6478 | 0.6536 | +0.0058 |
| dense — all-MiniLM-L6-v2 | 0.5129 | 0.5165 | +0.0036 |

Run-to-run drift for one model (0.0117) **exceeds the gap between two models**
(0.0094), and on `original` all three top dense positions changed places
between runs. Three unseeded sources explain it, all of them dense-only:
Chroma's HNSW index is approximate and takes only `hnsw:space`, no seed
(`indexing/chroma_store.py:33`); `model.encode()` pins no `batch_size`, so
batching shifts with text length (`indexing/embeddings.py:148`); and nothing
in the repo sets `torch.manual_seed` or `use_deterministic_algorithms`. Sparse
scoring is exact and uses none of them.

**This changes nothing we ship.** The locked configuration is sparse-only, so
no dense component reaches production; the dense arm survives only inside the
round-2 hybrid, which lost. We keep e5-base because the pre-registered
tie-break selects it on `resolved`, and we make no claim that it is better.

---

## 3. Round 2 — is a reranker worth it?

nDCG@5 on `resolved`, round-1 winners carried forward:

| Retriever | no reranker | MiniLM-L6 | **bge-reranker-large** |
|---|---|---|---|
| **sparse (BGE-M3)** | 0.8317 | 0.8642 | **0.8976** |
| hybrid (sparse + dense) | 0.7474 | 0.8164 | 0.8405 |
| dense (e5-base) | 0.6684 | 0.7994 | 0.8172 |

### The reranker was justified before the tournament ran

This is the strongest EDA→decision link in the project. EDA §7 measured, for
each question, how many wrong chunks share at least one of its rare terms — the
field a word-matching retriever still has to choose between:

| | |
|---|---|
| chunks in the corpus | 22,766 |
| median competitors after rare-term filtering | **25** |
| 90th percentile | 61 |
| worst case | 193 |
| questions narrowed to <=10 competitors | only **25.8%** |

Rare terms cut the field from 22,766 to about 25 — an enormous reduction, but
**25 is not 1**. The fast first pass gets close and cannot finish. That is a
data-level prediction that a second, more expensive scoring pass will pay off,
made independently of any tournament result.

The tournament then confirms it, and the confirmation lands where the EDA said
it would — at the top of the ranking, which is where "25 competitors" hurts:

| | first stage | + bge-reranker-large | delta |
|---|---|---|---|
| Hit@1 | 0.7260 | **0.8221** | **+0.0961** |
| nDCG@5 | 0.8317 | **0.8976** | +0.0659 |
| Hit@5 | 0.9181 | **0.9573** | +0.0392 |

The +0.096 at rank 1 is roughly 1.4x the ambiguity margin, and it is one of only
three comparisons in the whole tournament whose CI95 intervals do not overlap
([0.8688, 0.9229] vs [0.7952, 0.8668]). A real effect, not measurement slack.

Restoration made this case stronger, not weaker: on the larger corpus the median
question faces 25 competitors instead of 20, and the share narrowed to 10 or
fewer fell from 30.7% to 25.8%.

### Why hybrid loses

Fusing dense into sparse *costs* 0.057 nDCG@5. EDA §6 predicted the shape of
this: on `resolved`, sparse has a strong anchor on 59.5% of questions, and
reciprocal-rank fusion dilutes a strong signal with a weaker one. Dense's real
contribution is the 35.2% of questions with no rare term (EDA §7) — but that
subset is not large enough to pay for the dilution on the rest. The loss shows up
at every reranker level (0.7474 vs 0.8317 with none; 0.8164 vs 0.8642 with
MiniLM; 0.8405 vs 0.8976 with bge-large), which places it in the fusion step
rather than the reranker.

Stated precisely: the CI95 intervals overlap ([0.8032, 0.8753] vs
[0.8688, 0.9229]), so the honest claim is *"hybrid buys nothing measurable"*,
not *"hybrid is worse"*. **We keep this as a documented negative result rather
than dropping it**; it is the reason the final system is single-retriever.

### The cost we accepted

bge-reranker-large costs **510 ms p50** against MiniLM's 164 ms — 3x — for
+0.033 nDCG@5 — a gap inside the noise margin, with overlapping CI95 intervals.
Our tie-break is quality-first, so bge-large wins, but the evidence does not
separate the two rerankers. If a latency
budget ever forces the issue, MiniLM at 0.8642 is the documented fallback and
still beats no reranker at all.

---

## 4. Round 3 — chunk size

nDCG@5 on `resolved`, sparse + bge-reranker-large:

| Chunk size / overlap | nDCG@5 | Hit@5 |
|---|---|---|
| **512 / 64** | **0.8976** | **0.9573** |
| 1024 / 128 | 0.8862 | 0.9253 |
| 256 / 32 | 0.8507 | 0.9395 |

**Honest reading: this round has no winner worth the name.** The full spread is
0.047 nDCG@5 — *below* the 7.0% ambiguity margin, and all three CI95 intervals
overlap. EDA said so before we ran it:
NewsQA articles are short enough that chunk retrieval is almost the same task
as article retrieval, leaving little headroom for a chunking strategy.

The EDA measured that on the pre-restoration corpus (**1.74 chunks per
article**, max 3). Re-measured on the v2.0.0 corpus Phase 1 actually ran on, the
figure is **2.06 chunks per article** (max 7) — restoration appended 5.35M
characters to 41.6% of articles and grew the chunk count 19,263 → 22,766. The
median article is unchanged at 2 chunks, and **75.9% of articles still produce
at most 2**.

| | v1.0.0 (EDA) | **v2.0.0 (Phase 1)** |
|---|---|---|
| chunks | 19,263 | **22,766** |
| mean chunks/article | 1.74 | **2.06** |
| median tokens/article | 720 | **724** |
| articles with ≥3 chunks | 106 (0.96%) | **2,669 (24.1%)** |

This makes the null result *stronger*, not weaker. Restoration created real
headroom for chunk size to matter — 24% of articles now split three or more ways
instead of 1% — and the three configurations still land within noise of each
other.

We report 512/64 as the pick because it does score highest and it is the middle
setting — but the defensible claim is *"chunk size does not matter on this
corpus,"* not *"512/64 is optimal."* Stating it the other way would be reading
noise, the same mistake §2 describes.

---

## 5. The locked configuration

```
retriever      BGE-M3 learned sparse
top_k          20
reranker       BAAI/bge-reranker-large  ->  top 5
chunking       512 tokens / 64 overlap
```

Recorded in `docs/reports/phase1/winner_lock.jsonl`; this is what Phase 2 builds on.

| Metric (`resolved`, 281 dev questions) | Value |
|---|---|
| Hit@1 | 0.8221 |
| Hit@5 | 0.9573 |
| nDCG@5 | 0.8976 |
| MRR@5 | 0.8797 |
| p50 end-to-end retrieval | 510 ms |

**Reported as a range, per EDA §6:** `original` is the floor (nDCG@5 **0.4824**,
including 34 questions no system can answer), `resolved` is the realistic
ceiling (**0.8976**). Publishing either alone is misleading. Note also that 49
of the resolved questions are duplicates of each other after repair, so the
effective count of distinct questions is lower than the raw count — a footnote
on every score.

### Where these numbers may not hold

- **Tuned on this set.** All of the above is measured on the 50-article
  development split, which is also what we tuned on. The 150 held-out articles
  (871 questions) are run exactly once, at the end; expect the held-out numbers
  to be lower.
- **7.0%–24.5% of every score is scoring error**, not system error (EDA §7).
- **Truncation.** 41.6% of corpus articles are cut off (EDA §3), but answers sit
  at the 18th percentile of article length, so truncation almost never reaches
  them — 38 of the 200 evaluation articles lose more than 200 characters.
