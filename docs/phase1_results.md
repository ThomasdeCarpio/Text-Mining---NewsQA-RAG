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
Raw output: `reports/phase1/round{1,2,3}.csv`, winners in
`reports/phase1/winner_lock.jsonl`.

---

## 0. The one rule that governs every comparison

From EDA §7: about **6.5% of questions have a wrong-labelled chunk that
plausibly answers them** — the same news event covered in a second article our
scoring does not mark as correct. The absolute upper bound is 23.1%.

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
| **sparse — BGE-M3 learned sparse** | **0.4243** | **0.8317** | 78 ms |
| sparse — BM25 Okapi, stemmed | 0.3563 | 0.8123 | 59 ms |
| sparse — BM25+ simple | 0.2436 | 0.7206 | 102 ms |
| sparse — BM25 Okapi, simple | 0.2420 | 0.7087 | 110 ms |
| **dense — intfloat/e5-base-v2** | **0.2325** | **0.6661** | 16 ms |
| dense — BAAI/bge-small-en-v1.5 | 0.2285 | 0.6472 | 16 ms |
| dense — BAAI/bge-large-en-v1.5 | 0.2238 | 0.6478 | 28 ms |
| dense — all-MiniLM-L6-v2 | 0.1766 | 0.5129 | 11 ms |

**Winners: BGE-M3 sparse, and e5-base-v2 dense.**

### Why sparse wins, and why that is not an artefact

EDA §6 measured what the question repair does to the vocabulary: rare terms
shared with the correct chunk go **0.33 → 0.89**, and the share of questions with
at least one rare term to anchor on goes **27.7% → 57.7%**. Rare-term anchors are
exactly what word-matching retrieval feeds on, so the EDA warned in advance:
*"comparing sparse against dense using only the resolved set tilts the
comparison toward sparse."*

That warning is why the table above has both columns. **Sparse wins on both** —
by 0.166 nDCG@5 on resolved and by 0.192 on original, where the tilt does not
apply. Both gaps are far outside the 6.5% ambiguity margin. The verdict survives
the bias the data warned us about, which is the only reason we are allowed to
state it.

### Why BGE-M3 over BM25

BM25-stemmed is within 0.019 of BGE-M3 on `resolved` — inside the noise margin,
so on that set alone we could not separate them. On `original` the gap is
**0.068**, outside it. EDA §7 explains the mechanism: **37% of questions contain
no rare term at all**, and a pure lexical scorer has nothing to grab on those.
BGE-M3's learned term weights degrade more gracefully there.

### Why the dense pick was wrong the first time — the noise rule in action

Our first run selected the dense winner on the `original` set, where
bge-small scored 0.2285 and e5-base 0.2325. **A 0.004 gap — deep inside the
6.5% margin.** We were reading noise as a result.

On `resolved`, e5-base leads by 0.019 nDCG@5 and by **2.9 points of Hit@5**
(0.7829 vs 0.7544), and it is faster than bge-large at the same cost as
bge-small. `select_winner` now defaults to `resolved`
(`common/newsqa_rag/evaluation/phase1.py`), and rounds 1–2 were re-run. The
final locked configuration did not change — but the reported dense figure did,
and it is now the one we can defend.

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
| chunks in the corpus | 19,263 |
| median competitors after rare-term filtering | **20** |
| 90th percentile | 49 |
| questions narrowed to <=10 competitors | only **30.7%** |

Rare terms cut the field from 19,263 to about 20 — an enormous reduction, but
**20 is not 1**. The fast first pass gets close and cannot finish. That is a
data-level prediction that a second, more expensive scoring pass will pay off,
made independently of any tournament result.

The tournament then confirms it, and the confirmation lands where the EDA said
it would — at the top of the ranking, which is where "20 competitors" hurts:

| | first stage | + bge-reranker-large | delta |
|---|---|---|---|
| Hit@1 | 0.7260 | **0.8221** | **+0.0961** |
| nDCG@5 | 0.8317 | **0.8976** | +0.0659 |
| Hit@5 | 0.9181 | **0.9573** | +0.0392 |

The +0.096 at rank 1 is roughly 1.5x the ambiguity margin, so it is a real
effect, not measurement slack.

### Why hybrid loses

Fusing dense into sparse *costs* 0.057 nDCG@5. EDA §6 predicted the shape of
this: on `resolved`, sparse has a strong anchor on 57.7% of questions, and
reciprocal-rank fusion dilutes a strong signal with a weaker one. Dense's real
contribution is the 37% of questions with no rare term (EDA §7) — but that
subset is not large enough to pay for the dilution on the rest. **We keep this
as a documented negative result rather than dropping it**; it is the reason the
final system is single-retriever.

### The cost we accepted

bge-reranker-large costs **510 ms p50** against MiniLM's 164 ms — 3x — for
+0.033 nDCG@5. Our tie-break is quality-first, so bge-large wins. If a latency
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
0.047 nDCG@5 — *below* the 6.5% ambiguity margin. EDA said so before we ran it:
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

Recorded in `reports/phase1/winner_lock.jsonl`; this is what Phase 2 builds on.

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
- **6.5%–23% of every score is scoring error**, not system error (EDA §7).
- **Truncation.** 41.6% of corpus articles are cut off (EDA §3), but answers sit
  at the 18th percentile of article length, so truncation almost never reaches
  them — 38 of the 200 evaluation articles lose more than 200 characters.
