# Roadmap — what is left to build

Everything still open between here and the final presentation: the Phase 1
loose ends, the Phase 2 generation run, the deck, and the risks nobody has
logged yet.

Ordering inside each section is the order to do them in. Effort tags mean:
**no GPU** runs on the laptop, **T4** needs a Kaggle session.

Same content as the tracked checklist at
<https://claude.ai/code/artifact/7eda5030-9750-43c5-974e-9f79b41d3cf5> — edit
both together or the two will drift.

---

## Do these first — they gate everything else

- [ ] **G-1 · Confirm whether the Gemini `160k` limit counts requests or tokens.**
      If requests, 281 generation calls fit easily. If tokens, a full run needs
      roughly **790k input tokens** (281 questions × 5 contexts × ~512) and will
      not fit — over by about 5×. Every Phase 2 cost estimate hangs on this one
      word. *BLOCKER · 5 min*
- [ ] **G-2 · Run `notebooks/public/14_export_locked_index_kaggle.ipynb`.**
      Produces chunks, the BGE-M3 sparse index and the Chroma collection for the
      v2.0.0 corpus. Nothing in Phase 2 — and no live demo — can run until this
      bundle exists. *T4 · 25–45 min*
- [ ] **G-3 · Rotate both API keys when the project wraps.**
      The Gemini and Fireworks keys were pasted into a chat transcript. They live
      in `.env` and are gitignored, so they are not in the repo — but a
      transcript is not a secret store. *SECURITY · 2 min*

---

## Phase 1 — close it out

> **Good news that shrinks this list.** `scripts/generate_retrieval_figures.py`
> already draws all four charts the test plan demands — retriever comparison,
> reranker dumbbell, Pareto frontier, latency breakdown — and already saves at
> `dpi=300`. It is not missing; it is **orphaned**. It reads
> `results/retrieval/retrieval_ablation_summary_table.csv`, a file that does not
> exist, in a schema (`Top_K`, `Top_N`) the round CSVs do not use. So the job is
> an adapter, not five new charts.

- [ ] **P1-1 · Add `Recall@5` to the notebook 13 leaderboard.**
      The test plan §6 names six required columns — MRR@5, NDCG@5, Hit@1, Hit@5,
      **Recall@5**, P50. Recall is the one not surfaced. It is already in the
      CSVs; it just needs adding to `metrics()`. *no GPU · ~15 min*
- [ ] **P1-2 · Point the existing figure script at the round CSVs.**
      Write the adapter from `reports/phase1/round{1,2,3}.csv` into the schema
      `generate_retrieval_figures.py` expects, then run it for the four 300-DPI
      figures. Also re-add `/results/` to `.gitignore` — the refactor dropped
      that line. *no GPU · ~1 h*
- [ ] **P1-3 · Surface the round-3 no-op arm from `retrieval_initial.*`.**
      The plan called for each chunk size × {no-op, cross-encoder}; the run did
      each size × two cross-encoders instead. The no-op numbers are **not lost** —
      the pre-rerank columns hold them, so this is a reporting fix, not a re-run.
      *no GPU · ~30 min*
- [ ] **P1-4 · Run `notebooks/public/12_phase_1_chunking_strategy_kaggle.ipynb`.**
      Hierarchical chunking and the @3/@5/@7 cut-offs. The last experiment that
      could displace the 512/64 recursive lock, and the last input the chunking
      decision is waiting on. *T4 · after G-2*
- [ ] **P1-5 · Run the held-out validation on the 150 articles — once.**
      Every Phase 1 number is a tuning-set estimate: 23 configurations were
      compared on the same 281 questions and the best kept, so the winner's score
      is biased upward. This is the run that makes it honest. It must come after
      every choice is frozen. *T4 · LAST*
- [ ] **P1-6 · Fix stale paths in notebooks 02, 02_ran, 03, 05, 06.**
      They still reference `backend/newsqa_rag/…`, `docs/evaluation.md` and
      `docs/eda_report.md`. Prose only — nothing executes — but a reader
      following them lands nowhere. *no GPU · ~20 min*

---

## Phase 2 — generation

The order is load-bearing. The baseline has to land before any tuning, or you
cannot tell which knob moved the number — and the held-out run comes last,
after everything is frozen.

Full detail: [`Detailed Test Plans/phase_2_generation_tuning_plan.md`](Detailed%20Test%20Plans/phase_2_generation_tuning_plan.md).

- [ ] **P2-1 · Record `EXPECTED_CHUNKS` and its SHA back into notebook 14.**
      The first run prints both. Pasting them back locks every later rebuild to a
      byte-exact result. Also replaces the stale **19,263** chunk count still
      cited in the phase 2 execution guide, which came from the v1.0.0 corpus.
      *5 min*
- [ ] **P2-2 · Unpack the bundle into `data/locked_index/` and confirm the app picks it up.**
      `/retrieval/algorithms` should flip `locked` to available, and chat should
      switch off the Chroma route. This is also what makes the live demo
      possible. *10 min*
- [ ] **P2-3 · Smoke run: 5 questions, whole pipeline, judge included.**
      Seed 42, no mocked responses, no skipped judge. This is where a broken GLM
      route or an empty-content judge shows up for free instead of mid-run.
- [ ] **P2-4 · Baseline 2B — 281 questions on the current prompt.**
      The anchor. Without it no later tuning result means anything, because there
      is nothing to have improved on. *after G-1*
- [ ] **P2-5 · Score deterministically, split answerable vs evidence-missing.**
      Hit@5 is 0.9573, so about **12 of 281 questions** have no gold evidence in
      the five contexts. For those, refusing is the correct answer. Blending them
      into one F1 rewards a model for guessing.
- [ ] **P2-6 · RAGAS pilot, then the full judge run on GLM 5.3 Flash.**
      Pilot first to prove the judge pipeline, then resume across the rest. Keep
      `max_tokens >= 512` — this model spends most of its budget on reasoning
      before it writes anything visible.
- [ ] **P2-7 · Tune the prompt.**
      The largest lever and the cheapest to pull, and it has never been touched.
      Variants worth trying: forbid inference beyond context, permit "not enough
      information", constrain answer length, tighten the citation format,
      few-shot. *highest value*
- [ ] **P2-8 · Sweep contexts shown to the LLM — `top_n` ∈ {3, 5, 7}.**
      Coverage rises (Hit@3 0.932 → Hit@5 0.957) but extra context dilutes
      attention. Cannot be predicted from retrieval metrics; has to be measured
      end to end. If G-1 comes back as tokens, `top_n=3` is also the budget
      escape hatch.
- [ ] **P2-9 · Try hierarchical retrieve-child / feed-parent.**
      Hierarchical chunking's real payoff is here, not in retrieval: match on the
      small child for precision, hand the large parent to the generator so the
      answer is not cut in half. `parent_id` is already in the chunk metadata.
      *after P1-4*
- [ ] **P2-10 · Add query rewriting — the biggest prize on the board.**
      Original vs resolved questions differ by **0.4164 MRR@5** averaged over
      eight retrievers — larger than any model-to-model gap measured in Phase 1.
      Real users type the anchored form, and the pipeline has no rewriting step
      at all. Measure against raw original (floor) and hand-resolved (ceiling).
      *highest ceiling*
- [ ] **P2-11 · If budget remains: sweep `top_k`, revisit the reranker.**
      `top_k=20` was inherited from round 2 and never swept on its own; try 30
      and 50. And check whether MiniLM's 324 ms saving costs any answer F1 once
      the generator only reads five contexts. *optional*
- [ ] **P2-12 · Freeze the configuration.**
      Explicit step, not a formality. P1-5 cannot run until this is done, and the
      held-out set stops being held out the moment one decision is made on it.
      *GATE*

---

## Presentation

A deck already exists — 15 slides, fully offline. The refactor moved it to
`docs/archive/`, and its test has been quietly skipping ever since.

- [ ] **PR-1 · Move the deck out of `docs/archive/slides/` and un-skip its test.**
      `tests/test_slides.py` looks for `docs/slides/index.html`, does not find it,
      and calls `skipTest`. That is the 1 skipped in every test run. The deck —
      15 slides with speaker notes, no external URLs — is intact but unguarded.
      *15 min*
- [ ] **PR-2 · Bring the deck's content up to date.**
      It predates the corpus restoration to v2.0.0, the Phase 1 lock (BGE-M3
      sparse + bge-reranker-large), and the tuning/held-out framing. Any number
      on a slide should be traceable to `reports/phase1/`. *after P1-2*
- [ ] **PR-3 · Check the deck answers the lecturer's brief point by point.**
      EDA with original vs resolved; the 50 seeded papers ≈281 questions; how
      questions and chunks are embedded; metrics at @3, @5, @7; strategies tested
      including hierarchical chunking. Walk the brief as a checklist. *30 min*
- [ ] **PR-4 · Rehearse the live demo on the locked config.**
      The app now serves BGE-M3 sparse + bge-reranker-large by default. Warm the
      models before presenting — the first query loads both sets of weights, and
      on CPU the reranker is the slow half. Have a recorded fallback.
      *after P2-2*
- [ ] **PR-5 · Prepare the answer to "how do you know it generalises?"**
      The strongest question a lecturer can ask, and you have the strongest
      possible answer: an untouched 150-article held-out set and an explicit
      account of why tuning-set numbers are optimistic. Do not let this be the
      thing you improvise. *after P1-5*

---

## Loose ends worth flagging

Things noticed in passing that are not on anyone's plan.

- [ ] **R-1 · Repoint or retire the dense fallback collection.**
      `retrieval_service._DEFAULT_COLLECTION` is still `"newsqa_cnn"` — a
      323-chunk collection from early development. If anyone picks `dense` in the
      UI they silently query a nearly empty index. Point it at the exported
      Chroma or drop the route. *15 min*
- [ ] **R-2 · Tag `v2.0.0` on the Hugging Face dataset.**
      No tag exists, so every notebook pins the raw commit SHA `b81c8db…`. It
      works, but a tag is what the docs promise and what a reader will look for.
      *5 min*
- [ ] **R-3 · Decide on the trailing "All About &lt;topic&gt;" tags.**
      They appear in 3,047 of the restored article tails — 66.8%, against 1.1% in
      the published corpus. Still undecided. Fixing means regenerating and
      re-uploading the dataset, so decide before anything else pins to it.
      *decision*
- [ ] **R-4 · Delete the redundant `newsqa_200_11064_v2.0.0.zip` from the repo.**
      The same bundle is already published on Hugging Face and verified by
      checksum. Keeping a large binary in git buys nothing. *2 min*
- [ ] **R-5 · Move `_ragas_shim()` inside `_ragas_judge()`.**
      Calling `_ragas_judge()` directly raises `ModuleNotFoundError` on the
      ragas/vertexai import unless the shim ran first. Real callers go through
      `evaluate_ragas_rows`, so nothing is broken today — it is a trap for
      whoever calls it next. *5 min*
- [ ] **R-6 · Caveat any dense number that reaches the report.**
      Round 1 promoted `bge-small` on the original questions, winning by 0.0018
      MRR@5 — noise. On resolved questions `e5-base` leads instead. So every
      dense figure in rounds 2 and 3 is a lower bound on dense, not its ceiling.
      *when writing*
