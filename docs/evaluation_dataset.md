# NewsQA Evaluation Dataset

## Purpose

This workflow constructs a locked, reproducible RAG benchmark rather than a new
version of NewsQA. Original question wording is always the primary benchmark.
Human-approved clarifications form paired and full-size secondary benchmarks
for measuring the effect of resolving under-specified queries. Every question
variant queries the same article collection.

The corpus contains 1,000 articles:

- 200 uniformly sampled validation articles and all of their questions (1,340
  questions for the pinned revision and seed `42`).
- 800 uniformly sampled train articles used only as retrieval distractors.

The source revision, seed, selected context hashes, and artifact hashes are
recorded in committed manifests. Source text, generated JSONL, and database
files remain under the Git-ignored `data/` directory.

## Stage 1: Select the corpus

Select and stage the locked corpus without making model calls:

```bash
python scripts/prepare_evaluation_dataset.py stage1 --selection-only
```

Selection scans each source split completely twice. It operates over sorted
SHA-256 context identities, so the same revision and seed produce the same
sample even if source rows are returned in another order. The second scan
collects every question and verifies the question count observed in the first
scan.

The combined `stage1` command remains available when selection and triage
should run sequentially:

```bash
python scripts/prepare_evaluation_dataset.py stage1
```

## Stage 2: Build the baseline

Build the production chunks, original test set, Chroma collection, and BM25
index immediately after selection:

```bash
python scripts/prepare_evaluation_dataset.py build-baseline
```

The generated variant manifest has status `baseline_ready`. At this point,
`testset_original.jsonl` can be benchmarked even if triage has not started or
human review is incomplete.

## Stage 3: Triage and human review

Set `GEMINI_API_KEY` in `.env`, then run:

```bash
python scripts/prepare_evaluation_dataset.py triage
python scripts/format_review_queue.py
```

Gemini receives one request per article in the current validation data (a
25-question safety limit can split larger articles). Each question includes its
expected answer so Gemini can preserve semantic intent and answer type, but the
prompt prohibits copying or paraphrasing that answer into the clarification.
All answer evidence is redacted from every source window, and BM25 candidate
articles expose competing interpretations. Every successful article response
is checkpointed, so interrupted runs resume without repeating valid calls.

## Human review

Edit the authoritative hierarchical review file directly:

```text
data/evaluation/newsqa_200_1000/staging/review/review_queue_readable.json
```

`review_queue.jsonl` remains the immutable Gemini snapshot. The CSV is retained
only as a legacy/export format and is no longer consumed by finalization. The
formatter refuses to overwrite an existing hierarchical review file unless
`--overwrite` is explicitly passed.

Allowed decisions are:

| Decision | Required action |
| --- | --- |
| `approve` | Accept the LLM clarification and its supporting source quotes. |
| `edit` | Fill `final_clarified_question` and `review_supporting_quotes`. |
| `mark_standalone` | Override the suspected ambiguity; no clarification is created. |
| `needs_adjudication` | Keep the build blocked until the disagreement is resolved. |
| `pending` | Unreviewed; keeps the build blocked. |

Every resolved row requires `human_review.reviewer_id`. For an `edit`, put the
proposed replacement in `comparison.reviewed_candidate_clarified_question`,
copy the accepted wording to `comparison.final_clarified_question`, and put
exact source quotes in `human_review.supporting_quotes`. The original Gemini
proposal remains in `comparison.candidate_clarified_question` for provenance.

Answer corrections are made under `answer_and_evidence`. Preserve all
`source_*` fields, then update `expected_answer`, `accepted_answers`,
`evidence_text`, and `evidence_spans`; set `answer_modified` to `true` and
record the reason in `answer_review_notes`. Corrected spans use `[start, end)`
character offsets and are validated against the source article.

Rows with a non-empty `validation_warnings` column contain an LLM proposal that
used answer evidence, leaked the answer, or cited unsupported source text. They
cannot use `approve`; the reviewer must choose `edit` with new supporting quotes
or `mark_standalone`.

Check the gate with:

```bash
python scripts/prepare_evaluation_dataset.py review-status
```

## Stage 4: Finalize reviewed variants

After review is complete:

```bash
python scripts/prepare_evaluation_dataset.py finalize
```

Finalization rejects missing rows, unresolved decisions, answer leakage,
unsupported clarification context, changed baseline artifacts, or collection
count mismatches. It reuses the baseline chunks, Chroma collection, and BM25
index; it does not rechunk, re-embed, or create a second collection.

Generated files include:

- `testset_original.jsonl`: all original evaluation questions.
- `testset_clarified.jsonl`: paired, human-approved clarification variants.
- `testset_resolved.jsonl`: one row per original question, using approved
  clarified wording where available and original wording otherwise.
- `review_annotations.jsonl`: finalized labels and clarification provenance.
- `chunks.jsonl` and `bm25.pkl`: sparse/hybrid retrieval artifacts.
- `integrity_report.json`: final counts and gate results.

Run a manifest-verified benchmark using paths and the collection recorded by
the generated variant manifest:

```bash
python scripts/run_benchmark.py \
  --retriever hybrid \
  --testset data/evaluation/newsqa_200_1000/final/testset_original.jsonl \
  --variant-manifest evaluation/manifests/newsqa_200_1000.variant.json \
  --report-dir reports/newsqa_200_hybrid
```

The same command accepts `testset_resolved.jsonl` or
`testset_clarified.jsonl`; manifest preflight verifies all three against the
same collection.

## Interpretation and limitations

NewsQA is document-conditioned QA. A vague original question can be valid with
its source article supplied but ambiguous across 1,000 articles. Report original
questions as the primary result. Compare the same-size original and resolved
sets, then use the clarified-only set for paired diagnostics. Do not combine
originals and clarified duplicates into a primary aggregate because that would
give ambiguous source questions extra weight. LLM `standalone` labels are not
human gold labels unless separately audited.

Uniform sampling improves representativeness but does not establish demographic
or topical fairness because this NewsQA representation has no dependable topic
or demographic fields. Distractors may also contain alternate valid evidence;
source-article retrieval remains the primary relevance definition and suspected
multiple matches are reported as an ambiguity reason.
