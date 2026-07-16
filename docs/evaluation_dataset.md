# NewsQA Evaluation Dataset

## Purpose

This workflow constructs a locked, reproducible benchmark for evaluating the
news RAG pipeline. It does not silently treat every NewsQA annotation as valid.
Every selected question, answer, and evidence span is reviewed against the full
source article before it enters a scored reviewed benchmark.

The locked corpus contains:

- 200 validation articles and all their questions (1,340 questions for revision
  `728e52920b8e4ffcfaad93fa47556f26a1d82546` and seed `42`).
- All 10,864 eligible train articles used only as retrieval distractors.
- One shared corpus, chunking configuration, Chroma collection, and BM25 index
  for every question variant.

The raw NewsQA extraction remains immutable. Reviewed answers and exclusions
are separate derived artifacts with explicit provenance.

## 1. Select the locked corpus

```bash
python scripts/prepare_evaluation_dataset.py stage1 --selection-only
```

Selection scans the complete validation and train splits twice. It samples
sorted context identities with seed `42`, then rescans the source to collect
every question belonging to each selected article. It writes:

- `staging/corpus/evaluation_articles.jsonl`: 200 articles with all questions.
- `staging/corpus/distractor_articles.jsonl`: 10,864 retrieval-only articles.
- `staging/questions/original_questions.jsonl`: immutable flattened source rows.
- `evaluation/manifests/newsqa_200_11064.selection.json`: revision, seed,
  selection IDs, filtering statistics, and artifact hashes.

The legacy Gemini workflow is available only through the explicit
`--legacy-gemini-triage` opt-in. It is not part of the primary full-review
pipeline and is not required.

## 2. Build the baseline corpus

```bash
python scripts/prepare_evaluation_dataset.py build-baseline
```

This applies the current production chunker to all 11,064 articles, maps source
evidence to relevant chunks, and builds the shared Chroma and BM25 indexes. It
writes the immutable raw benchmark to `final/testset_original.jsonl` and sets
the variant manifest status to `baseline_ready`.

The raw original file is useful for provenance and diagnostic comparisons. The
reviewed-original file produced at finalization is the primary scored benchmark.

## 3. Reuse or initialize the full review

The locked 200-article evaluation sample already has a completed Codex and
human review. After stage 1, migrate it into the expanded corpus:

```bash
python scripts/prepare_evaluation_dataset.py migrate-review
```

Migration succeeds only when the complete evaluation articles and protected
source question fields are identical. The new review manifest records the
source queue, selection, and audit location. Changing retrieval-only
distractors does not require another semantic or human review.

For a genuinely new evaluation sample, initialize a clean review instead.

```bash
python scripts/prepare_evaluation_dataset.py init-review --archive-existing
```

The old `staging/review` and `staging/triage` directories are moved under a
timestamped `staging/legacy/gemini_review_*` directory. They are preserved but
none of their decisions are imported.

Initialization creates:

```text
staging/review/review_queue_readable.json
staging/review/manifest.json
```

The authoritative queue contains all 1,340 questions, including questions that
an earlier model considered standalone. Its default proposer provenance is
`codex-cli` / `sol-5.6`. Initialization does not need an API key and does not
make a model call.

## 4. Prepare Codex review packets

```bash
python scripts/prepare_evaluation_dataset.py prepare-review-packets
```

This is a deterministic preparation step, not an automatic judge. Each packet
contains at most 20 complete source articles and 150 questions. For every
question it includes the source answer, exact evidence offsets, and the five
highest-ranked competing article snippets from the complete 11,064-article
corpus. Files and their hashes are recorded under:

```text
staging/review/packets/review_001.json
staging/review/packets/...
staging/review/packets/manifest.json
```

Run the semantic review in Codex CLI one packet at a time. Codex should read the
packet, update the matching rows in `review_queue_readable.json`, and save a
packet-level proposal audit under `staging/review/audits/`. The proposal audit
should record the packet ID, input packet SHA-256, changed question IDs, model,
timestamp, rationales, and proposed changes. Do not mark human decisions during
the Codex proposal pass. `staging/review/audits/schema.json` defines the required
append-only audit fields.

Use this prompt for a proposal-only packet review:

```text
Review packet review_NNN as a Codex proposal-only batch. Inspect the full
article and every question in the packet, apply the Codex proposal contract in
this guide, write an exact-coverage proposal file under
staging/review/proposals/, apply it to review_queue_readable.json with an
immutable packet audit, and validate the result. Do not make any human-review
decision; leave every human_review.decision as pending.
```

Codex writes the machine-applicable proposal to
`staging/review/proposals/review_NNN.json`. Apply or revalidate a packet with:

```bash
python scripts/apply_review_proposals.py \
  --packet data/evaluation/newsqa_200_11064/staging/review/packets/review_NNN.json \
  --proposals data/evaluation/newsqa_200_11064/staging/review/proposals/review_NNN.json
```

The command requires exact packet coverage, validates evidence offsets and
non-answer supporting quotes, rejects answer-bearing clarifications, preserves
all protected source and human-review fields, updates the queue atomically, and
creates an immutable audit. It refuses to overwrite an existing audit.

## 5. Codex proposal contract

For every question, Codex must inspect the full article and fill:

- `codex_assessment.label`: `standalone`, `non_standalone`, `invalid`, or
  `uncertain`.
- `codex_assessment.issue_codes`: one or more allowed issue codes where needed.
- `codex_assessment.rationale`: a short article-grounded explanation.
- `comparison.candidate_clarified_question`: a minimal answer-preserving repair
  when the original is not standalone or is repairable.
- `codex_assessment.proposed_supporting_quotes`: exact, non-answer article text
  supporting every added detail.
- `codex_assessment.proposal`: set `status` to `proposed` and record `tool`,
  `model`, `batch_id`, and `created_at`.

Allowed quality issues include ambiguity, malformed questions, truncated or
wrong answers, incorrect evidence, yes/no answer mismatches, and facts that are
not present in the article.

Never change `original_question`, `source_expected_answer`, or
`source_evidence_spans`. For a supported answer correction, update
`expected_answer`, `accepted_answers`, `evidence_spans`, and `evidence_text`, set
`answer_modified` to `true`, and explain it in `answer_review_notes`. Reviewed
spans use `[start, end)` offsets and must exactly match the source article.

Clarification may add only identifying article context and must not reveal any
accepted answer. A question is excluded only when it is unanswerable from the
article, has an irreparable semantic mismatch, or has no unique defensible gold
answer. Repair is preferred when the original semantic target can be preserved.

## 6. Human approval

Review each Codex proposal and fill `human_review` in the authoritative queue.
The default reviewer ID for this dataset is `thomas`.

| Decision | Meaning |
| --- | --- |
| `mark_standalone` | Keep the original wording; accept any reviewed answer fields. |
| `approve` | Accept the Codex clarification and proposed supporting quotes. |
| `edit` | Put accepted wording in `final_clarified_question` and exact quotes in `human_review.supporting_quotes`. |
| `exclude` | Remove the row from scored variants; requires an allowed exclusion code and human notes. |
| `needs_adjudication` | Record a disagreement; finalization remains blocked. |
| `pending` | Not reviewed; finalization remains blocked. |

Human approval never overwrites raw source fields. Preserve a batch approval
audit under `staging/review/audits/` with the reviewer ID, timestamp, approved
question IDs, edits, exclusions, and reasons.

Check progress at any time:

```bash
python scripts/prepare_evaluation_dataset.py review-status
```

The status report includes proposal coverage, human decision counts, labels,
issue counts, corrected answers, exclusions, and the final readiness gate.

## 7. Finalize

After all proposals and human decisions are complete:

```bash
python scripts/prepare_evaluation_dataset.py finalize
```

Finalization requires one annotation per raw question and rejects pending
proposals, unresolved human decisions, changed source fields, invalid evidence
offsets, undocumented answer corrections, answer-leaking clarifications,
unsupported added context, invalid exclusions, changed baseline artifacts, and
database count mismatches.

It writes:

- `testset_original.jsonl`: all 1,340 immutable raw NewsQA rows.
- `testset_reviewed_original.jsonl`: primary scored set using original wording,
  reviewed answers/evidence, and no excluded rows.
- `testset_resolved.jsonl`: the same scored rows, using a human-approved
  clarification where one exists.
- `testset_clarified.jsonl`: paired clarified subset only.
- `excluded_questions.jsonl`: raw and reviewed fields plus exclusion reasons.
- `review_annotations.jsonl`: one finalized annotation for every raw question,
  including excluded questions.
- `chunks.jsonl`, `bm25.pkl`, `integrity_report.json`, and the updated variant
  manifest.

Required count relationships are:

```text
raw original = review annotations = 1,340
reviewed original = resolved = raw original - excluded
clarified = approved or edited questions with a clarification
```

No rechunking, embedding, or second database collection occurs during
finalization.

## 8. Build the semantic-deduplicated variant

After finalization, export Codex's proposed clusters for human review:

```bash
.venv/bin/python scripts/export_duplicate_question_report.py
```

After reading every proposed cluster, explicitly record the decisions. The
`--approve-all` command is appropriate only when the reviewer accepts every
cluster in the report:

```bash
.venv/bin/python scripts/record_question_dedup_approval.py \
  --reviewer-id REVIEWER_ID \
  --approve-all
```

Then create the performance-blind, within-article deduplicated variant:

```bash
.venv/bin/python scripts/deduplicate_evaluation_dataset.py
```

Use `--overwrite` only when intentionally rebuilding an existing derived
output. The command validates the finalized artifact hashes and the semantic
decision file before writing:

```text
data/evaluation/newsqa_200_11064/final_deduplicated/
evaluation/manifests/newsqa_200_11064.deduplicated.variant.json
```

The raw 1,340-question extraction and all 1,340 review annotations remain
complete. Only the scored reviewed/resolved variants are reduced from 1,336 to
1,152 unique within-article semantic targets. The 184 removed rows and the full
cluster partition are preserved in `duplicate_questions.jsonl` and
`question_clusters.jsonl`. Corpus, chunks, BM25, Chroma, and distractors are
unchanged, so review work and retrieval infrastructure are reused.

Codex proposals and human approval are separate immutable artifacts:

```text
evaluation/question_dedup/newsqa_200_11064.semantic_clusters.json
evaluation/question_dedup/newsqa_200_11064.human_approval.json
```

The approval references the proposal and resolved testset hashes and contains
one decision for every proposed multi-question cluster. The builder rejects
missing, stale, incomplete, or rejected approval records. Candidate detection
uses resolved wording, then applies the approved partition to both the
reviewed-original and resolved scored variants so their question populations
remain paired. The raw original extraction is not deduplicated.

The complete file-by-file output contract, finalized counts, schemas,
manifests, and benchmark commands are documented in
[`final_evaluation_output.md`](final_evaluation_output.md).
The code and artifact handoff checklist, including archive and reconstruction
instructions, is documented in
[`evaluation_dataset_handoff.md`](evaluation_dataset_handoff.md).

## Reporting

Report `testset_reviewed_original.jsonl` as the primary result. Compare it with
the same-size `testset_resolved.jsonl` to measure the effect of clarification,
and use `testset_clarified.jsonl` for paired ambiguity analysis. Report raw,
corrected, clarified, and excluded counts from the integrity manifest.

Do not combine original and clarified duplicates into one primary aggregate;
that would give ambiguous source questions extra weight. Exclusions must be
reported rather than hidden, together with their reason distribution.
