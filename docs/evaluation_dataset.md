# NewsQA evaluation dataset

The locked corpus contains 200 validation articles and 10,864 train distractor
articles (NewsQA revision `728e529...`, seed `42`). Raw extraction stays
immutable; reviewed answers, clarifications, exclusions, and dedup decisions
are derived artifacts with hashes.

## Build workflow

```bash
# Select locked articles/questions
python scripts/prepare_evaluation_dataset.py stage1 --selection-only

# Chunk the shared corpus and build Chroma/BM25
python scripts/prepare_evaluation_dataset.py build-baseline

# Reuse the completed review when source fields match
python scripts/prepare_evaluation_dataset.py migrate-review

# For a genuinely new sample instead
python scripts/prepare_evaluation_dataset.py init-review --archive-existing
python scripts/prepare_evaluation_dataset.py prepare-review-packets

# Check approval progress, then finalize
python scripts/prepare_evaluation_dataset.py review-status
python scripts/prepare_evaluation_dataset.py finalize
```

Review proposals are packet-complete JSON files applied with:

```bash
python scripts/apply_review_proposals.py --packet PATH_TO_PACKET --proposals PATH_TO_PROPOSALS
```

Codex proposes labels, supported minimal clarifications, answer/evidence fixes,
and rationales. It must not change raw question, answer, or evidence fields.
Human review then marks each proposal `mark_standalone`, `approve`, `edit`,
`exclude`, or `needs_adjudication`; pending/adjudication rows block finalization.

Finalization validates exact evidence offsets, protected source fields,
answer-leak-free clarifications, documented corrections/exclusions, artifact
hashes, and database counts.

## Final artifacts

| File | Use |
| --- | --- |
| `testset_original.jsonl` | Immutable raw extraction |
| `testset_reviewed_original.jsonl` | Primary scored original wording |
| `testset_resolved.jsonl` | Same rows with approved clarification |
| `testset_clarified.jsonl` | Paired clarified subset |
| `excluded_questions.jsonl` | Explicit exclusions and reasons |
| `review_annotations.jsonl` | Complete review provenance |
| `chunks.jsonl`, `bm25.pkl` | Shared retrieval corpus |
| `integrity_report.json`, variant manifest | Counts and hashes |

Required counts: raw rows equal review annotations; reviewed-original equals
resolved; both equal raw minus exclusions. No rechunking or second collection
occurs during finalization.

## Optional semantic deduplication

```bash
python scripts/export_duplicate_question_report.py
python scripts/record_question_dedup_approval.py --reviewer-id ID --approve-all
python scripts/deduplicate_evaluation_dataset.py
```

Only use `--approve-all` after reviewing every proposed cluster. Deduplication
keeps raw/review artifacts intact and applies one approved within-article
partition to both scored variants.

Report reviewed-original as primary, compare same-size resolved rows for the
clarification effect, and always report correction/clarification/exclusion
counts. See [evaluation.md](evaluation.md) for metric contracts.
