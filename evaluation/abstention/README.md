# NewsQA Abstention Dataset Workflow

This directory stores review inputs and manifests, not a modified copy of the
locked Phase 2 corpus. Every deletion is represented by a reproducible overlay.

## 1. Authored proposals

Create `authored_cases.jsonl` for the two case types that require semantic
authoring. Counterfactual example:

```json
{"case_type":"counterfactual","base_question_id":"QUESTION_ID","question":"Minimally changed but unsupported question?","construction":{"changed_field":"location","original_value":"...","replacement_value":"..."}}
```

External NewsQA example:

```json
{"case_type":"external_unanswerable","base_question_id":"EXTERNAL_QUESTION_ID","source_article_id":"EXTERNAL_ARTICLE_ID","partition":"development","question":"Standalone question from an article absent from the corpus?","construction":{"source":"NewsQA","source_revision":"...","duplicate_check":"pending","corpus_search_review":"pending"}}
```

Do not include an external case until its source article and near duplicates
have been confirmed absent from the locked corpus.

## 2. Prepare and review

Run `scripts/prepare_abstention_dataset.py prepare` as documented in the Phase
3 test plan. The manifest reports exact deficits rather than silently changing
the requested class distribution.

Review `review_queue.jsonl` without changing IDs, source questions, gold fields,
or overlay fields. Set `human_review.decision` to `approved`, record a reviewer,
set `scope_verified` to `true`, and explain why evidence is sufficient or
insufficient. Corpus-level cases also require an independent
`secondary_review` with a different reviewer ID.
For every other case type, select and independently approve at least 20% of
records; finalization enforces the quota per case type.

## 3. Finalize and run

`prepare_abstention_dataset.py finalize` validates exact target counts and
freezes `cases.jsonl`, `corpus_overlays.jsonl`, approval provenance and hashes.
Use `collect_abstention_predictions.py` for structured, resumable generation
and `score_abstention_predictions.py` for abstention precision/recall/F1,
false-answer rate, false-abstention rate and per-case-type results.
