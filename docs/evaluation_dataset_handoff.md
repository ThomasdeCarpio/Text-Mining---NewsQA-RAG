# NewsQA Evaluation Dataset Handoff

## Purpose

This document lists the code, configuration, generated artifacts, and sharing
steps for the finalized `newsqa_200_11064` evaluation database. The benchmark
contains 200 evaluation articles, all 1,340 of their questions, and 10,864
retrieval-only distractor articles.

Use these companion documents for details:

- [`evaluation_dataset.md`](evaluation_dataset.md): selection, review, and
  finalization workflow.
- [`final_evaluation_output.md`](final_evaluation_output.md): final file schemas,
  counts, and benchmark usage.
- [`evaluation.md`](evaluation.md): retrieval and answer evaluation metrics.
- [`indexing_guide.md`](indexing_guide.md): indexing architecture.
- [`database.md`](database.md): Chroma storage behavior.

## Required project files

### Entry-point scripts

| File | Role |
| --- | --- |
| `scripts/prepare_evaluation_dataset.py` | Canonical workflow entry point. It provides `stage1`, `init-review`, `migrate-review`, `prepare-review-packets`, `review-status`, `build-baseline`, and `finalize`. |
| `scripts/apply_review_proposals.py` | Validates a complete Codex proposal packet, applies it atomically to the authoritative review queue, and writes an immutable audit. Required only when reviewing a new evaluation sample. |
| `scripts/run_benchmark.py` | Runs dense, BM25, or hybrid retrieval evaluation and optionally answer generation and RAGAS. Its `--variant-manifest` preflight prevents mismatched test sets and indexes. |
| `scripts/format_review_queue.py` | Converts the older flat Gemini review queue into hierarchical JSON. It is retained for legacy data and is not needed for the completed Codex-reviewed benchmark. |

Do not use `scripts/prepare_testset.py`, `scripts/build_mini_testset.py`, or
`scripts/build_chroma_collection.py` to recreate the locked benchmark. They are
older or general-purpose utilities and do not implement the complete selection,
review, manifest, and finalization contract.

### Implementation modules

| File | Role |
| --- | --- |
| `src/evaluation/testset.py` | Locked NewsQA sampling, source extraction, evidence validation, chunk relevance mapping, artifact derivation, hashing, and manifest helpers. |
| `src/evaluation/question_review.py` | Review schema, packet creation, proposal validation, human-decision validation, answer correction, clarification, and exclusion handling. |
| `src/evaluation/metrics.py` | Retrieval, answer, and RAGAS metrics used by the benchmark runner. |
| `src/ingestion/chunker.py` | Production chunking implementation used to create the shared corpus chunks. |
| `src/indexing/embeddings.py` | Embedding factory; this benchmark uses local `all-MiniLM-L6-v2` embeddings. |
| `src/indexing/chroma_store.py` | Persistent Chroma collection creation, insertion, querying, and count validation. |
| `src/indexing/bm25_index.py` | BM25 construction, persistence, and retrieval over the same chunks as Chroma. |

### Configuration and environment

| File | Role |
| --- | --- |
| `configs/config.yaml` | Locked chunking, embedding, database, retrieval, reranking, generation, and evaluation configuration. The variant manifest records its hash. |
| `requirements.txt` | Python dependencies needed to construct and evaluate the dataset. |
| `.env` | Local API credentials for generation/RAGAS only. Never include this file in a shared archive. Retrieval construction uses local embeddings and does not require an embedding API key. |
| `tests/test_evaluation_dataset.py` | Dataset selection, review migration, artifact derivation, manifest preflight, and finalization contract tests. |

Create the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first build requires access to the locked NewsQA revision, the Tiktoken
encoding, and the sentence-transformer model unless they are already cached.
Validate the code before a build with:

```bash
.venv/bin/python -m unittest tests.test_evaluation_dataset
```

## Canonical generation workflow

Run every command from the repository root.

### 1. Select the corpus

```bash
.venv/bin/python scripts/prepare_evaluation_dataset.py stage1 --selection-only
```

This scans the locked NewsQA revision with seed `42` and writes:

```text
data/evaluation/newsqa_200_11064/staging/corpus/evaluation_articles.jsonl
data/evaluation/newsqa_200_11064/staging/corpus/distractor_articles.jsonl
data/evaluation/newsqa_200_11064/staging/questions/original_questions.jsonl
evaluation/manifests/newsqa_200_11064.selection.json
```

Expected counts are 200 evaluation articles, 10,864 distractors, and 1,340
questions. The selection manifest records the dataset revision, seed, selected
article IDs, filtering statistics, and hashes.

### 2. Provide reviewed annotations

For this exact 200-article evaluation sample, reuse the completed review:

```bash
.venv/bin/python scripts/prepare_evaluation_dataset.py migrate-review
```

Migration is allowed only when the complete evaluation articles and protected
source questions are identical to `newsqa_200_1000`. It writes:

```text
data/evaluation/newsqa_200_11064/staging/review/review_queue_readable.json
data/evaluation/newsqa_200_11064/staging/review/manifest.json
data/evaluation/newsqa_200_11064/staging/review/audits/schema.json
```

The completed bundle already contains these files, so a recipient does not
need to run migration again.

For a different evaluation sample, do not migrate the old review. Use:

```bash
.venv/bin/python scripts/prepare_evaluation_dataset.py init-review
.venv/bin/python scripts/prepare_evaluation_dataset.py prepare-review-packets
```

Review every packet with Codex, apply each proposal with
`scripts/apply_review_proposals.py`, complete every `human_review` decision in
`review_queue_readable.json`, preserve the audit files, and then run:

```bash
.venv/bin/python scripts/prepare_evaluation_dataset.py review-status
```

Do not continue until the report has `ready: true`, no `pending` decisions, and
no `needs_adjudication` decisions.

### 3. Build the test set and retrieval indexes

```bash
.venv/bin/python scripts/prepare_evaluation_dataset.py build-baseline
```

This command applies `configs/config.yaml` to all 11,064 articles and creates:

```text
data/evaluation/newsqa_200_11064/final/testset_original.jsonl
data/evaluation/newsqa_200_11064/final/chunks.jsonl
data/evaluation/newsqa_200_11064/final/bm25.pkl
data/evaluation/newsqa_200_11064/final/integrity_report.json
data/chroma_db/
evaluation/manifests/newsqa_200_11064.variant.json
```

The finalized collection is `newsqa_val200_s42_7e16e8_66785e` with 19,263
chunks. Use `--overwrite` only when intentionally replacing an existing
collection with the same name. Use `--skip-index` only for artifact inspection;
it does not produce a runnable retrieval benchmark.

### 4. Finalize reviewed variants

```bash
.venv/bin/python scripts/prepare_evaluation_dataset.py review-status
.venv/bin/python scripts/prepare_evaluation_dataset.py finalize
```

Finalization verifies the selection, config and artifact hashes, Chroma count,
review completeness, evidence offsets, corrections, clarifications, and
exclusions. It adds:

```text
data/evaluation/newsqa_200_11064/final/testset_reviewed_original.jsonl
data/evaluation/newsqa_200_11064/final/testset_resolved.jsonl
data/evaluation/newsqa_200_11064/final/testset_clarified.jsonl
data/evaluation/newsqa_200_11064/final/excluded_questions.jsonl
data/evaluation/newsqa_200_11064/final/review_annotations.jsonl
```

The variant manifest and integrity report are updated to
`status: review_complete` and `phase: review_complete`.

## Complete generated file inventory

### Staging and review files

| Path | Purpose |
| --- | --- |
| `staging/corpus/evaluation_articles.jsonl` | Full text, metadata, and source questions for the fixed 200 evaluation articles. |
| `staging/corpus/distractor_articles.jsonl` | Full text and metadata for all 10,864 train distractors. |
| `staging/questions/original_questions.jsonl` | Immutable flattened source-question records. |
| `staging/review/review_queue_readable.json` | Authoritative Codex proposal and human-review document for all 1,340 questions. |
| `staging/review/manifest.json` | Review mode, counts, artifact hashes, selection reference, and migration provenance. |
| `staging/review/audits/schema.json` | Schema governing append-only review audits. |

### Final evaluation files

| Path | Rows/items | Purpose |
| --- | ---: | --- |
| `final/testset_original.jsonl` | 1,340 | Immutable raw baseline, including questions later excluded. |
| `final/testset_reviewed_original.jsonl` | 1,336 | Primary scored benchmark with original wording and reviewed answers/evidence. |
| `final/testset_resolved.jsonl` | 1,336 | Same questions and gold data as the primary set, using approved clarification when available. |
| `final/testset_clarified.jsonl` | 1,078 | Paired clarified subset for ambiguity analysis. |
| `final/excluded_questions.jsonl` | 4 | Non-scored questions with explicit exclusion reasons. |
| `final/review_annotations.jsonl` | 1,340 | Final review and correction audit table. |
| `final/chunks.jsonl` | 19,263 | Shared chunks for every evaluation variant and both retrieval indexes. |
| `final/bm25.pkl` | 1 index | Serialized sparse index tied to `chunks.jsonl`. |
| `final/integrity_report.json` | 1 report | Final counts, readiness state, and integrity result. |

### Manifests and dense database

| Path | Purpose |
| --- | --- |
| `evaluation/manifests/newsqa_200_11064.selection.json` | Reproducible selection and source-artifact manifest. |
| `evaluation/manifests/newsqa_200_11064.variant.json` | Final hashes, config, collection, chunk count, and integrity manifest. |
| `data/chroma_db/` | Persistent Chroma database. It currently contains the finalized collection and may also contain collections from earlier experiments. |

The dataset directory is approximately 130 MB. The current shared Chroma
directory is approximately 369 MB. Both `data/` and generated reports are
Git-ignored, so pushing the repository does not share these artifacts.

## Sharing options

Stop any API, benchmark, or indexing process that may be writing to Chroma
before creating an archive. Always create the archive from the repository root
so manifest-relative paths remain valid. Exclude `.env`, `.venv`, `.DS_Store`,
API keys, and generated reports.

### Option A: Evaluation artifacts without Chroma

This is the recommended smaller bundle. It supports inspection, JSONL scoring,
BM25 use, and deterministic Chroma reconstruction.

```bash
zip -r -9 newsqa_200_11064_artifacts.zip \
  data/evaluation/newsqa_200_11064 \
  evaluation/manifests/newsqa_200_11064.selection.json \
  evaluation/manifests/newsqa_200_11064.variant.json \
  configs/config.yaml \
  requirements.txt \
  docs/evaluation_dataset.md \
  docs/final_evaluation_output.md \
  docs/evaluation_dataset_handoff.md \
  -x '*/.DS_Store'

shasum -a 256 newsqa_200_11064_artifacts.zip \
  > newsqa_200_11064_artifacts.zip.sha256
```

The recipient first verifies the archive in the directory containing both
downloaded files, then extracts it into the root of the same project version:

```bash
cd /path/to/archive-directory
shasum -a 256 -c newsqa_200_11064_artifacts.zip.sha256

cd /path/to/Text-Mining---NewsQA-RAG
unzip /path/to/archive-directory/newsqa_200_11064_artifacts.zip -d .
```

If dense retrieval is needed, reconstruct the database and restore final status:

```bash
.venv/bin/python scripts/prepare_evaluation_dataset.py build-baseline --overwrite
.venv/bin/python scripts/prepare_evaluation_dataset.py finalize
```

Rebuilding uses local sentence-transformer inference but may download the model
and tokenizer on first use.

### Option B: Ready-to-run retrieval bundle

Add the persistent database to avoid rebuilding 19,263 embeddings:

```bash
zip -r -9 newsqa_200_11064_runnable.zip \
  data/evaluation/newsqa_200_11064 \
  data/chroma_db \
  evaluation/manifests/newsqa_200_11064.selection.json \
  evaluation/manifests/newsqa_200_11064.variant.json \
  configs/config.yaml \
  requirements.txt \
  docs/evaluation_dataset.md \
  docs/final_evaluation_output.md \
  docs/evaluation_dataset_handoff.md \
  -x '*/.DS_Store'

shasum -a 256 newsqa_200_11064_runnable.zip \
  > newsqa_200_11064_runnable.zip.sha256
```

Chroma stores multiple collections in a shared SQLite database and segment
directories. Do not copy an individual UUID directory; share the complete
`data/chroma_db` directory. This bundle may include unrelated older collections,
but the variant manifest selects only
`newsqa_val200_s42_7e16e8_66785e`.

### Option C: Provenance-complete academic bundle

The `newsqa_200_11064` review was migrated from the unchanged 200-article review
in `newsqa_200_1000`. To preserve the proposal packets and immutable review
audits referenced by the migration manifest, add:

```text
data/evaluation/newsqa_200_1000/staging/review/
evaluation/manifests/newsqa_200_1000.selection.json
```

These files add approximately 12 MB. They are not needed to run the benchmark,
but they preserve the full review provenance for presentations, auditing, and
academic reproducibility.

To create this form, use either archive command above and add both provenance
paths before the `-x '*/.DS_Store'` exclusion.

## Recipient verification

From the repository root, verify the review and manifest-backed retrieval setup:

```bash
.venv/bin/python scripts/prepare_evaluation_dataset.py review-status

.venv/bin/python scripts/run_benchmark.py \
  --retriever hybrid \
  --testset data/evaluation/newsqa_200_11064/final/testset_reviewed_original.jsonl \
  --variant-manifest evaluation/manifests/newsqa_200_11064.variant.json \
  --n-eval 1 \
  --report-dir reports/newsqa_200_11064/smoke
```

The expected review status is `ready: true`. Manifest preflight must pass, the
collection must contain 19,263 chunks, and the smoke benchmark must complete.

For the full experiment, evaluate `testset_reviewed_original.jsonl` as the
primary result and `testset_resolved.jsonl` separately as the clarification
treatment. Never merge original and clarified variants into one aggregate.

## Sharing checklist

- Share code through Git and data artifacts through an archive or approved
  institutional storage.
- Use the same repository version and extract archives at the repository root.
- Include both `newsqa_200_11064` manifests and `configs/config.yaml`.
- Include all of `data/chroma_db` or omit it and rebuild; never copy Chroma
  segment directories selectively.
- Include the old review directory when full proposal/audit provenance matters.
- Generate and verify a SHA-256 checksum for the archive.
- Never share `.env`, credentials, `.venv`, caches, or unrelated reports.
- Check the NewsQA distribution terms before sharing full article text outside
  the project team or institution.
