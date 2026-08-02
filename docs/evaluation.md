# Evaluation pipeline

CLI and dashboard share one resumable path:

```text
locked testset + matching index manifest
  -> collect retrieval/generation traces
  -> score deterministic metrics
  -> judge with RAGAS (optional)
  -> summarize experiment runs
```

## Required invariants

- Testset, chunks, BM25, Chroma collection, config, and manifest must describe the same corpus.
- Original and resolved variants use the same locked articles and source question IDs.
- Successful questions are reused on resume; failed attempts remain auditable.
- Development and final-test articles never overlap.
- Final-test stays closed until development choices are locked.

Manifest preflight runs before model initialization. The prepared NewsQA corpus
contains 200 evaluation articles, 10,864 distractors, and 19,263 indexed chunks.

## Retrieval stages

The scorer keeps both rankings:

- `retrieval_initial`: top K returned by dense, BM25, or hybrid retrieval;
- `retrieval`: the final top N after no-op truncation or cross-encoder reranking.

This distinction explains cases where a gold chunk starts at rank 9, is removed
by a top-5 no-op baseline, but is moved into the top five by a reranker.

## Metrics

| Group | Metrics |
| --- | --- |
| Retrieval | Hit rate, recall, MRR, NDCG at K |
| Answer | Exact match and token F1 over accepted answers |
| Citation | Validity, precision, recall, F1 |
| Operations | Coverage, failures, stage latency, token usage, estimated cost |
| Comparison | Article macro mean, bootstrap 95% CI, paired delta, Pareto frontier |

A retrieval failure means the gold chunk missed the evaluated cutoff. Verify
chunk presence and collection count separately before diagnosing missing data.

## Run artifacts

| File | Purpose |
| --- | --- |
| `run_manifest.json` | Fingerprint, paths, status, and controls |
| `retrievals.jsonl` | Resumable retrieval cache |
| `predictions.jsonl` | Answers, citations, latency, usage |
| `deterministic_scores.jsonl` | Per-question scores and pairing keys |
| `report.json` | Aggregate metrics, coverage, and failures |
| `judge_results.jsonl` | Optional cached judge scores |
| `environment.json` | Git/runtime/hardware provenance |

Use [benchmarking.md](benchmarking.md) for one-off commands and
[experiments.md](experiments.md) for matrix runs and dashboard behavior.
