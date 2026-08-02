# Evaluation pipeline

This is the technical overview. Use [benchmarking.md](benchmarking.md) for
commands and [evaluation_dataset.md](evaluation_dataset.md) for dataset review.

## Scope

The current benchmark evaluates a fixed single-pass pipeline:

```text
question → retrieve → optional rerank → optional generate → score → optional judge
```

It does not evaluate multi-step planning or tool selection.

## Stages

| Stage | Implementation | Output |
| --- | --- | --- |
| Dataset/index validation | `apply_manifest_preflight` | A coherent test set, chunks, BM25 index, and Chroma collection |
| Collection | `collect_benchmark_predictions.py` | Resumable retrieval and prediction JSONL traces |
| Deterministic scoring | `score_benchmark_predictions.py` | Retrieval, QA, citation, latency, and coverage metrics |
| LLM judging | `judge_benchmark_predictions.py` | Cached RAGAS scores |
| Reporting | scorer + `eval_service.py` | JSON/text reports and dashboard responses |

`backend/newsqa_rag/evaluation/benchmark_io.py` owns shared artifact, preflight, retry, resume,
and summary behavior. Metric functions in `backend/newsqa_rag/evaluation/metrics.py` are pure
and do not access the database.

## Required invariants

1. The test set and retrieval indexes come from the same article corpus and
   chunking configuration.
2. Every scored question has at least one `relevant_chunk_ids` value.
3. A run directory belongs to one immutable fingerprint. Changed inputs or
   models require a new directory.
4. Original and resolved variants remain separate paired runs.
5. Failed generations stay in deterministic coverage/QA denominators.
6. The primary judge model differs from the generator model.

## Main records

### Test-set row

```json
{
  "question_id": "q-123",
  "article_id": "newsqa_...",
  "question": "Who won the hearing?",
  "ground_truth": "Alice",
  "accepted_answers": ["Alice"],
  "relevant_chunk_ids": ["newsqa_..._chunk_0"]
}
```

### Prediction record

Each row records `question_id`, status, attempts, ground truth, relevant chunk
IDs, and either a successful full RAG trace or a sanitized terminal error.
Retrieval is checkpointed separately so generation retries do not query the
index again.

### Report

`report.json` contains:

- `coverage`: expected, recorded, successful, failed, and missing counts.
- `retrieval_initial` and `retrieval`: Hit Rate, Recall, MRR, and NDCG.
- `reranker_delta`: post-rerank change in MRR/NDCG.
- `qa`: Exact Match and token F1 against all accepted answers.
- `citations`: validity and gold-chunk precision/recall/F1.
- `latency`: mean, p50, p95, max, and sample count by stage.
- `ragas`: optional judge metrics plus judge coverage.

## Extension points

| Change | Main location | Rebuild required? |
| --- | --- | --- |
| Chunking | `backend/newsqa_rag/ingestion/chunker.py` | Chunks, evidence mapping, BM25, Chroma |
| Embedding | `backend/newsqa_rag/indexing/embeddings.py` | Chroma collection |
| Retrieval/fusion | `backend/newsqa_rag/retrieval/` | Usually no; reuse matching indexes |
| Reranker | `backend/newsqa_rag/retrieval/reranker.py` | No; reuse retrieval candidates |
| Generator/prompt | `backend/newsqa_rag/llm.py`, `backend/newsqa_rag/agents/rag_agent.py` | No; reuse retrieval traces when compatible |
| Metrics | `backend/newsqa_rag/evaluation/metrics.py` | No; rescore saved predictions |

The next planned layer is experiment orchestration: YAML specs, fixed
article-level development/final-test partitions, run matrices, cross-run
summaries, confidence intervals, and cost/environment logging. It should call
the existing scripts rather than duplicate the pipeline.
