# Resumable RAG Benchmark

This is the canonical benchmark workflow for the finalized
`newsqa_200_11064` dataset. It evaluates the fixed RAG pipeline implemented by
`RAGAgent`: retrieval, reranking, cited generation, and later scoring. It does
not evaluate an agentic orchestrator or tool-selection policy.

## Why collection and evaluation are separate

`collect_benchmark_predictions.py` is the only stage that queries the database
or generator. It checkpoints retrieval before generation and appends one result
per question. Re-running the exact command resumes the run and skips successful
questions. `score_benchmark_predictions.py` and
`judge_benchmark_predictions.py` operate only on the saved trace; they never
rerun retrieval or generation.

Do not edit a run manifest or combine the reviewed-original and resolved test
sets. Each run fingerprint locks the dataset, question IDs, collection,
retriever, reranker, generator, prompt, and ranking sizes.

## Components and metrics

| Component | Metrics |
| --- | --- |
| Chunking | Token-size distribution, metadata completeness, semantic integrity, duplicate rate |
| Indexing | Manifest/count integrity, self-retrieval Recall@1, duplicate rate, write latency when rebuilding |
| Retrieval | Hit Rate, Recall, MRR, and binary NDCG at 1, 3, 5, and 10 |
| Reranking | Post-rerank metrics, delta MRR, delta NDCG, latency |
| Generation | Best accepted-answer Exact Match/F1, answer correctness, faithfulness, answer relevancy |
| Context | RAGAS context precision and context recall |
| References | Citation validity, gold-chunk precision/recall/F1, answer citation coverage |
| End to end | Success/failure coverage and mean, p50, p95, and maximum latency |

The deterministic retrieval judge compares ordered chunk IDs with
`relevant_chunk_ids`. The answer judge first computes EM/F1 against every
`accepted_answers` value. RAGAS then uses an LLM to score the saved answer and
the exact contexts seen by the generator. Failed answers remain in the primary
EM/F1 denominator and RAGAS reports its successful-answer coverage separately.

## 1. Collect a retrieval baseline

No API key is needed for BM25, deterministic scoring, or a locally cached
Sentence Transformers model. Dense/hybrid retrieval and the cross-encoder may
download their configured Hugging Face models on first use.

First record the one-time corpus and index diagnostics:

```bash
.venv/bin/python scripts/benchmark_corpus.py \
  --variant-manifest evaluation/manifests/newsqa_200_11064.variant.json \
  --output reports/benchmarks/newsqa_200_11064_corpus.json
```

Use `--skip-self-retrieval` when validating an offline bundle before the local
embedding model has been downloaded. That preserves all structural checks but
records self-retrieval Recall@1 as unmeasured. Index write latency cannot be
recovered from a completed persistent collection and must be measured during a
fresh build.

```bash
.venv/bin/python scripts/collect_benchmark_predictions.py \
  --retriever bm25 \
  --reranker noop \
  --retrieval-only \
  --testset data/evaluation/newsqa_200_11064/final/testset_reviewed_original.jsonl \
  --variant-manifest evaluation/manifests/newsqa_200_11064.variant.json \
  --run-dir reports/benchmarks/original_bm25_noop \
  --progress

.venv/bin/python scripts/score_benchmark_predictions.py \
  --run-dir reports/benchmarks/original_bm25_noop
```

Repeat with `dense` and `hybrid`, and with `--reranker cross-encoder`, using a
different run directory for each configuration. The local reranker defaults to
`cross-encoder/ms-marco-MiniLM-L-6-v2`. No-op runs are required baselines for
interpreting delta MRR/NDCG.

## 2. Collect cited RAG answers

The canonical end-to-end configuration is hybrid retrieval plus the local
cross-encoder. The example uses DeepSeek generation; `DEEPSEEK_API_KEY` must be
set in `.env`. Without that variable, the shared gateway uses
`OPENAI_API_KEY`/`OPENAI_BASE_URL` instead.

Start with a fixed plumbing pilot:

```bash
.venv/bin/python scripts/collect_benchmark_predictions.py \
  --retriever hybrid \
  --reranker cross-encoder \
  --generator-model deepseek-chat \
  --testset data/evaluation/newsqa_200_11064/final/testset_reviewed_original.jsonl \
  --variant-manifest evaluation/manifests/newsqa_200_11064.variant.json \
  --run-dir reports/benchmarks/original_hybrid_crossencoder_deepseek \
  --n-eval 50 --seed 42 --progress
```

The selected question IDs are part of the fingerprint. A full run therefore
uses a new run directory and omits `--n-eval`. Stop with Ctrl-C at any time and
rerun the identical command to resume. Add `--retry-failed` to reconsider
records that exhausted their default three attempts.

Run the full resolved set separately by changing the test set to
`testset_resolved.jsonl` and using a `resolved_*` run directory.

## 3. Score without API calls

```bash
.venv/bin/python scripts/score_benchmark_predictions.py \
  --run-dir reports/benchmarks/original_hybrid_crossencoder_deepseek
```

This writes `deterministic_scores.jsonl`, `report.json`, and
`report_summary.txt`. It is safe to rerun after more predictions or judge
results arrive.

## 4. Run the LLM judge

Use a judge different from the generator for reported results. For example, a
DeepSeek generator with an OpenAI judge needs both `DEEPSEEK_API_KEY` and
`OPENAI_API_KEY`.

```bash
# Fixed 50-question judge pilot
.venv/bin/python scripts/judge_benchmark_predictions.py \
  --run-dir reports/benchmarks/original_hybrid_crossencoder_deepseek \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --n-eval 50 --seed 42 --progress

# Resume and judge every remaining successful answer
.venv/bin/python scripts/judge_benchmark_predictions.py \
  --run-dir reports/benchmarks/original_hybrid_crossencoder_deepseek \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --progress

# Merge cached judge scores into report.json
.venv/bin/python scripts/score_benchmark_predictions.py \
  --run-dir reports/benchmarks/original_hybrid_crossencoder_deepseek
```

Judge records are fingerprinted by inference run, provider, model, metric set,
and RAGAS version. The command refuses to mix incompatible results. Using the
same generator and judge requires `--allow-same-judge` and must be disclosed as
a limitation.

Gemini can judge saved successful answers independently of the generator. It
uses `GEMINI_API_KEY`, Google's OpenAI-compatible endpoint, local
`all-MiniLM-L6-v2` embeddings, and bounded RAGAS concurrency:

```bash
.venv/bin/python scripts/judge_benchmark_predictions.py \
  --run-dir reports/benchmarks/full_resolved_hybrid_cross-encoder_deepseek-v4-flash \
  --judge-provider gemini \
  --judge-model gemini-3.1-flash-lite \
  --batch-size 5 --max-workers 4 \
  --seed 42 --progress
```

Only successful prediction records are judged. Rerunning the identical command
resumes the cache; add `--retry-failed` after a transient provider failure.

## Run artifacts

| File | Purpose |
| --- | --- |
| `run_manifest.json` | Immutable run fingerprint, paths, models, selected IDs, and completion counts |
| `attempts.jsonl` | Success/failure audit for every retrieval, generation, and judge attempt |
| `retrievals.jsonl` | Retrieval checkpoint reused by generation retries |
| `predictions.jsonl` | Final per-question inference records and terminal failures |
| `deterministic_scores.jsonl` | Per-question retrieval, QA, and citation scores |
| `judge_results.jsonl` | Resumable per-question RAGAS scores |
| `report.json` | Aggregate machine-readable report |
| `report_summary.txt` | Short human-readable summary |

The separate corpus report contains chunking, metadata, duplicate, collection
count, and index self-retrieval diagnostics.

Use `notebooks/05_run_final_benchmark.ipynb` to execute corpus checks,
retrieval experiments, generation, scoring, and judging through the same
resumable CLI commands. Its cost-bearing cells are disabled by default. Use
`notebooks/04_final_benchmark_analysis.ipynb` only to compare completed reports;
that notebook does not query the pipeline or spend API credit.

Install and open the optional notebook environment with:

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --user --name newsqa-rag --display-name "NewsQA RAG"
.venv/bin/jupyter lab notebooks/05_run_final_benchmark.ipynb
```

Select the `NewsQA RAG` kernel when Jupyter opens.
