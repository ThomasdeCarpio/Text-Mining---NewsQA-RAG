# Run one benchmark

Use this path for debugging one configuration. Use [experiments.md](experiments.md)
for comparisons.

## Retrieval-only run

```bash
python scripts/collect_benchmark_predictions.py --retriever hybrid --reranker noop --retrieval-only --testset data/evaluation/newsqa_200_11064/final/testset_reviewed_original.jsonl --variant-manifest evaluation/manifests/newsqa_200_11064.variant.json --run-dir reports/benchmarks/hybrid_noop --progress
python scripts/score_benchmark_predictions.py --run-dir reports/benchmarks/hybrid_noop
```

`collect` validates the manifest before loading models. Repeating the same
command resumes successful questions. Add `--retry-failed` to retry exhausted
questions.

Useful controls:

- `--n-eval 10`: deterministic smoke subset.
- `--question-ids-file ids.json`: use a locked JSON list of IDs.
- `--top-k 10 --rerank-top-n 5`: retrieval depth.
- `--reranker cross-encoder`: enable local reranking.

## Answer generation

Remove `--retrieval-only` and optionally add `--generator-model MODEL`. Provider
credentials come from `.env`; they are never written to reports.

```bash
python scripts/collect_benchmark_predictions.py --retriever hybrid --reranker cross-encoder --generator-model gpt-4o-mini --testset data/evaluation/newsqa_200_11064/final/testset_resolved.jsonl --variant-manifest evaluation/manifests/newsqa_200_11064.variant.json --run-dir reports/benchmarks/resolved_rag --progress
python scripts/score_benchmark_predictions.py --run-dir reports/benchmarks/resolved_rag
```

## Optional LLM judge

```bash
python scripts/judge_benchmark_predictions.py --run-dir reports/benchmarks/resolved_rag --judge-provider openai --judge-model gpt-4.1-mini --n-eval 50 --progress
python scripts/score_benchmark_predictions.py --run-dir reports/benchmarks/resolved_rag
```

Use a judge model different from the generator. The judge reads cached
predictions and can resume independently.

## Corpus check

Run once before a final benchmark:

```bash
python scripts/benchmark_corpus.py --variant-manifest evaluation/manifests/newsqa_200_11064.variant.json --output reports/benchmarks/corpus.json
```
