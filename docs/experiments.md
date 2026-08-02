# Run and inspect experiments

An experiment YAML defines a locked dataset/index pair and a matrix of retrieval
settings. CLI and dashboard use the same runner and artifacts.

## Quick start in the dashboard

1. Start the API and frontend.
2. Sign in as `admin` and open **Evaluation Desk**.
3. Select an experiment.
4. Use **Preview config** to inspect its partitions and run matrix.
5. Use **Run / resume** for missing work, or **Load saved results** for an existing report.

The bundled `newsqa_retrieval_smoke.yaml` expands to eight retrieval-only runs:

```text
2 question variants x 2 retrievers x 2 rerankers = 8 runs
```

Each run samples 10 questions from the locked 50-article development partition.
It is a UI/pipeline smoke test, not a final benchmark.

### Dashboard controls

| Control | What it does | Model work |
| --- | --- | --- |
| Preview config | Validates and displays articles, questions, and matrix | None |
| Run / resume | Executes incomplete runs and reuses successful questions | Only missing work |
| Rebuild summary | Recreates comparison files from completed artifacts | None |
| Load saved results | Reads existing comparison and per-run reports | None |

A completed experiment is immutable. For a genuinely fresh run, copy its YAML
and change `experiment.id`; do not delete or overwrite the old result.

## Read the results

Choose a run in **Run being viewed**. Metric cards and Failure Analysis always
refer to that exact run. The bar chart compares every run in the selected
experiment.

| Metric | Meaning |
| --- | --- |
| MRR@5 | Reciprocal rank of the first gold chunk in the top five; higher is better |
| NDCG@5 | Rewards gold chunks appearing near the top; higher is better |
| Recall@5 | Fraction of gold chunks present in the top five |
| P95 latency | 95% of questions completed within this time, including cold starts |
| Coverage | Fraction of expected questions that completed successfully |
| Run time | Wall-clock time for the whole configuration |

`★` marks a run that is not dominated on both MRR@5 and P95 latency.

### Interpret failures correctly

`No ground-truth chunk in reranked top 5` means the gold chunk was not in the
final evaluated top five. It does not by itself mean the article or chunk is
missing from the database.

For example, a retriever may find the gold chunk at rank 9:

- `noop` keeps only the configured top five, so the question fails Recall@5;
- a cross-encoder may move rank 9 to rank 3, so the same question passes.

Check corpus integrity and collection count before treating retrieval failures
as data-loading failures.

## CLI equivalent

```bash
# Validate and print commands only
python scripts/run_experiment.py configs/experiments/newsqa_retrieval_smoke.yaml --dry-run

# Run or resume
python scripts/run_experiment.py configs/experiments/newsqa_retrieval_smoke.yaml

# Rebuild comparison.json and comparison.csv
python scripts/summarize_experiments.py reports/experiments/newsqa-retrieval-smoke
```

## Main YAML fields

| Section | Meaning |
| --- | --- |
| `dataset.indexes` | Config, variant manifest, and testsets for each index |
| `fixed` | Controls shared by every run |
| `matrix` | Values to expand: variant, retriever, reranker, K, model, partition |
| `runtime` | Retry count, progress, and optional smoke `n_eval` |
| `judge` | Optional LLM judge provider/model |
| `summary` | Metrics used for confidence intervals, paired delta, and Pareto comparison |

Keep the development article count and seed fixed while tuning. Open the
final-test partition only after selecting and locking the final configuration.

## Artifacts and resume

Results live under `reports/experiments/<experiment-id>/`:

```text
registry.json                 run status and wall time
comparison.json / .csv       cross-run comparison
partitions/                   locked article and question IDs
<run-id>/report.json          aggregate metrics and failures
<run-id>/predictions.jsonl    generated/retrieval results
<run-id>/retrievals.jsonl     reusable retrieval traces
<run-id>/environment.json     Git, Python, package, and hardware provenance
```

Stopping the API interrupts the active subprocess, but saved JSONL records
remain reusable. Start the API again and choose **Run / resume**.

## Hugging Face downloads

Each run uses an isolated Python subprocess. Sentence Transformers may check
Hugging Face and reload cached model weights for each subprocess. Usually only
the first request downloads the model; later runs use the local cache.

After all required models are cached, network checks can be disabled before
starting the API or CLI:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

Do not enable offline mode before the embedding and cross-encoder models have
been downloaded at least once.
