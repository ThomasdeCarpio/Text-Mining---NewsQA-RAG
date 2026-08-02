# NewsQA RAG

Question answering over the NewsQA/CNN corpus with dense, BM25, and hybrid
retrieval, optional cross-encoder reranking, cited LLM answers, and a resumable
evaluation pipeline.

## What works

- Ingestion, cleaning, chunking, Chroma indexing, and BM25 indexing.
- Dense, BM25, and hybrid retrieval.
- Optional cross-encoder reranking.
- Direct chat and retrieval-augmented chat with numbered citations.
- Reproducible NewsQA evaluation datasets with evidence-to-chunk mapping,
  review, manifests, and semantic deduplication.
- Resumable retrieval/generation traces, deterministic scoring, RAGAS judging,
  and report-backed dashboard metrics.
- FastAPI backend and React/Vite UI.

Multi-step, multi-source agentic planning is not implemented. The current RAG
path performs one retrieve → rerank → generate pass.

## Setup

Python 3.11+ and Node.js 20+ are recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure the provider keys you use. Local
BM25 needs no API key; dense retrieval may download a Hugging Face model on
first use.

## Run the app

```bash
# Backend: http://localhost:8000
python -m uvicorn newsqa_rag.api.main:app --reload

# Frontend: http://localhost:5173
cd frontend
npm install
npm run dev
```

Useful checks:

```bash
python scripts/prepare_evaluation_dataset.py --help
python scripts/inspect_collection.py --help
python scripts/check_model_gateway.py
```

## Evaluation

The canonical pipeline is split so expensive work is cached:

```text
dataset + matching index
        ↓
collect_benchmark_predictions.py   # retrieval/generation; resumable
        ↓
score_benchmark_predictions.py     # deterministic; no API calls
        ↓
judge_benchmark_predictions.py     # optional LLM judge; resumable
```

The test set and collection must use the same articles, chunker, embedding
configuration, and chunk IDs. Variant manifests enforce this before model
initialization.

See [docs/benchmarking.md](docs/benchmarking.md) for runnable commands and
[docs/evaluation_dataset.md](docs/evaluation_dataset.md) for dataset creation
and review.

## Repository map

| Path | Purpose |
| --- | --- |
| `backend/newsqa_rag/api/` | FastAPI routes and schemas |
| `frontend/` | React/Vite client |
| `backend/newsqa_rag/ingestion/` | Load, clean, and chunk articles |
| `backend/newsqa_rag/indexing/` | Embeddings, Chroma, and BM25 indexes |
| `backend/newsqa_rag/retrieval/` | Dense, BM25, hybrid, and reranking logic |
| `backend/newsqa_rag/agents/` | Single-pass RAG pipeline |
| `backend/newsqa_rag/evaluation/` | Dataset, metrics, cache, retry, and review logic |
| `backend/newsqa_rag/crawler/` | News discovery, parsing, and storage |
| `experiments/notebooks/` | Experiment and benchmark notebooks |
| `scripts/` | CLI entry points |
| `evaluation/` | Versioned manifests and review decisions |
| `reports/` | Benchmark artifacts and reports |
| `tests/` | Offline unit and integration tests |

## Documentation

| Document | Use it for |
| --- | --- |
| [Evaluation pipeline](docs/evaluation.md) | Architecture, contracts, metrics, invariants |
| [Benchmarking](docs/benchmarking.md) | Collect, resume, score, and judge runs |
| [Evaluation dataset](docs/evaluation_dataset.md) | Build/review/finalize NewsQA variants |
| [Database](docs/database.md) | Chunk IDs, metadata, Chroma, embeddings |
| [Model gateway](docs/model_gateway.md) | Provider and model configuration |
| [UI](docs/ui.md) | Screens and API-facing behavior |

Presentation material remains under `docs/presentation/` and `docs/slides/`.

## Remaining work

1. Add an experiment-spec runner around the existing collect/score/judge
   scripts: YAML validation, matrix expansion, run registry, and resume.
2. Split the 200 evaluation articles into fixed development/final-test
   partitions before further tuning.
3. Add cross-run summaries, paired confidence intervals, cost/environment
   logging, and Pareto comparison.
4. Run the planned chunking, embedding, retrieval, reranking, and generation
   ablations; lock configs before opening the final-test partition.
5. Add multi-source agentic orchestration only if it remains part of the
   project scope.

Do not add production auth, Redis sessions, or deployment infrastructure for
the coursework demo unless deployment becomes a real requirement.
