# NewsQA RAG

Question answering over the NewsQA/CNN corpus with dense, BM25, and hybrid
retrieval, optional cross-encoder reranking, cited answers, and reproducible
evaluation experiments.

## Setup

Python 3.11+ and Node.js 20+ are recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. BM25 needs no API key; dense retrieval downloads
its Sentence Transformer model on first use.

## Run the app

```bash
# Terminal 1: http://localhost:8000
python -m uvicorn newsqa_rag.api.main:app --reload --port 8000

# Terminal 2: http://localhost:5173
cd frontend
npm install
npm run dev
```

Demo admin login: `admin` / `admin123`.

## Evaluation

```text
experiment YAML
  -> locked testset + matching index manifest
  -> collect retrieval/generation traces (resumable)
  -> score deterministic metrics (no API calls)
  -> judge with RAGAS (optional, resumable)
  -> compare runs in CLI or Evaluation Desk
```

Validate the bundled smoke matrix without loading models:

```bash
python scripts/run_experiment.py configs/experiments/newsqa_retrieval_smoke.yaml --dry-run
```

The testset and index must share articles, chunk IDs, chunking, and embedding
configuration. Variant manifests enforce this before model initialization.

- [Run experiments and use the dashboard](docs/experiments.md)
- [Run one benchmark configuration](docs/benchmarking.md)
- [Understand evaluation contracts and metrics](docs/evaluation.md)
- [Build or review the evaluation dataset](docs/evaluation_dataset.md)

## Repository map

| Path | Purpose |
| --- | --- |
| `backend/newsqa_rag/api/` | FastAPI routes and schemas |
| `backend/newsqa_rag/services/` | Chat, retrieval, and experiment workflows |
| `backend/newsqa_rag/agents/` | Single-pass RAG pipeline |
| `backend/newsqa_rag/retrieval/` | Dense, BM25, hybrid, and rerankers |
| `backend/newsqa_rag/indexing/` | Embeddings and persisted indexes |
| `backend/newsqa_rag/ingestion/` | Load, clean, and chunk articles |
| `backend/newsqa_rag/evaluation/` | Dataset, metrics, cache, retry, and review logic |
| `backend/newsqa_rag/experiments.py` | Matrix, partitions, runner, and summaries |
| `frontend/` | React/Vite client |
| `configs/experiments/` | Versioned experiment specifications |
| `scripts/` | CLI entry points |
| `evaluation/` | Dataset manifests and review decisions |
| `reports/` | Versioned benchmark and experiment artifacts |
| `tests/` | Offline tests |

## More documentation

- [Architecture](docs/architecture.md)
- [Database and chunk contract](docs/database.md)
- [Model gateway](docs/model_gateway.md)
- [Web UI](docs/ui.md)
- [Crawler](docs/crawler.md)

The current RAG path is one retrieve -> rerank -> generate pass. Multi-source
planning, production authentication, Redis sessions, and deployment
infrastructure are outside the current coursework scope.
