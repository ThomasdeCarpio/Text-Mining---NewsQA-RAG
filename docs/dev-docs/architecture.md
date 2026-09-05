# Architecture

```text
frontend -> FastAPI routers -> services -> RAGAgent -> retrieval -> reranker -> LLM
                                          \-> direct LLM fallback

articles -> clean -> chunk -> Chroma + BM25
testset + index manifest -> benchmark -> experiment comparison
```

## Code map

| Path | Owns |
| --- | --- |
| `app/frontend/src/` | React routes, screens, API/SSE client |
| `app/assets/` | Static files the frontend serves (Vite `publicDir`) |
| `app/backend/newsqa_app/api/` | FastAPI composition and HTTP contracts |
| `app/backend/newsqa_app/services/` | Request and dashboard workflows |
| `common/newsqa_rag/agents/` | Single retrieve/rerank/generate pass |
| `common/newsqa_rag/retrieval/` | Dense, BM25, hybrid, rerankers |
| `common/newsqa_rag/indexing/` | Embeddings and persisted indexes |
| `common/newsqa_rag/ingestion/` | Load, clean, chunk |
| `common/newsqa_rag/evaluation/` | Dataset/metric/checkpoint primitives |
| `common/newsqa_rag/experiments.py` | Matrix, partitions, runner, comparison |
| `scripts/` | Thin offline CLI entry points |
| `outputs/` | Every run artifact: benchmarks, experiments, EDA, frontend build |

Two packages, both installed by `pyproject.toml`: `common/newsqa_rag` as
`newsqa_rag` (shared by the app, the scripts, and the notebooks) and
`app/backend/newsqa_app` as `newsqa_app` (the FastAPI layer, which nothing
outside `app/` imports). Import from those packages directly; do not add nested
`src` paths or manipulate `sys.path` in library code.

## Runtime boundaries

- API startup is light; Chroma, embedding, reranker, and LLM clients load on first use.
- Chat history and activity logs are process-local memory and disappear on restart.
- Admin login is for a coursework demo; the backend does not provide production authorization.
- RAG is single-pass. Multi-source planning is not implemented.
- CLI and dashboard evaluation share the same experiment runner and report artifacts.

## Entry points

```bash
python -m uvicorn newsqa_app.api.main:app --reload
cd app/frontend && npm run dev
python scripts/build_chroma_collection.py --help
python scripts/run_experiment.py configs/experiments/newsqa_retrieval_smoke.yaml --dry-run
```

Change HTTP behavior in a router plus its service; retrieval behavior under
`retrieval/`; provider behavior in `llm.py`/`model_gateway.py`; benchmark
contracts under `evaluation/`; and research notebooks under `notebooks/`.
