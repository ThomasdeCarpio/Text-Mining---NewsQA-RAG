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
| `frontend/src/` | React routes, screens, API/SSE client |
| `backend/newsqa_rag/api/` | FastAPI composition and HTTP contracts |
| `backend/newsqa_rag/services/` | Request and dashboard workflows |
| `backend/newsqa_rag/agents/` | Single retrieve/rerank/generate pass |
| `backend/newsqa_rag/retrieval/` | Dense, BM25, hybrid, rerankers |
| `backend/newsqa_rag/indexing/` | Embeddings and persisted indexes |
| `backend/newsqa_rag/ingestion/` | Load, clean, chunk |
| `backend/newsqa_rag/evaluation/` | Dataset/metric/checkpoint primitives |
| `backend/newsqa_rag/experiments.py` | Matrix, partitions, runner, comparison |
| `scripts/` | Thin offline CLI entry points |

`pyproject.toml` installs `backend/newsqa_rag` as `newsqa_rag`. Import from that
package directly; do not add nested `src` paths or manipulate `sys.path` in
library code.

## Runtime boundaries

- API startup is light; Chroma, embedding, reranker, and LLM clients load on first use.
- Chat history and activity logs are process-local memory and disappear on restart.
- Admin login is for a coursework demo; the backend does not provide production authorization.
- RAG is single-pass. Multi-source planning is not implemented.
- CLI and dashboard evaluation share the same experiment runner and report artifacts.

## Entry points

```bash
python -m uvicorn newsqa_rag.api.main:app --reload
cd frontend && npm run dev
python scripts/build_chroma_collection.py --help
python scripts/run_experiment.py configs/experiments/newsqa_retrieval_smoke.yaml --dry-run
```

Change HTTP behavior in a router plus its service; retrieval behavior under
`retrieval/`; provider behavior in `llm.py`/`model_gateway.py`; and benchmark
contracts under `evaluation/` or `experiments.py`.
