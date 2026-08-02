# Architecture reference

## Mental model

NewsQA RAG is a Python package plus a React client. Its main online path is a
single-pass RAG pipeline; its offline paths prepare the corpus and produce
reproducible evaluation reports.

```text
React UI -> FastAPI routers -> services -> RAGAgent -> retriever -> reranker -> LLM
                                      \-> direct LLM fallback

news HTML -> clean -> chunk -> Chroma (dense) + BM25 (sparse)

test set -> collect predictions -> deterministic score -> optional RAGAS judge
```

The package uses a `src`-style layout under `backend/`: `pyproject.toml` installs
`backend/newsqa_rag` as the `newsqa_rag` package.

## Architecture by layer

| Layer | Main files | Responsibility |
| --- | --- | --- |
| Web client | `frontend/src/` | Authentication state, routing, chat SSE consumption, retrieval playground, report dashboard |
| HTTP API | `backend/newsqa_rag/api/` | FastAPI app, route registration, request/response schemas |
| Application services | `backend/newsqa_rag/services/` | Chat mode selection, session state, retrieval facade, report-backed dashboard data |
| RAG pipeline | `backend/newsqa_rag/agents/rag_agent.py` | One retrieve -> rerank -> generate pass, timings, citation parsing |
| Retrieval | `backend/newsqa_rag/retrieval/` | Dense, BM25, hybrid RRF, reranker strategies, retriever factory |
| Models/indexes | `indexing/`, `llm.py`, `model_gateway.py` | Embeddings, Chroma persistence, BM25 persistence, OpenAI-compatible generation |
| Corpus preparation | `ingestion/`, `crawler/` | Discover/fetch articles; clean, chunk, and normalize them for indexing |
| Evaluation | `evaluation/`, `scripts/*benchmark*` | Dataset contracts, checkpoints, retries, metrics, RAGAS judging, reports |

## Folder map

```text
backend/newsqa_rag/          Installable Python package
  api/                       FastAPI composition root, routers, Pydantic schemas
  services/                  Request-level orchestration and in-memory state
  agents/                    Single-pass RAGAgent (`orchestrator.py` is empty)
  retrieval/                 Retriever/reranker interfaces and implementations
  indexing/                  Chroma, embedding adapters, persisted BM25
  ingestion/                 HTML cleaning and configurable chunking
  crawler/                   Feed discovery, Playwright fetch, parse, storage adapters
  evaluation/                Dataset/review logic, metrics, resumable benchmark I/O
  llm.py                     Thin chat-completions wrapper and RAG prompt formatting
  model_gateway.py           Provider/env selection and OpenAI-compatible clients
  tools/                     Empty placeholders; no runtime tools are wired

frontend/                    React 19 + TypeScript + Vite + Tailwind UI
  src/main.tsx               Browser bootstrap
  src/App.tsx                Route and role guards
  src/api/                   HTTP/SSE client and shared API types
  src/context/               localStorage-backed auth context
  src/pages/                 Login, chat, retrieval, dashboard screens
  src/components/            Reusable presentation components

scripts/                     Offline CLI entry points; thin orchestration over package code
configs/config.yaml          Embedding, chunking, index, retrieval, LLM, metric defaults
data/                        Local corpora, chunks/indexes, and test sets; mostly generated/large
evaluation/                  Versioned dataset manifests and human review decisions
reports/                     Benchmark run checkpoints and aggregate reports
experiments/                 Notebooks and experiment notes
tests/                       Offline unit/integration tests
docs/                        Operational and architecture documentation
outputs/                     Generated presentation artifacts
database/                    Local/test database artifacts
models/, dev-docs/           Currently empty reserved directories
```

## Main entry points

| Entry point | How to start | What it composes |
| --- | --- | --- |
| Backend API | `python -m uvicorn newsqa_rag.api.main:app --reload` | FastAPI, CORS, auth/chat/admin/retrieval routers |
| Frontend | `cd frontend; npm run dev` | Vite -> `src/main.tsx` -> router/auth provider -> `App.tsx` |
| Canonical index build | `python scripts/build_chroma_collection.py ...` | clean -> chunk/cache -> Chroma -> BM25 |
| Crawler | `python -m newsqa_rag.crawler.crawl_articles ...` | discover -> fetch -> parse/filter -> selected storage backend |
| Evaluation dataset | `python scripts/prepare_evaluation_dataset.py ...` | NewsQA sampling, evidence mapping, review/manifests |
| Benchmark collection | `python scripts/collect_benchmark_predictions.py ...` | config + test set + indexes -> resumable traces/predictions |
| Deterministic scoring | `python scripts/score_benchmark_predictions.py ...` | cached predictions -> retrieval/QA/citation metrics + report |
| Optional judging | `python scripts/judge_benchmark_predictions.py ...` | successful predictions -> resumable RAGAS scores |
| Legacy combined benchmark | `python scripts/run_benchmark.py ...` | Older all-in-one benchmark path |

`scripts/ingest.py` is a smaller hard-coded ingestion path. Prefer
`build_chroma_collection.py`, which reads `configs/config.yaml`, caches chunks,
and builds both dense and sparse indexes. `scripts/query.py` currently contains
only usage documentation and is not executable.

## Startup flow

### Backend

```text
uvicorn imports newsqa_rag.api.main:app
  -> load project-root .env
  -> construct FastAPI app and CORS middleware
  -> register /auth, /chat, /admin, /retrieval, /health
  -> wait for requests
```

Heavy optional dependencies are deliberately lazy. Chroma, embedding models,
rerankers, and the model client initialize on first use, not API startup.

### Frontend

```text
Vite serves index.html
  -> src/main.tsx mounts React
  -> BrowserRouter + AuthProvider wrap App
  -> AuthProvider restores user/session from localStorage
  -> App routes to /login, /chat, /retrieval, or /dashboard
```

`RequireAuth` and `RequireAdmin` are UI guards. The backend does not validate the
returned session ID or enforce roles on admin routes.

## Basic chat request flow

```text
ChatPage.handleSubmit
  -> api/client.askStream POST /chat/ask
  -> chat router stores the user message
  -> chat_service.ask loads CHAT_* settings
     -> direct mode: bounded history -> OpenAILLM -> provider gateway
     -> auto/rag mode: lazy RAGAgent
          -> DenseRetriever queries Chroma
          -> configured reranker reduces/reorders chunks
          -> OpenAILLM generates a numbered-citation answer
          -> RAGAgent maps cited numbers back to chunks
     -> any unavailable local RAG path falls back to direct chat
  -> router serializes AgentEvent objects as SSE frames
  -> browser updates progress/final answer from the stream
  -> router stores the final assistant message after streaming completes
```

The same `SessionStore` singleton holds per-session history and a global bounded
trace deque used by the admin dashboard. State disappears when the API process
restarts and is not shared across multiple workers.

## Retrieval request flow

```text
RetrievalPage -> POST /retrieval/search
  -> retrieval router -> retrieval_service.search
  -> lazy cached ChromaStore + embedding function
  -> DenseRetriever.retrieve_with_timing
  -> query embedding -> Chroma query -> ranked chunks + timings
```

Core code implements dense, BM25, hybrid Reciprocal Rank Fusion, no-op reranking,
and cross-encoder reranking. The live retrieval API currently marks hybrid and
reranked search unavailable and accepts only `dense`.

## Offline data flows

### Crawl and index

```text
source feeds/pages
  -> sources.py discovery
  -> PlaywrightFetcher
  -> parser.py normalized article
  -> filters.py
  -> FilesystemStorage / HuggingFaceBucketStorage / CompositeStorage
  -> data/articles HTML
  -> NewsCleaner
  -> TextChunker selected by config
  -> ChromaStore + BM25Index
```

### Benchmark

```text
test set + matching variant manifest + config/index
  -> collect_benchmark_predictions.py
       writes run_manifest.json, attempts.jsonl, retrievals.jsonl, predictions.jsonl
  -> score_benchmark_predictions.py
       writes deterministic_scores.jsonl, report.json, report_summary.txt
  -> judge_benchmark_predictions.py (optional)
       writes/resumes judge_results.jsonl
  -> score again to merge judge metrics into report.json
```

The manifest fingerprint prevents incompatible configs, datasets, and indexes
from being mixed. JSONL checkpoints and latest-record-per-question logic make
expensive collection and judging resumable.

## Design patterns in use

- **Layered architecture:** routes delegate to services; services compose domain components.
- **Strategy + factory:** retrievers, rerankers, chunkers, embeddings, and crawler storage are selected behind stable contracts.
- **Adapters/facades:** `ChromaStore`, embedding functions, `OpenAILLM`, and `model_gateway.py` isolate external APIs.
- **Pipeline:** corpus ingestion, RAG answering, and evaluation are explicit staged transformations.
- **Lazy initialization/cache:** expensive local models, Chroma, RAGAgent, and remote clients load on demand.
- **Singleton repository:** `SessionStore` provides process-local history and traces.
- **Append-only checkpoints:** benchmark JSONL records plus immutable fingerprints support resume/retry.

## Current boundaries

- RAG is one retrieve -> rerank -> generate pass; multi-step agent orchestration is not implemented.
- `agents/orchestrator.py` and `tools/` are placeholders, not active architecture.
- Authentication is demo-only: hard-coded users, random client session IDs, no backend authorization.
- Admin crawler triggering is a stub returning `true`; the real crawler is CLI-only.
- Dashboard metrics read existing `reports/*/report.json` files rather than running evaluation live.
- API chat uses dense retrieval. Hybrid/BM25 are mainly used by offline benchmark code today.
- `configs/config.yaml` selects rerankers by `retrieval.reranker.type`; without that key, code falls back to the no-op reranker even if `enabled/provider` fields are present.

## Where to change common behavior

| Change | Start here |
| --- | --- |
| Add/modify an HTTP endpoint | `backend/newsqa_rag/api/routers/` then its service |
| Change chat/RAG fallback behavior | `backend/newsqa_rag/services/chat_service.py` |
| Change retrieval algorithms | `backend/newsqa_rag/retrieval/` and `services/retrieval_service.py` |
| Change answer prompting/provider | `backend/newsqa_rag/llm.py`, `model_gateway.py`, `.env` |
| Change chunking/embedding/indexing | `configs/config.yaml`, `ingestion/`, `indexing/` |
| Change benchmark contracts/metrics | `evaluation/`, then the three benchmark scripts |
| Change UI routes or API calls | `frontend/src/App.tsx`, `frontend/src/api/client.ts` |
