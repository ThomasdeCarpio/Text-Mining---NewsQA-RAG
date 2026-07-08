# Text-Mining---NewsQA-RAG

## Project Overview

Goal: Build an **Agentic Retrieval-Augmented Generation (RAG)** system for News QA. Unlike standard RAG, our system uses an LLM Agent to autonomously trigger search tools and synthesize multiple news sources. The primary academic focus is rigorously evaluating Retrieval and Generation metrics.

**Datasets:**
- **NewsQA** — single-document fact retrieval
- **Multi-News** — multi-document synthesis and comparison

**Tech Stack:** Python, LangChain, ChromaDB, OpenAI API, Ragas (evaluation), Streamlit (UI).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Prepare config

Edit `configs/config.yaml` to set embedding model, chunking strategy, and database parameters. See [Config Reference](#config-reference) below.

---

## Usage

### Build ChromaDB Collection (End-to-End Pipeline)

The main pipeline script. Reads raw article files, cleans, chunks, embeds, and indexes them into a ChromaDB collection in one pass.

**Input folder structure:**

```
data/articles/
    |- 001.html
    |- 002.txt
    |- 003.md
    |_ <article_id>.<ext>
```

Supported formats: `.html`, `.txt`, `.md`

**Run:**

```bash
python scripts/build_chroma_collection.py \
    --source data/articles/ \
    --db-path database/ \
    --collection basic_collection \
    --config configs/config.yaml
```

| Argument | Required | Description |
|---|---|---|
| `--source` | Yes | Path to folder containing raw article files |
| `--db-path` | Yes | Path to ChromaDB persistent storage |
| `--collection` | Yes | Name of the Chroma collection to create/update |
| `--config` | No | Path to config file (default: `configs/config.yaml`) |


### Run Evaluation Benchmark

Evaluate the RAG pipeline against ground-truth QA pairs using Ragas metrics.

```bash
python scripts/run_benchmark.py \
    --db-path database/ \
    --collection basic_collection \
    --test-set data/test_qa.json \
    --output results/benchmark_results.json \
    --config configs/config.yaml
```

| Argument | Required | Description |
|---|---|---|
| `--db-path` | Yes | Path to ChromaDB storage |
| `--collection` | Yes | Collection name to query |
| `--test-set` | Yes | Path to test QA pairs (JSON) |
| `--output` | No | Path to save results (default: stdout) |
| `--config` | No | Config file path |


### Query CLI (Quick Test)

Interactive CLI to test the RAG system with ad-hoc questions.

```bash
python scripts/query.py \
    --db-path database/ \
    --collection basic_collection \
    --question "What happened in the 2024 US election?"
```

| Argument | Required | Description |
|---|---|---|
| `--db-path` | Yes | Path to ChromaDB storage |
| `--collection` | Yes | Collection name |
| `--question` | No | Single question (omit for interactive mode) |


### Inspect Collection

Utility to inspect and debug a ChromaDB collection.

```bash
python scripts/inspect_collection.py \
    --db-path database/ \
    --collection basic_collection \
    --action stats
```

| Argument | Required | Description |
|---|---|---|
| `--db-path` | Yes | Path to ChromaDB storage |
| `--collection` | Yes | Collection name |
| `--action` | No | `stats` (default), `sample`, `search`, `delete` |


### Run the API + UI

The app is split into a FastAPI backend (`api/`) and a React frontend (`ui/`).
Business logic lives behind `src/services/` — currently returning mock data;
see [Roadmap](#roadmap--remaining-work) for what still needs real
implementations.

```bash
# Terminal 1 — backend (http://localhost:8000)
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend (http://localhost:5173)
cd ui
npm install
npm run dev
```

Mock login credentials (see `src/services/auth_service.py`):
| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | admin (Evaluation Dashboard + News Chat) |
| `analyst` | `pass123` | user (News Chat only) |

---

## Config Reference

All pipeline behavior is controlled through `configs/config.yaml`. See the file for full documentation of available parameters including:

- **Embedding**: model provider (OpenAI / Sentence-Transformers), model name, dimensions
- **Chunking**: strategy (recursive / semantic / sentence), chunk size, overlap
- **Database**: HNSW parameters, collection settings
- **LLM**: model name, temperature, max tokens
- **Retrieval**: top-k, reranker settings, hybrid search weights

---

## Project Structure

```
Text-Mining---NewsQA-RAG/
|
|-- configs/
|   |-- config.yaml          # Main configuration (embedding, chunking, DB, LLM)
|   |__ setting.py            # Python-level settings, path constants
|
|-- data/                     # (gitignored) Raw data and test sets
|   |-- articles/             # Raw article files for ingestion
|   |__ test_qa.json          # Ground-truth QA pairs for evaluation
|
|-- database/                 # (gitignored) ChromaDB persistent storage
|
|-- docs/
|   |-- database.md           # Database contract: metadata schema, HNSW config, ID format
|   |-- ingestion_guide.md    # Implementation guide for src/ingestion/
|   |__ indexing_guide.md     # Implementation guide for src/indexing/
|
|-- scripts/
|   |-- build_chroma_collection.py  # End-to-end pipeline: load -> clean -> chunk -> embed -> index
|   |-- run_benchmark.py            # Evaluate RAG with Ragas metrics
|   |-- query.py                    # CLI for ad-hoc RAG queries
|   |__ inspect_collection.py       # Utility: inspect/debug ChromaDB collections
|
|-- src/
|   |-- ingestion/            # Data ingestion pipeline (implemented)
|   |   |-- loader.py         # Load raw files (HTML/TXT/MD) into Document objects
|   |   |-- cleaner.py        # Normalize and clean raw text
|   |   |__ chunker.py        # Split documents into chunks (configurable strategy)
|   |
|   |-- indexing/             # Vector store and index management
|   |   |-- embeddings.py     # Embedding function (OpenAI / Sentence-Transformers) - implemented
|   |   |-- chroma_store.py   # ChromaDB collection CRUD operations - implemented
|   |   |__ bm25_index.py     # BM25 sparse index for hybrid retrieval - TODO
|   |
|   |-- retrieval/            # Retrieval strategies - all TODO
|   |   |-- dense.py          # Dense vector retrieval (ChromaDB cosine search)
|   |   |-- hybrid.py         # Hybrid retrieval (dense + BM25 fusion)
|   |   |__ reranker.py       # Cohere/cross-encoder reranking
|   |
|   |-- agents/               # LangGraph agentic orchestration - TODO
|   |   |-- rag_agent.py      # RAG agent with tool-calling capabilities
|   |   |__ orchestrator.py   # Multi-step agent workflow (LangGraph)
|   |
|   |-- evaluation/           # Evaluation with Ragas - TODO
|   |   |__ metrics.py        # Metric computation (faithfulness, relevancy, etc.)
|   |
|   |-- tools/                # LangChain tools for agent use - TODO
|   |   |-- retrieval_tools.py  # Search/retrieve tool wrappers
|   |   |__ ingestion_tools.py  # On-the-fly ingestion tool (for live crawl)
|   |
|   |-- services/             # Business logic behind the API (mock data for now)
|   |   |-- types.py          # Shared dataclasses (AgentEvent, Citation, ChatMessage, User)
|   |   |-- session_store.py  # In-memory chat history + trace log
|   |   |-- auth_service.py   # Hardcoded mock login
|   |   |-- chat_service.py   # Mock ReAct-style agent event generator
|   |   |__ eval_service.py   # Mock dashboard metrics/comparison/failure cases
|   |
|   |__ llm.py                # LLM client initialization (OpenAI) - TODO
|
|-- api/                      # FastAPI backend (serves src/services/* over HTTP)
|   |-- main.py                # App entrypoint, CORS, router registration
|   |-- schemas.py             # Pydantic request/response models (wire contract)
|   |__ routers/
|       |-- auth.py            # POST /auth/login
|       |-- chat.py            # POST /chat/ask (SSE stream), GET/POST history & clear
|       |__ admin.py           # GET /admin/metrics|search-comparison|failure-cases|pipeline-logs, POST /admin/trigger-crawler
|
|-- ui/                       # React + Vite + TypeScript frontend
|   |-- src/
|   |   |-- api/               # fetch-based client mirroring api/schemas.py
|   |   |-- context/           # AuthContext (session persisted to localStorage)
|   |   |-- pages/             # LoginPage, ChatPage, DashboardPage
|   |   |__ components/        # Sidebar, ChatBubble, CitationList, MetricCard
|   |__ package.json
|
|-- .env.example              # Environment variable template
|-- .gitignore
|-- LICENSE
|-- README.md
|__ requirements.txt
```

---

## Development Docs

| Document | Description |
|---|---|
| [docs/database.md](docs/database.md) | Database contract: metadata schema, ID format, HNSW config, embedding model spec |
| [docs/ingestion_guide.md](docs/ingestion_guide.md) | Implementation guide for `src/ingestion/` (loader, cleaner, chunker) |
| [docs/indexing_guide.md](docs/indexing_guide.md) | Implementation guide for `src/indexing/` (embeddings, chroma_store, bm25) |
| [docs/ui.md](docs/ui.md) | UI/UX spec: user roles, required screens, evaluation dashboard |

---

## Roadmap / Remaining Work

The API (`api/`) and UI (`ui/`) are wired end-to-end today, but every endpoint
in `src/services/` returns **mock data**. Nothing yet touches ChromaDB, an
LLM, or Ragas. Below is what's left, grouped by milestone — each milestone
should leave the app fully runnable end-to-end, just with more of it real.

### Milestone 1 — Single-source RAG (replaces the chat mock with a real answer)
- [ ] `src/llm.py` — OpenAI client init, mirroring the `get_embedding_function(config)`
      factory pattern already used in `src/indexing/embeddings.py`.
- [ ] `src/retrieval/dense.py` — dense-only retrieval against `ChromaStore`
      (`src/indexing/chroma_store.py`, already implemented).
- [ ] `src/tools/retrieval_tools.py` — LangChain tool wrapper around dense
      retrieval.
- [ ] `src/agents/rag_agent.py` — single-tool-call agent (no multi-step loop
      yet): question → retrieve → generate answer + citations.
- [ ] Wire `scripts/query.py` (currently docstring-only) against this.
- [ ] Replace `src/services/chat_service.py`'s mock generator internals with
      a call into the real agent — the `AsyncIterator[AgentEvent]` shape and
      the `/chat/ask` SSE endpoint do not need to change.
- [ ] Validate against **NewsQA** (single-document fact-finding), per
      `docs/ui.md`'s "CORE FUNCTIONALITY" section.

### Milestone 2 — Hybrid retrieval + reranking
- [ ] `src/indexing/bm25_index.py` — BM25 sparse index.
- [ ] `src/retrieval/hybrid.py` — dense + BM25 fusion (weights already defined
      in `configs/config.yaml` under `retrieval.hybrid`).
- [ ] `src/retrieval/reranker.py` — Cohere or cross-encoder reranking
      (`retrieval.reranker` config already defined).
- [ ] Swap `retrieval_tools.py` to use hybrid+rerank behind the same tool
      interface.

### Milestone 3 — Full agentic orchestration (multi-source synthesis)
- [ ] `src/agents/orchestrator.py` — LangGraph multi-step tool-calling loop
      (the README already names this as the intended framework).
- [ ] `src/tools/ingestion_tools.py` — on-the-fly/live crawl tool, wired to
      the Admin "Manual Crawler Trigger" (`eval_service.trigger_crawler()` is
      currently a no-op mock).
- [ ] Multi-source synthesis + comparison queries (per `docs/ui.md`'s
      "ADVANCED FUNCTIONALITY" section) — needs the Multi-News dataset
      modification mentioned there; do this only after Milestone 1 is solid.
- [ ] `chat_service.ask` keeps streaming `AgentEvent`s — real tool calls just
      replace the mock `thought`/`tool_call`/`tool_result` steps.

### Milestone 4 — Real evaluation
- [ ] `src/evaluation/metrics.py` — Ragas metrics (`faithfulness`,
      `answer_relevancy`, `context_precision`, `context_recall`; metric list
      and judge model already defined in `configs/config.yaml`).
- [ ] Implement `scripts/run_benchmark.py` (currently docstring-only) end to
      end: load `data/test_qa.json` → retrieve → generate → score → write
      results JSON.
- [ ] Replace `src/services/eval_service.py`'s hardcoded numbers with real
      results (read the benchmark output instead of returning fabricated
      dicts) — the Admin Dashboard UI does not need to change.
- [ ] Failure Analysis table: derive real low-scoring cases from benchmark
      output instead of the two hardcoded rows.

### Milestone 5 — Productionization / polish (do only once 1–4 are solid)
- [ ] `configs/setting.py` — settings loader unifying `config.yaml` + `.env`
      (currently empty; nothing parses `config.yaml` into Python yet).
- [ ] Real user store + token-based auth (JWT or session cookie) instead of
      the hardcoded dict in `auth_service.py` — only worth it if this goes
      beyond local/demo use.
- [ ] Persistent session store (Redis or DB) instead of the in-memory
      `SessionStore` singleton — needed if the API ever runs multiple workers
      or needs to survive restarts.
- [ ] Multi-conversation history switcher in the UI (currently one running
      session per login, called out as a simplification when the UI was
      scaffolded).
- [ ] Error/loading states and retries in `ui/src/api/client.ts` (currently
      no handling beyond throwing on non-2xx).
- [ ] Basic tests: unit tests for retrieval/evaluation once real, and an
      API-level smoke test for `api/routers/*`.
