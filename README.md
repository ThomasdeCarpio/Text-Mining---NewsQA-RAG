# Text-Mining---NewsQA-RAG

## Project Overview

Goal: Build an **Agentic Retrieval-Augmented Generation (RAG)** system for News QA. Unlike standard RAG, our system uses an LLM Agent to autonomously trigger search tools and synthesize multiple news sources. The primary academic focus is rigorously evaluating Retrieval and Generation metrics.

**Datasets:**
- **NewsQA** — single-document fact retrieval
- **Multi-News** — multi-document synthesis and comparison

**Tech Stack:** Python, LangChain/LangGraph, ChromaDB, OpenAI API, Ragas (evaluation), FastAPI (backend), React + Vite + TypeScript (UI).

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

### Run Ingestion (working today)

`scripts/ingest.py` is the pipeline that actually works right now — it reads
raw HTML from `data/raw/`, cleans it (`src/ingestion/cleaner.py`), chunks it
(`src/ingestion/chunker.py`, 500-token chunks via tiktoken, 50-token overlap),
embeds with `sentence-transformers/all-MiniLM-L6-v2` (local, no API key
needed), and upserts into a ChromaDB collection named `newsqa_cnn` at
`data/chroma_db/`.

```bash
python scripts/ingest.py
```

Notes:
- On Windows, if you see `UnicodeEncodeError` printing the emoji log lines,
  run with `PYTHONIOENCODING=utf-8 python scripts/ingest.py` instead (Windows
  console defaults to cp1252).
- Some CNN articles in the raw dataset are cp1252-encoded, not UTF-8;
  `NewsCleaner.clean_file` already falls back to cp1252 automatically on a
  decode failure — a handful of "❌ Error processing ..." lines for other
  reasons is still worth checking, but encoding errors are handled.
- Safe to re-run: `ChromaStore.upsert_chunks` upserts by ID, so re-running
  after adding new raw files won't duplicate existing chunks.
- `PIPELINE_CONFIG` inside `scripts/ingest.py` is currently hardcoded
  (embedding provider/model, paths) rather than reading `configs/config.yaml`
  — see Roadmap Milestone 5.

`scripts/build_chroma_collection.py` (below) documents an intended
config-driven alternative to this same pipeline, but is currently
docstring-only (not implemented) — use `scripts/ingest.py` for now.

### Build ChromaDB Collection (config-driven, not yet implemented)

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

The canonical evaluation workflow builds one shared 11,064-article retrieval
collection. The original benchmark is usable while question review continues:

```bash
# Select 200 evaluation articles and all 10,864 eligible train distractors
python scripts/prepare_evaluation_dataset.py stage1 --selection-only

# Reuse the completed review for the unchanged 200-article evaluation sample
python scripts/prepare_evaluation_dataset.py migrate-review

# Build the original benchmark and the shared Chroma/BM25 indexes immediately
python scripts/prepare_evaluation_dataset.py build-baseline

# Validate and finalize the reviewed question variants
python scripts/prepare_evaluation_dataset.py review-status
python scripts/prepare_evaluation_dataset.py finalize
```

See [`docs/evaluation_dataset.md`](docs/evaluation_dataset.md) for artifact
schemas, review decisions, failure conditions, and manifest-verified benchmark
commands. See
[`docs/final_evaluation_output.md`](docs/final_evaluation_output.md) for the
final `newsqa_200_11064` file inventory, row schemas, counts, and scoring usage.
See
[`docs/evaluation_dataset_handoff.md`](docs/evaluation_dataset_handoff.md) for
the required scripts, reproducible build commands, complete artifact inventory,
and teammate sharing instructions.
The older mini builder below remains available for exploratory runs, but it
does not provide the human approval workflow.

Score the finalized RAG pipeline with durable collection, deterministic
scoring, and LLM judging as separate resumable stages:

```bash
.venv/bin/python scripts/collect_benchmark_predictions.py \
  --retriever hybrid --reranker noop --retrieval-only \
  --testset data/evaluation/newsqa_200_11064/final/testset_reviewed_original.jsonl \
  --variant-manifest evaluation/manifests/newsqa_200_11064.variant.json \
  --run-dir reports/benchmarks/original_hybrid_noop

.venv/bin/python scripts/score_benchmark_predictions.py \
  --run-dir reports/benchmarks/original_hybrid_noop
```

See [`docs/benchmarking.md`](docs/benchmarking.md) for cross-encoder,
generation, retry/resume, LLM judge, and original-versus-resolved commands.
Completed reports can be compared in `notebooks/04_final_benchmark_analysis.ipynb`.
The complete workflow can also be run interactively from
`notebooks/05_run_final_benchmark.ipynb`.


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

**Prerequisites:**
1. `pip install -r requirements.txt` and `cd ui && npm install`.
2. `cp .env.example .env` and configure either the OpenAI-compatible gateway
   variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CHAT_MODEL`) or
   `DEEPSEEK_API_KEY`.
3. Optional RAG: run `python scripts/ingest.py` once to create the `newsqa_cnn`
   collection at `data/chroma_db/`. Without it, chat falls back to the model
   gateway and the Retrieval Playground reports that no collection is available.

```bash
# Terminal 1 — backend (http://localhost:8000)
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend (http://localhost:5173)
cd ui
npm run dev
```

Open `http://localhost:5173` and log in (see `src/services/auth_service.py`
for the hardcoded credentials):
| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | admin — News Chat + Evaluation Desk + Retrieval Playground |
| `analyst` | `pass123` | user — News Chat only |

**What's real vs mock right now:**
- **News Chat** is real and calls the OpenAI-compatible gateway configured by
  `OPENAI_BASE_URL` and `CHAT_MODEL`, or DeepSeek when `DEEPSEEK_API_KEY` is
  set. `CHAT_MODE=auto` runs the real RAG pipeline when the local collection
  is available and falls back to direct multi-turn chat otherwise.
- **Retrieval Playground** (admin, `/retrieval`) is real — dense vector search
  with a per-request latency breakdown (embed vs ChromaDB query time).
- **Evaluation Desk** (admin) reads real reports from `reports/` once you run a
  benchmark — see [Run Evaluation Benchmark](#run-evaluation-benchmark) and
  `notebooks/02_evaluation.ipynb`. Empty until the first report exists.

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
|-- data/                     # (gitignored) Raw data, processed articles, and test sets
|   |-- raw/                  # Raw HTML articles for ingestion (scripts/ingest.py input)
|   |-- processed/            # Cleaned text + metadata JSON (scripts/ingest.py intermediate output)
|   |-- chroma_db/            # ChromaDB persistent storage (scripts/ingest.py output) - "newsqa_cnn" collection
|   |__ test_qa.json          # Ground-truth QA pairs for evaluation
|
|-- database/                 # (gitignored) Alternate ChromaDB storage path used by
|                              #   scripts/build_chroma_collection.py once implemented
|
|-- docs/
|   |-- database.md           # Database contract: metadata schema, HNSW config, ID format
|   |-- ingestion_guide.md    # Implementation guide for src/ingestion/
|   |__ indexing_guide.md     # Implementation guide for src/indexing/
|
|-- scripts/
|   |-- ingest.py                   # Working ingestion pipeline (clean -> chunk -> embed -> index) - USE THIS
|   |-- build_chroma_collection.py  # Intended config-driven version of the above - TODO (docstring only)
|   |-- run_benchmark.py            # Evaluate RAG with Ragas metrics - TODO (docstring only)
|   |-- query.py                    # CLI for ad-hoc RAG queries - TODO (docstring only)
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
|   |-- retrieval/            # Retrieval strategies
|   |   |-- dense.py          # Dense vector retrieval - implemented (embeds query,
|   |   |                     #   then queries ChromaDB separately so callers can time each phase)
|   |   |-- hybrid.py         # Hybrid retrieval (dense + BM25 fusion) - TODO
|   |   |__ reranker.py       # Cohere/cross-encoder reranking - TODO
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
|   |-- services/             # Business logic behind the API
|   |   |-- types.py             # Shared dataclasses (AgentEvent, Citation, ChatMessage, User)
|   |   |-- session_store.py     # In-memory chat history + trace log
|   |   |-- auth_service.py      # Hardcoded mock login
|   |   |-- chat_service.py      # Mock ReAct-style agent event generator
|   |   |-- eval_service.py      # Mock dashboard metrics/comparison/failure cases
|   |   |__ retrieval_service.py # REAL - dense search against the ChromaDB collection
|   |                            #   scripts/ingest.py produces; hybrid/reranked raise
|   |                            #   NotImplementedError until Milestone 2
|   |
|   |__ llm.py                # LLM client initialization (OpenAI) - TODO
|
|-- api/                      # FastAPI backend (serves src/services/* over HTTP)
|   |-- main.py                # App entrypoint, CORS, router registration
|   |-- schemas.py             # Pydantic request/response models (wire contract)
|   |__ routers/
|       |-- auth.py            # POST /auth/login
|       |-- chat.py            # POST /chat/ask (SSE stream), GET/POST history & clear
|       |-- admin.py           # GET /admin/metrics|search-comparison|failure-cases|pipeline-logs, POST /admin/trigger-crawler
|       |__ retrieval.py       # GET /retrieval/algorithms|stats, POST /retrieval/search (REAL)
|
|-- ui/                       # React + Vite + TypeScript frontend (retro/newspaper theme)
|   |-- src/
|   |   |-- api/               # fetch-based client mirroring api/schemas.py
|   |   |-- context/           # AuthContext (session persisted to localStorage)
|   |   |-- pages/             # LoginPage, ChatPage, DashboardPage, RetrievalPage
|   |   |__ components/        # Sidebar, ChatBubble, CitationList, MetricCard, RetrievalResultCard
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

**New to this project? Start with [docs/explainer/](docs/explainer/README.md)**
— a plain-language (Vietnamese) walkthrough of what the system does, how the
pieces connect, a file-by-file map, and what's missing and why. No code-level
detail — for that, see the guides below. Prefer reading it as a styled page?
Open `docs/explainer/index.html` in a browser (works offline, fonts/styles
are embedded) instead of the raw `.md` files.

| Document | Description |
|---|---|
| [docs/database.md](docs/database.md) | Database contract: metadata schema, ID format, HNSW config, embedding model spec |
| [docs/ingestion_guide.md](docs/ingestion_guide.md) | Implementation guide for `src/ingestion/` (loader, cleaner, chunker) |
| [docs/indexing_guide.md](docs/indexing_guide.md) | Implementation guide for `src/indexing/` (embeddings, chroma_store, bm25) |
| [docs/model_gateway.md](docs/model_gateway.md) | XAH gateway setup for NewsQA, Codex, Claude Code, OpenCode, and embeddings |
| [docs/ui.md](docs/ui.md) | UI/UX spec: user roles, required screens, evaluation dashboard |

---

## Roadmap / Remaining Work

Ingestion is real (`scripts/ingest.py` → ChromaDB, 413 chunks from the sample
CNN/NewsQA data as of this writing) and dense retrieval is real (Retrieval
Playground). Chat and the Evaluation Desk still return **mock data** from
`src/services/chat_service.py` / `eval_service.py`. Below is what's left,
grouped by milestone — each milestone should leave the app fully runnable
end-to-end, just with more of it real.

### Milestone 1 — Single-source RAG (replaces the chat mock with a real answer)
- [x] `src/retrieval/dense.py` — dense-only retrieval against `ChromaStore`,
      embeds the query and queries ChromaDB as two separate timed steps.
- [x] `src/services/retrieval_service.py` + `api/routers/retrieval.py` +
      the `/retrieval` Retrieval Playground page — lets you test retrieval
      quality/latency directly against real ingested data, independent of
      any LLM. **Not yet wired into chat** — this is a standalone debugging
      surface, not (yet) something `chat_service.ask` calls into.
- [ ] `src/llm.py` — OpenAI client init, mirroring the `get_embedding_function(config)`
      factory pattern already used in `src/indexing/embeddings.py`.
- [ ] `src/tools/retrieval_tools.py` — LangChain tool wrapper around
      `retrieval_service`/`dense_search`, for agent tool-calling (distinct
      from the direct service-call path the Retrieval Playground uses).
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
- [ ] `src/services/retrieval_service.py`'s `search()` already dispatches on
      an `algorithm` string (`"dense"` implemented, `"hybrid"`/`"reranked"`
      raise `NotImplementedError`) and `list_algorithms()` marks them
      `available: false` — implement the two branches there and flip
      `available` to `true`; the Retrieval Playground UI already renders
      whatever `list_algorithms()` returns, no frontend change needed.
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
      (currently empty; nothing parses `config.yaml` into Python yet). Once
      this exists, remove the duplicated embedding/path config hardcoded in
      both `scripts/ingest.py`'s `PIPELINE_CONFIG` and
      `src/services/retrieval_service.py` — they must currently be kept in
      sync by hand.
- [ ] Consolidate `scripts/ingest.py` (works, hardcoded config) and
      `scripts/build_chroma_collection.py` (intended config-driven CLI,
      docstring-only) into one script.
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
