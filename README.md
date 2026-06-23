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


### Launch Streamlit UI

```bash
streamlit run ui/app.py
```

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
|   |-- ingestion/            # Data ingestion pipeline
|   |   |-- loader.py         # Load raw files (HTML/TXT/MD) into Document objects
|   |   |-- cleaner.py        # Normalize and clean raw text
|   |   |__ chunker.py        # Split documents into chunks (configurable strategy)
|   |
|   |-- indexing/             # Vector store and index management
|   |   |-- embeddings.py     # Embedding function (OpenAI / Sentence-Transformers)
|   |   |-- chroma_store.py   # ChromaDB collection CRUD operations
|   |   |__ bm25_index.py     # BM25 sparse index for hybrid retrieval
|   |
|   |-- retrieval/            # Retrieval strategies
|   |   |-- dense.py          # Dense vector retrieval (ChromaDB cosine search)
|   |   |-- hybrid.py         # Hybrid retrieval (dense + BM25 fusion)
|   |   |__ reranker.py       # Cohere/cross-encoder reranking
|   |
|   |-- agents/               # LangGraph agentic orchestration
|   |   |-- rag_agent.py      # RAG agent with tool-calling capabilities
|   |   |__ orchestrator.py   # Multi-step agent workflow (LangGraph)
|   |
|   |-- evaluation/           # Evaluation with Ragas
|   |   |__ metrics.py        # Metric computation (faithfulness, relevancy, etc.)
|   |
|   |-- tools/                # LangChain tools for agent use
|   |   |-- retrieval_tools.py  # Search/retrieve tool wrappers
|   |   |__ ingestion_tools.py  # On-the-fly ingestion tool (for live crawl)
|   |
|   |__ llm.py                # LLM client initialization (OpenAI)
|
|-- ui/
|   |__ app.py                # Streamlit web interface
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
