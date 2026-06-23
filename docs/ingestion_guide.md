# Ingestion Pipeline Implementation Guide

**Module:** `src/ingestion/`

This guide specifies what each file in the ingestion module must implement, its public API, input/output contracts, and how the files connect to each other.

The ingestion pipeline transforms raw article files into chunked, metadata-enriched objects ready for indexing.

---

## Pipeline Flow

```
loader.py → cleaner.py → chunker.py
```

Each stage receives output from the previous stage. Scripts in `scripts/` orchestrate this pipeline — the ingestion module itself only exposes functions, not CLI entry points.

---

## 1. `loader.py` — File Loader

**Purpose:** Read raw article files from a directory and convert them into structured `Document` objects.

### Supported Input Formats

| Extension | Parsing Method |
|---|---|
| `.html` | Parse with BeautifulSoup, extract `<article>` or `<body>` text |
| `.txt` | Read as plain text (UTF-8) |
| `.md` | Read as plain text (treat markdown as raw text for now) |

### Public API

```python
def load_articles(source_dir: str) -> list[Document]:
    """
    Scan source_dir for article files, parse each into a Document.

    Args:
        source_dir: Path to folder containing article files.

    Returns:
        List of Document objects, one per file.
    """
```

### Document Object

```python
from dataclasses import dataclass

@dataclass
class Document:
    article_id: str       # Filename without extension (e.g., "001")
    raw_text: str         # Full text content of the article
    filename: str         # Original filename (e.g., "001.html")
    metadata: dict        # Extracted metadata (title, source, date, etc.)
```

### Metadata Extraction

The loader should attempt to extract metadata from file content:

- **HTML files:** Extract `<title>`, `<meta name="author">`, `<meta name="date">`, `<meta name="source">` if available.
- **TXT/MD files:** If the first line looks like a title (short, no period), use it as `title`. Other fields default to `""`.
- Any field that cannot be extracted → set to `""`.

### Error Handling

- Skip files that cannot be read (log warning, continue).
- Skip empty files.
- Skip files with unrecognized extensions (log warning).

---

## 2. `cleaner.py` — Text Cleaner

**Purpose:** Normalize and clean raw text from loaded documents. Ensure consistent formatting before chunking.

### Public API

```python
def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Apply cleaning pipeline to each document's raw_text.
    Returns new Document objects with cleaned text (does not mutate input).
    """

def clean_text(text: str) -> str:
    """
    Clean a single text string. Used internally and can be called standalone.
    """
```

### Cleaning Steps (in order)

1. **Strip HTML tags** — Remove any residual HTML tags (in case loader left some).
2. **Decode HTML entities** — `&amp;` → `&`, `&lt;` → `<`, etc.
3. **Normalize whitespace** — Collapse multiple spaces/tabs into single space. Collapse 3+ newlines into 2.
4. **Normalize Unicode** — NFC normalization. Replace smart quotes with ASCII quotes.
5. **Remove boilerplate** — Strip common patterns: "Subscribe to our newsletter", "Advertisement", cookie banners, etc. (configurable list in config if needed).
6. **Strip leading/trailing whitespace** per line and overall.

### Contract

- Input: `Document` with `raw_text` field populated.
- Output: `Document` with `raw_text` replaced by cleaned version. All other fields unchanged.
- The cleaner must **not** remove meaningful content (paragraphs, sentences).
- The cleaner must **not** change metadata.

---

## 3. `chunker.py` — Document Chunker

**Purpose:** Split cleaned documents into chunks suitable for embedding and retrieval. The chunking strategy is configurable.

### Public API

```python
def chunk_documents(
    documents: list[Document],
    config: dict
) -> list[dict]:
    """
    Split each document into chunks according to the strategy in config.

    Args:
        documents: Cleaned Document objects.
        config: Chunking config (from config.yaml["chunking"]).

    Returns:
        List of chunk dicts, each containing:
        {
            "id": "001_chunk_003",
            "text": "chunk content...",
            "metadata": { NewsChunkMetadata }
        }
    """

def get_chunker(config: dict):
    """
    Factory function. Returns the appropriate chunker based on config["strategy"].
    """
```

### Strategy Implementations

Each strategy must be a callable that takes `(text: str, config: dict) -> list[str]` and returns a list of chunk text strings.

#### `recursive` (default)
- Use LangChain's `RecursiveCharacterTextSplitter`.
- Separators: `["\n\n", "\n", ". ", " ", ""]`
- Respect `chunk_size` and `chunk_overlap` from config.

#### `semantic`
- Use embedding similarity between consecutive sentences to find natural breakpoints.
- Group sentences where similarity is above a threshold into the same chunk.
- Falls back to `recursive` if chunks exceed `chunk_size`.

#### `sentence`
- Split text into sentences (use `nltk` or regex-based sentence tokenizer).
- Group `sentences_per_chunk` consecutive sentences per chunk.
- Overlap by `sentence_overlap` sentences.

### ID Generation

```python
def make_chunk_id(article_id: str, chunk_index: int) -> str:
    return f"{article_id}_chunk_{chunk_index:03d}"
```

### Output Contract

Each chunk dict must match the schema defined in `docs/database.md`:

```python
{
    "id": str,           # make_chunk_id(article_id, index)
    "text": str,         # The chunk content
    "metadata": {
        "source": str,
        "article_id": str,
        "title": str,
        "url": str,
        "published_date": str,
        "author": str,
        "category": str,
        "chunk_index": int,
        "total_chunks": int,
    }
}
```

---

## Config Dependencies

The ingestion module reads from `configs/config.yaml`:

```yaml
chunking:
  strategy: "recursive"
  chunk_size: 512
  chunk_overlap: 64
  sentences_per_chunk: 8     # sentence strategy only
  sentence_overlap: 2        # sentence strategy only
```

---

## Testing Checklist

When implementing, verify:

- [ ] `loader.py` correctly parses HTML, TXT, MD files
- [ ] `loader.py` extracts metadata from HTML `<meta>` tags
- [ ] `loader.py` skips unreadable/empty/unknown files with warning
- [ ] `cleaner.py` removes HTML tags but preserves paragraph structure
- [ ] `cleaner.py` does not destroy meaningful content
- [ ] `chunker.py` produces correct IDs following the format `<article_id>_chunk_<NNN>`
- [ ] `chunker.py` respects `chunk_size` and `chunk_overlap` config
- [ ] `chunker.py` populates all metadata fields (empty string for missing)
- [ ] Switching strategy via config produces valid output for all 3 strategies
- [ ] End-to-end: raw HTML file → list of chunk dicts with correct schema
