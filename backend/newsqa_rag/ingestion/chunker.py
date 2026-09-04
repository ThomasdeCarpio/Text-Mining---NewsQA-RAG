import os
import json
import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, encoding_name: str = "cl100k_base"):
        """
        Initializes the TextChunker
        
        Args:
            chunk_size: Maximum number of tokens per chunk. 
            chunk_overlap: Tokens to overlap between chunks. 
            encoding_name: The tokenizer to use. 
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def generate_article_id(self, url: str, filename: str) -> str:
        """
        Generates a unique, deterministic ID for an article based on its URL or filename.
        """

        base_string = url if url and url != "Unknown URL" else filename

        return hashlib.md5(base_string.encode('utf-8')).hexdigest()[:12]

    def chunk_article(self, article_data: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
        """
        Takes a single article's dictionary (from cleaner.py) and splits it into chunk dictionaries 
        """

        text = article_data.get("text", "")
        base_metadata = article_data.get("metadata", {})
        
        article_id = self.generate_article_id(base_metadata.get("url"), filename)
        
        raw_chunks = self.text_splitter.split_text(text)
        
        formatted_chunks = []
        
        for i, chunk_text in enumerate(raw_chunks):
            chunk_metadata = {
                "article_id": article_id,
                "chunk_index": i,
                "title": str(base_metadata.get("title", "")),
                "url": str(base_metadata.get("url", "")),
                "publish_date": str(base_metadata.get("publish_date", "")),
                "publisher": str(base_metadata.get("publisher", "")),
                "author": str(base_metadata.get("author", ""))
            }
            
            chunk_id = f"{article_id}_chunk_{i}"
            
            formatted_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
        return formatted_chunks

    def chunk_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """
        Loops through a directory of cleaned JSON files, chunks them all, 
        and returns a list ready for database ingestion.
        """

        all_chunks = []
        
        if not os.path.exists(input_dir):
            print(f"❌ Directory not found: {input_dir}")
            return all_chunks

        for filename in os.listdir(input_dir):
            if filename.endswith("_clean.json"):
                file_path = os.path.join(input_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        article_data = json.load(f)
                        
                    article_chunks = self.chunk_article(article_data, filename)
                    all_chunks.extend(article_chunks)
                    
                    print(f"✅ Chunked: {filename} -> Created {len(article_chunks)} chunks.")
                    
                except Exception as e:
                    print(f"❌ Error chunking {filename}: {e}")
                    
        print(f"\n🎯 Total chunks generated across all files: {len(all_chunks)}")

        return all_chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class HierarchicalChunker(TextChunker):
    """Two-level chunking: split into parents, then split each parent into children.

    The children are what gets emitted, embedded and retrieved - they are short,
    so a query matches a tight passage instead of being diluted across a long
    one. Each child records the parent it came from, so a later stage can widen
    a hit back out to the parent's full text for the generator to read.

    Both levels are produced by the same recursive splitter, so a child is still
    a verbatim substring of the article. That matters: chunk_char_ranges() locates
    every chunk by searching for its text in the source article, and evidence
    spans are mapped to chunks through those positions.

    config["chunking"]:
        chunk_size / chunk_overlap              the parent level
        child_chunk_size / child_chunk_overlap  the child level, and what is
                                                indexed. Default to half the
                                                parent, which roughly doubles
                                                the chunk count.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        child_chunk_size: int = None,
        child_chunk_overlap: int = None,
        encoding_name: str = "cl100k_base",
    ):
        super().__init__(chunk_size, chunk_overlap, encoding_name)
        self.child_chunk_size = child_chunk_size or max(64, chunk_size // 2)
        self.child_chunk_overlap = (
            child_chunk_overlap if child_chunk_overlap is not None
            else min(chunk_overlap, self.child_chunk_size // 4)
        )
        if self.child_chunk_size >= chunk_size:
            raise ValueError(
                f"child_chunk_size ({self.child_chunk_size}) must be smaller than "
                f"chunk_size ({chunk_size}), otherwise there is no hierarchy"
            )
        self.child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_article(self, article_data: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
        text = article_data.get("text", "")
        base_metadata = article_data.get("metadata", {})
        article_id = self.generate_article_id(base_metadata.get("url"), filename)

        formatted_chunks = []
        index = 0
        for parent_index, parent_text in enumerate(self.text_splitter.split_text(text)):
            children = self.child_splitter.split_text(parent_text) or [parent_text]
            for child_index, chunk_text in enumerate(children):
                formatted_chunks.append({
                    "id": f"{article_id}_chunk_{index}",
                    "text": chunk_text,
                    "metadata": {
                        "article_id": article_id,
                        "chunk_index": index,
                        # Chroma metadata values must stay scalar.
                        "parent_index": parent_index,
                        "parent_id": f"{article_id}_parent_{parent_index}",
                        "child_index": child_index,
                        "title": str(base_metadata.get("title", "")),
                        "url": str(base_metadata.get("url", "")),
                        "publish_date": str(base_metadata.get("publish_date", "")),
                        "publisher": str(base_metadata.get("publisher", "")),
                        "author": str(base_metadata.get("author", "")),
                    },
                })
                index += 1
        return formatted_chunks


def get_chunker(config: dict):
    """
    Factory. Reads config["chunking"].
    Supported strategies: "recursive", "hierarchical".
    """
    chunking_cfg = config.get("chunking", {})
    strategy = chunking_cfg.get("strategy", "recursive")

    if strategy == "recursive":
        return TextChunker(
            chunk_size=chunking_cfg.get("chunk_size", 500),
            chunk_overlap=chunking_cfg.get("chunk_overlap", 50),
        )

    if strategy == "hierarchical":
        return HierarchicalChunker(
            chunk_size=chunking_cfg.get("chunk_size", 512),
            chunk_overlap=chunking_cfg.get("chunk_overlap", 64),
            child_chunk_size=chunking_cfg.get("child_chunk_size"),
            child_chunk_overlap=chunking_cfg.get("child_chunk_overlap"),
        )

    raise ValueError(
        f"Unknown chunking strategy: '{strategy}'. "
        "Supported: 'recursive', 'hierarchical'."
    )


# ---------------------------------------------------------------------------
# JSONL persistence helpers
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str) -> None:
    """Write chunks to a JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {path}")


def load_chunks(path: str) -> list[dict]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _selfcheck() -> None:
    """The property everything downstream needs: a chunk is a verbatim substring.

    chunk_char_ranges() locates each chunk by searching for its text in the
    source article, and evidence spans are mapped to chunks through those
    positions. A chunk that is not findable breaks the whole dataset build, so
    the hierarchical splitter is checked against that here rather than only in
    a Kaggle run.
    """
    article = "\n\n".join(
        f"Paragraph {i}. " + "The quick brown fox jumps over the lazy dog. " * 12
        for i in range(8)
    )
    payload = {"text": article, "metadata": {"url": "", "title": "t", "publisher": "CNN"}}

    flat = get_chunker({"chunking": {"strategy": "recursive",
                                     "chunk_size": 512, "chunk_overlap": 64}})
    deep = get_chunker({"chunking": {"strategy": "hierarchical",
                                     "chunk_size": 512, "chunk_overlap": 64}})
    flat_chunks = flat.chunk_article(payload, filename="a.json")
    deep_chunks = deep.chunk_article(payload, filename="a.json")

    assert len(deep_chunks) > len(flat_chunks), "hierarchy should add children"
    ids = [c["id"] for c in deep_chunks]
    assert len(ids) == len(set(ids)), "chunk IDs must be unique"
    cursor = 0
    for position, chunk in enumerate(deep_chunks):
        assert chunk["text"] in article, f"chunk {position} is not a substring"
        assert chunk["metadata"]["chunk_index"] == position
        assert chunk["metadata"]["parent_index"] >= cursor
        cursor = chunk["metadata"]["parent_index"]
    assert cursor > 0, "one parent only - the test article is too short to nest"

    try:
        get_chunker({"chunking": {"strategy": "hierarchical", "chunk_size": 128,
                                  "child_chunk_size": 256}})
    except ValueError:
        pass
    else:
        raise AssertionError("a child larger than its parent must be rejected")

    print(f"selfcheck ok: {len(flat_chunks)} recursive -> {len(deep_chunks)} "
          f"hierarchical chunks, all substrings, IDs unique")


if __name__ == "__main__":
    _selfcheck()
