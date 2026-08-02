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

def get_chunker(config: dict) -> "TextChunker":
    """
    Factory. Reads config["chunking"].
    Currently supports strategy "recursive" only.
    """
    chunking_cfg = config.get("chunking", {})
    strategy = chunking_cfg.get("strategy", "recursive")

    if strategy == "recursive":
        return TextChunker(
            chunk_size=chunking_cfg.get("chunk_size", 500),
            chunk_overlap=chunking_cfg.get("chunk_overlap", 50),
        )

    raise ValueError(
        f"Unknown chunking strategy: '{strategy}'. Supported: 'recursive'."
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
