import os
import sys
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.ingestion.cleaner import NewsCleaner
from src.ingestion.chunker import TextChunker
from src.indexing.embeddings import get_embedding_function  
from src.indexing.chroma_store import ChromaStore

load_dotenv()

# --- CENTRAL CONFIGURATION ---
PIPELINE_CONFIG = {
    "embedding": {
        # --- FREE LOCAL MODEL (Hugging Face) ---
        "provider": "sentence-transformers",
        "model_name": "all-MiniLM-L6-v2"
        
        # --- TO TEST OPENAI LATER, UNCOMMENT BELOW & COMMENT ABOVE ---
        # "provider": "openai",
        # "model_name": "text-embedding-3-small",
        # "dimensions": 1536
    }
}

def run_ingestion_pipeline():
    """
    Orchestrates the full RAG ingestion process.
    """
    
    RAW_DATA_DIR = os.path.join(project_root, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(project_root, "data", "processed")
    CHROMA_DB_DIR = os.path.join(project_root, "data", "chroma_db")
    COLLECTION_NAME = "newsqa_cnn"

    print("🚀 Starting NewsRAG Ingestion Pipeline...")

    # --- STEP 1: CLEANING ---
    print(f"\n--- Phase 1: Cleaning HTML from {RAW_DATA_DIR} ---")
    cleaner = NewsCleaner(fallback_publisher="CNN")
    cleaner.process_directory(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    # --- STEP 2: CHUNKING ---
    print(f"\n--- Phase 2: Chunking Cleaned Articles from {PROCESSED_DATA_DIR} ---")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    final_chunks = chunker.chunk_directory(PROCESSED_DATA_DIR)

    if not final_chunks:
        print("❌ No chunks generated. Pipeline stopped.")
        return

    # --- STEP 3: INDEXING ---
    print(f"\n--- Phase 3: Embedding and Indexing into ChromaDB ---")
    
    embedding_provider = PIPELINE_CONFIG["embedding"]["provider"]
    print(f"🧠 Using Embedding Model: {PIPELINE_CONFIG['embedding']['model_name']} ({embedding_provider})")
    
    embedding_fn = get_embedding_function(PIPELINE_CONFIG)
    
    # Initialize the store
    store = ChromaStore(db_path=CHROMA_DB_DIR, embedding_function=embedding_fn)
    
    # Create the collection
    store.get_or_create_collection(name=COLLECTION_NAME)
    
    # Upsert the data (handles updates and prevents duplicates)
    result = store.upsert_chunks(collection_name=COLLECTION_NAME, chunks=final_chunks)

    # --- FINAL SUMMARY ---
    print("\n" + "="*40)
    print("✅ INGESTION COMPLETE")
    print(f"Total Chunks Processed: {result['total']}")
    print(f"Batches Upserted:      {result['batches']}")
    print(f"Database Location:     {CHROMA_DB_DIR}")
    print("="*40)

    # Verify count
    stats = store.get_collection_stats(COLLECTION_NAME)
    print(f"Current database count: {stats['count']} chunks.")

if __name__ == "__main__":
    # Check if OpenAI key is missing ONLY if the config is set to use OpenAI
    if PIPELINE_CONFIG["embedding"]["provider"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in .env file. Please add it or switch to 'sentence-transformers'.")
    else:
        run_ingestion_pipeline()