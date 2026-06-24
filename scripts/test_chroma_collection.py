"""
Test script for ChromaStore and embedding functions.
Uses SentenceTransformer (free, local) so no API key is needed.

Run: python scripts/test_chroma_collection.py
"""

import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.indexing.embeddings import (
    SentenceTransformerEmbeddingFunction,
    get_embedding_function,
)
from src.indexing.chroma_store import ChromaStore

TEST_DB_PATH = os.path.join("..", "database", "chroma_test_db")
COLLECTION_NAME = "test_collection"
HNSW_CONFIG = {
    "space": "cosine",
    "ef_construction": 100,
    "max_neighbors": 16,
    "ef_search": 50,
}

SAMPLE_CHUNKS = [
    {
        "id": "001_chunk_000",
        "text": "The Federal Reserve announced a 0.25% interest rate hike on Wednesday, marking the third consecutive increase this year.",
        "metadata": {
            "source": "CNN",
            "article_id": "001",
            "title": "Fed Raises Rates Again",
            "url": "https://example.com/001",
            "published_date": "2024-03-15T10:00:00Z",
            "author": "John Smith",
            "category": "Business",
            "chunk_index": 0,
            "total_chunks": 3,
        },
    },
    {
        "id": "001_chunk_001",
        "text": "Economists predict the rate hike will slow down housing market activity and increase mortgage rates for consumers.",
        "metadata": {
            "source": "CNN",
            "article_id": "001",
            "title": "Fed Raises Rates Again",
            "url": "https://example.com/001",
            "published_date": "2024-03-15T10:00:00Z",
            "author": "John Smith",
            "category": "Business",
            "chunk_index": 1,
            "total_chunks": 3,
        },
    },
    {
        "id": "001_chunk_002",
        "text": "The stock market reacted negatively, with the S&P 500 dropping 1.2% in afternoon trading following the announcement.",
        "metadata": {
            "source": "CNN",
            "article_id": "001",
            "title": "Fed Raises Rates Again",
            "url": "https://example.com/001",
            "published_date": "2024-03-15T10:00:00Z",
            "author": "John Smith",
            "category": "Business",
            "chunk_index": 2,
            "total_chunks": 3,
        },
    },
    {
        "id": "002_chunk_000",
        "text": "SpaceX successfully launched its Starship rocket from Texas, achieving orbit for the first time in the vehicle's history.",
        "metadata": {
            "source": "Reuters",
            "article_id": "002",
            "title": "SpaceX Starship Reaches Orbit",
            "url": "https://example.com/002",
            "published_date": "2024-03-16T08:30:00Z",
            "author": "Jane Doe",
            "category": "Tech",
            "chunk_index": 0,
            "total_chunks": 2,
        },
    },
    {
        "id": "002_chunk_001",
        "text": "NASA congratulated SpaceX on the milestone, noting it brings the agency closer to its Artemis moon landing goals.",
        "metadata": {
            "source": "Reuters",
            "article_id": "002",
            "title": "SpaceX Starship Reaches Orbit",
            "url": "https://example.com/002",
            "published_date": "2024-03-16T08:30:00Z",
            "author": "Jane Doe",
            "category": "Tech",
            "chunk_index": 1,
            "total_chunks": 2,
        },
    },
]


def cleanup():
    if os.path.exists(TEST_DB_PATH):
        import gc
        gc.collect()
        import time
        time.sleep(0.5)
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


def test_embedding_function():
    print("=" * 60)
    print("TEST: Embedding Function")
    print("=" * 60)

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    texts = ["Hello world", "This is a test"]
    embeddings = ef(texts)

    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
    assert len(embeddings[0]) == 384, f"Expected dim=384, got {len(embeddings[0])}"
    import numpy as np
    assert all(isinstance(v, (float, int, np.floating)) for v in embeddings[0]), "Embeddings should be numeric"

    print(f"  ✅ Produced {len(embeddings)} embeddings, dim={len(embeddings[0])}")

    config = {
        "embedding": {
            "provider": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "dimensions": 384,
        }
    }
    ef2 = get_embedding_function(config)
    assert isinstance(ef2, SentenceTransformerEmbeddingFunction)
    print("  ✅ Factory function works correctly")

    info = ef.get_info()
    assert info["provider"] == "sentence-transformers"
    assert info["model_name"] == "all-MiniLM-L6-v2"
    assert info["output_dimensions"] == 384
    assert info["max_input_tokens"] == 256
    assert isinstance(info["use_cases"], str) and len(info["use_cases"]) > 0
    print(f"  ✅ get_info(): provider={info['provider']}, model={info['model_name']}, "
          f"dims={info['output_dimensions']}, max_tokens={info['max_input_tokens']}")
    print(f"     use_cases: {info['use_cases']}")


def test_create_collection():
    print("\n" + "=" * 60)
    print("TEST: Create Collection")
    print("=" * 60)

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    store = ChromaStore(TEST_DB_PATH, ef)
    col = store.get_or_create_collection(COLLECTION_NAME, HNSW_CONFIG)

    assert col is not None, "Collection should not be None"
    assert col.count() == 0, "New collection should be empty"
    print(f"  ✅ Created collection '{COLLECTION_NAME}', count={col.count()}")

    col2 = store.get_or_create_collection(COLLECTION_NAME, HNSW_CONFIG)
    assert col2 is not None, "Should get existing collection"
    print(f"  ✅ Retrieved existing collection '{COLLECTION_NAME}'")


def test_upsert_chunks():
    print("\n" + "=" * 60)
    print("TEST: Upsert Chunks")
    print("=" * 60)

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    store = ChromaStore(TEST_DB_PATH, ef)

    result = store.upsert_chunks(COLLECTION_NAME, SAMPLE_CHUNKS)
    print(f"  ✅ Upserted: {result}")

    stats = store.get_collection_stats(COLLECTION_NAME)
    assert stats["count"] == 5, f"Expected 5, got {stats['count']}"
    print(f"  ✅ Collection count: {stats['count']}")

    result2 = store.upsert_chunks(COLLECTION_NAME, SAMPLE_CHUNKS)
    stats2 = store.get_collection_stats(COLLECTION_NAME)
    assert stats2["count"] == 5, f"Re-upsert should keep count at 5, got {stats2['count']}"
    print(f"  ✅ Idempotent upsert verified, count still: {stats2['count']}")


def test_query():
    print("\n" + "=" * 60)
    print("TEST: Query")
    print("=" * 60)

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    store = ChromaStore(TEST_DB_PATH, ef)

    # Single text query
    results = store.query(COLLECTION_NAME, query_texts="interest rate increase", n_results=3)

    assert "ids" in results, "Results should have 'ids' key"
    assert "documents" in results, "Results should have 'documents' key"
    assert "distances" in results, "Results should have 'distances' key"
    assert len(results["ids"][0]) <= 3, "Should return at most 3 results"

    print(f"  ✅ Single text query returned {len(results['ids'][0])} results")
    for i, (doc_id, doc, dist) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    )):
        print(f"     [{i+1}] id={doc_id}, distance={dist:.4f}")
        print(f"         {doc[:80]}...")

    # Multiple text queries at once
    results_multi = store.query(
        COLLECTION_NAME,
        query_texts=["interest rate", "rocket launch"],
        n_results=2,
    )
    assert len(results_multi["ids"]) == 2, "Should have 2 result sets for 2 queries"
    print(f"  ✅ Multi-text query returned {len(results_multi['ids'])} result sets")

    # Query with metadata filter
    results_filtered = store.query(
        COLLECTION_NAME,
        query_texts="rocket launch",
        n_results=5,
        where={"source": "Reuters"},
    )
    for doc_meta in results_filtered["metadatas"][0]:
        assert doc_meta["source"] == "Reuters", f"Filter failed: got source={doc_meta['source']}"
    print(f"  ✅ Filtered query (source=Reuters) returned {len(results_filtered['ids'][0])} results")

    # Query with where_document filter
    results_doc_filter = store.query(
        COLLECTION_NAME,
        query_texts="economy",
        n_results=5,
        where_document={"$contains": "Federal Reserve"},
    )
    for doc in results_doc_filter["documents"][0]:
        assert "Federal Reserve" in doc, "where_document filter failed"
    print(f"  ✅ where_document filter returned {len(results_doc_filter['ids'][0])} results")

    # Query with include parameter
    results_with_embeddings = store.query(
        COLLECTION_NAME,
        query_texts="stock market",
        n_results=2,
        include=["documents", "distances", "metadatas", "embeddings"],
    )
    assert "embeddings" in results_with_embeddings, "Should include embeddings"
    assert results_with_embeddings["embeddings"] is not None, "Embeddings should not be None"
    print(f"  ✅ Query with include=['embeddings'] returned embedding vectors")

    # Query with raw embeddings
    sample_embedding = results_with_embeddings["embeddings"][0][0]
    results_by_embedding = store.query(
        COLLECTION_NAME,
        query_embeddings=[sample_embedding],
        n_results=2,
    )
    assert len(results_by_embedding["ids"][0]) <= 2, "Should return results for embedding query"
    print(f"  ✅ Query by raw embedding returned {len(results_by_embedding['ids'][0])} results")


def test_get():
    print("\n" + "=" * 60)
    print("TEST: Get")
    print("=" * 60)

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    store = ChromaStore(TEST_DB_PATH, ef)

    # Get by IDs
    result = store.get(COLLECTION_NAME, ids=["001_chunk_000", "002_chunk_000"])
    assert len(result["ids"]) == 2, f"Expected 2 results, got {len(result['ids'])}"
    print(f"  ✅ Get by IDs returned {len(result['ids'])} items")

    # Get by metadata filter
    result2 = store.get(COLLECTION_NAME, where={"category": "Tech"})
    assert all(m["category"] == "Tech" for m in result2["metadatas"]), "Filter should only return Tech"
    print(f"  ✅ Get by where filter (category=Tech) returned {len(result2['ids'])} items")

    # Get with where_document
    result3 = store.get(COLLECTION_NAME, where_document={"$contains": "SpaceX"})
    for doc in result3["documents"]:
        assert "SpaceX" in doc, "where_document filter failed"
    print(f"  ✅ Get by where_document ($contains SpaceX) returned {len(result3['ids'])} items")

    # Get with limit and offset
    result4 = store.get(COLLECTION_NAME, limit=2)
    assert len(result4["ids"]) == 2, f"Limit=2 should return 2, got {len(result4['ids'])}"
    print(f"  ✅ Get with limit=2 returned {len(result4['ids'])} items")

    result5 = store.get(COLLECTION_NAME, limit=2, offset=2)
    assert len(result5["ids"]) == 2, f"Limit=2 offset=2 should return 2, got {len(result5['ids'])}"
    assert result5["ids"] != result4["ids"], "Offset should return different items"
    print(f"  ✅ Get with limit=2, offset=2 returned different items")

    # Get with include embeddings
    result6 = store.get(COLLECTION_NAME, ids=["001_chunk_000"], include=["documents", "embeddings"])
    assert "embeddings" in result6, "Should include embeddings"
    assert result6["embeddings"] is not None, "Embeddings should not be None"
    print(f"  ✅ Get with include=['embeddings'] returned embedding vectors")

    # Get all (no filter)
    result_all = store.get(COLLECTION_NAME)
    assert len(result_all["ids"]) == 5, f"Get all should return 5, got {len(result_all['ids'])}"
    print(f"  ✅ Get all (no filter) returned {len(result_all['ids'])} items")


def test_collection_stats():
    print("\n" + "=" * 60)
    print("TEST: Collection Stats")
    print("=" * 60)

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    store = ChromaStore(TEST_DB_PATH, ef)

    stats = store.get_collection_stats(COLLECTION_NAME)
    assert stats["exists"] is True
    assert stats["count"] == 5
    assert "embedding_info" in stats
    assert stats["embedding_info"]["provider"] == "sentence-transformers"
    assert stats["embedding_info"]["output_dimensions"] == 384
    print(f"  ✅ Stats: exists={stats['exists']}, count={stats['count']}")
    print(f"     metadata={stats['metadata']}")
    print(f"     embedding: {stats['embedding_info']['provider']} / {stats['embedding_info']['model_name']} "
          f"(dims={stats['embedding_info']['output_dimensions']}, max_tokens={stats['embedding_info']['max_input_tokens']})")
    print(f"     use_cases: {stats['embedding_info']['use_cases']}")

    stats_missing = store.get_collection_stats("nonexistent_collection")
    assert stats_missing["exists"] is False
    assert "embedding_info" in stats_missing
    print(f"  ✅ Non-existent collection: exists={stats_missing['exists']} (still has embedding_info)")


def test_delete_collection():
    print("\n" + "=" * 60)
    print("TEST: Delete Collection")
    print("=" * 60)

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    store = ChromaStore(TEST_DB_PATH, ef)

    store.delete_collection(COLLECTION_NAME)

    stats = store.get_collection_stats(COLLECTION_NAME)
    assert stats["exists"] is False, "Collection should be deleted"
    print(f"  ✅ Collection deleted successfully")

    try:
        store.delete_collection("nonexistent_collection")
        assert False, "Should have raised ValueError"
    except ValueError:
        print(f"  ✅ Deleting non-existent collection raises ValueError")


def test_error_handling():
    print("\n" + "=" * 60)
    print("TEST: Error Handling")
    print("=" * 60)

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    store = ChromaStore(TEST_DB_PATH, ef)

    try:
        store.upsert_chunks("nonexistent", SAMPLE_CHUNKS)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✅ Upsert to non-existent collection: {e}")

    try:
        store.query("nonexistent", query_texts="test query")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✅ Query non-existent collection: {e}")

    result = store.upsert_chunks(COLLECTION_NAME, [])
    # Collection was deleted, but empty chunks should return early
    # Actually, we need to handle empty chunks first
    ef2 = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    store2 = ChromaStore(TEST_DB_PATH, ef2)
    store2.get_or_create_collection("temp_col")
    empty_result = store2.upsert_chunks("temp_col", [])
    assert empty_result["total"] == 0, "Empty upsert should return 0"
    print(f"  ✅ Empty chunks upsert: {empty_result}")
    store2.delete_collection("temp_col")


if __name__ == "__main__":
    print("🚀 ChromaStore & Embeddings Test Suite")
    print("Using SentenceTransformer (all-MiniLM-L6-v2) - no API key needed\n")

    cleanup()

    try:
        test_embedding_function()
        test_create_collection()
        test_upsert_chunks()
        test_query()
        test_get()
        test_collection_stats()
        test_delete_collection()
        test_error_handling()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()
        # pass if you want to see real database
