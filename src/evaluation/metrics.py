import hashlib
import math
import re
from collections import Counter
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """1 if any of the top-k retrieved IDs is relevant, else 0."""
    return float(any(r in set(relevant) for r in retrieved[:k]))


def mrr_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result in top-k."""
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved[:k]):
        if r in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Fraction of relevant IDs that appear in top-k results."""
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in set(relevant))
    return hits / len(relevant)


def ndcg_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(retrieved[:k])
        if r in relevant_set
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_retrieval(
    samples: list[dict],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    """
    Aggregate retrieval metrics over a list of samples.

    Args:
        samples: list of {relevant_chunk_ids: list[str], retrieved_ids: list[str]}
        k_values: K values to evaluate at.

    Returns:
        Dict of metric_name → mean value across all samples.
    """
    results = {}
    for k in k_values:
        hr = [hit_rate_at_k(s["relevant_chunk_ids"], s["retrieved_ids"], k) for s in samples]
        mrr = [mrr_at_k(s["relevant_chunk_ids"], s["retrieved_ids"], k) for s in samples]
        rec = [recall_at_k(s["relevant_chunk_ids"], s["retrieved_ids"], k) for s in samples]
        ndcg = [ndcg_at_k(s["relevant_chunk_ids"], s["retrieved_ids"], k) for s in samples]
        results[f"hit_rate@{k}"] = round(float(np.mean(hr)), 4)
        results[f"mrr@{k}"] = round(float(np.mean(mrr)), 4)
        results[f"recall@{k}"] = round(float(np.mean(rec)), 4)
        results[f"ndcg@{k}"] = round(float(np.mean(ndcg)), 4)
    results["n_samples"] = len(samples)
    return results


# ---------------------------------------------------------------------------
# QA metrics
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def f1_score_qa(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth (standard NewsQA metric)."""
    pred_tokens = _normalize_answer(prediction).split()
    truth_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_qa(samples: list[dict]) -> dict:
    """
    Aggregate QA metrics.

    Args:
        samples: list of {prediction: str, ground_truth: str}

    Returns:
        {exact_match, f1, n_samples}
    """
    em_scores = [exact_match(s["prediction"], s["ground_truth"]) for s in samples]
    f1_scores = [f1_score_qa(s["prediction"], s["ground_truth"]) for s in samples]
    return {
        "exact_match": round(float(np.mean(em_scores)), 4),
        "f1": round(float(np.mean(f1_scores)), 4),
        "n_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Chunking diagnostic metrics
# ---------------------------------------------------------------------------

def evaluate_chunking(chunks: list[dict]) -> dict:
    """
    Diagnostic stats for a chunked collection. No ground truth needed.

    Args:
        chunks: list of {id, text, metadata}

    Returns:
        {total_chunks, mean_tokens, std_tokens, min_tokens, max_tokens,
         chunks_per_article_mean, chunks_per_article_std}
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(enc.encode(c["text"])) for c in chunks]

    article_ids = [c["metadata"].get("article_id", c["id"]) for c in chunks]
    from collections import Counter as _Counter
    per_article = list(_Counter(article_ids).values())

    return {
        "total_chunks": len(chunks),
        "mean_tokens": round(float(np.mean(token_counts)), 1),
        "std_tokens": round(float(np.std(token_counts)), 1),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "chunks_per_article_mean": round(float(np.mean(per_article)), 2),
        "chunks_per_article_std": round(float(np.std(per_article)), 2),
    }


_SENTENCE_END = re.compile(r'[.!?]["\')\]]?$')


def deduplication_rate(texts: list[str]) -> float:
    """Fraction of chunks that are exact duplicates (by MD5). 0.0 = all unique. (spec 3.2)"""
    if not texts:
        return 0.0
    hashes = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
    return round(1 - len(set(hashes)) / len(hashes), 4)


def semantic_integrity(texts: list[str]) -> float:
    """
    Fraction of chunks that end at a sentence boundary (spec 3.1).
    Proxy for "chunk not cut mid-sentence"; higher is better.
    ponytail: regex heuristic, swap for spaCy sentence segmentation if it misjudges.
    """
    if not texts:
        return 0.0
    ok = sum(1 for t in texts if _SENTENCE_END.search(t.strip()))
    return round(ok / len(texts), 4)


def delta_mrr(samples_initial: list[dict], samples_reranked: list[dict], k: int = 5) -> float:
    """MRR(reranked) - MRR(initial): did the reranker improve ordering? (spec 3.4)"""
    mi = np.mean([mrr_at_k(s["relevant_chunk_ids"], s["retrieved_ids"], k) for s in samples_initial])
    mr = np.mean([mrr_at_k(s["relevant_chunk_ids"], s["retrieved_ids"], k) for s in samples_reranked])
    return round(float(mr - mi), 4)


# ---------------------------------------------------------------------------
# RAGAS wrapper
# ---------------------------------------------------------------------------

def _ragas_shim() -> None:
    """Let ragas 0.4.x import under langchain 1.x (Vertex AI symbols were removed)."""
    import sys
    import types

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        m = types.ModuleType("langchain_community.chat_models.vertexai")
        m.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = m
    import langchain_community.llms as _llms
    if not hasattr(_llms, "VertexAI"):
        _llms.VertexAI = type("VertexAI", (), {})


def _ragas_judge(llm_model: str):
    """
    Build the RAGAS judge LLM from env, cheapest option first.

    DEEPSEEK_API_KEY set → DeepSeek (OpenAI-compatible, ~10x cheaper than GPT-4o).
    Else → OpenAI with OPENAI_API_KEY. Embeddings are always local (free): DeepSeek
    has no embeddings endpoint and answer_relevancy needs one.
    """
    import os

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if os.getenv("DEEPSEEK_API_KEY"):
        model = llm_model if llm_model.startswith("deepseek") else "deepseek-chat"
        chat = ChatOpenAI(model=model, api_key=os.environ["DEEPSEEK_API_KEY"],
                          base_url="https://api.deepseek.com", temperature=0, max_retries=2)
    else:
        chat = ChatOpenAI(model=llm_model, temperature=0)

    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)


def evaluate_ragas(
    samples: list[dict],
    metrics: Optional[list[str]] = None,
    llm_model: str = "deepseek-chat",
) -> dict:
    """
    Run RAGAS evaluation with a configurable judge LLM.

    Args:
        samples: list of {question, answer, contexts (list[str]), ground_truth}
        metrics: subset of ["faithfulness", "answer_relevancy", "context_precision",
                            "context_recall", "answer_correctness"]. Defaults to all five.
        llm_model: judge model name. DeepSeek is used automatically when DEEPSEEK_API_KEY
                   is set (see _ragas_judge); otherwise this is the OpenAI model.

    Returns:
        Dict of metric_name → mean score.
    """
    import os

    _ragas_shim()

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    )
    from datasets import Dataset

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }

    if metrics is None:
        metrics = list(metric_map.keys())

    selected = [metric_map[m] for m in metrics if m in metric_map]
    judge, embeddings = _ragas_judge(llm_model)

    # DeepSeek only supports n=1; answer_relevancy defaults to 3 generations -> 400 error.
    if os.getenv("DEEPSEEK_API_KEY"):
        answer_relevancy.strictness = 1

    dataset = Dataset.from_list([
        {
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in samples
    ])

    result = evaluate(dataset=dataset, metrics=selected, llm=judge, embeddings=embeddings)

    df = result.to_pandas()
    return {m: round(float(df[m].mean()), 4) for m in metrics if m in df.columns}


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    config_snapshot: dict,
    retrieval_metrics: dict | None = None,
    qa_metrics: dict | None = None,
    ragas_metrics: dict | None = None,
    chunking_metrics: dict | None = None,
) -> dict:
    """Assemble a single report dict from all available metric groups."""
    report = {"config": config_snapshot}
    if chunking_metrics:
        report["chunking"] = chunking_metrics
    if retrieval_metrics:
        report["retrieval"] = retrieval_metrics
    if qa_metrics:
        report["qa"] = qa_metrics
    if ragas_metrics:
        report["ragas"] = ragas_metrics
    return report


if __name__ == "__main__":
    # ponytail self-check: metrics that break silently would ruin a whole report
    assert hit_rate_at_k(["a"], ["b", "a"], 2) == 1.0
    assert hit_rate_at_k(["a"], ["b", "c"], 2) == 0.0
    assert mrr_at_k(["a"], ["b", "a"], 5) == 0.5
    assert recall_at_k(["a", "b"], ["a", "x"], 5) == 0.5
    assert round(ndcg_at_k(["a"], ["a"], 5), 4) == 1.0
    assert exact_match("The Cat.", "the cat") == 1.0
    assert 0.66 < f1_score_qa("a b c", "a b d") < 0.67
    assert deduplication_rate(["x", "x", "y"]) == round(1 / 3, 4)
    assert semantic_integrity(["Ends here.", "cut mid"]) == 0.5
    assert delta_mrr(
        [{"relevant_chunk_ids": ["a"], "retrieved_ids": ["b", "a"]}],
        [{"relevant_chunk_ids": ["a"], "retrieved_ids": ["a", "b"]}],
    ) == 0.5
    print("metrics self-check OK")
