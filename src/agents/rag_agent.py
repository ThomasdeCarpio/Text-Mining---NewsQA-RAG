import time

from src.retrieval import BaseRetriever
from src.retrieval.reranker import BaseReranker
from src.llm import OpenAILLM


class RAGAgent:
    """
    Ties retriever → reranker → LLM into a single pipeline call.

    run() returns a dict with the full trace:
      question, retrieved_chunks, reranked_chunks, contexts, answer, timing_ms
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        llm: OpenAILLM,
        top_k: int = 10,
        rerank_top_n: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

    def run(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a single question.

        Returns:
            {
              "question": str,
              "retrieved_chunks": list[dict],   # top_k from retriever
              "reranked_chunks": list[dict],    # top rerank_top_n after reranker
              "contexts": list[str],            # text of reranked chunks
              "answer": str,
              "timing_ms": {retrieve_ms, rerank_ms, llm_ms, total_ms}
            }
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, self.top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reranked = self.reranker.rerank(question, retrieved, self.rerank_top_n)
        rerank_ms = (time.perf_counter() - t0) * 1000

        contexts = [r["text"] for r in reranked]

        t0 = time.perf_counter()
        answer = self.llm.generate_rag_answer(question, contexts)
        llm_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        return {
            "question": question,
            "retrieved_chunks": retrieved,
            "reranked_chunks": reranked,
            "contexts": contexts,
            "answer": answer,
            "timing_ms": {
                "retrieve_ms": round(retrieve_ms, 1),
                "rerank_ms": round(rerank_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "total_ms": round(total_ms, 1),
            },
        }

    def run_retrieval_only(self, question: str) -> dict:
        """
        Run only retrieval + reranking (no LLM). Useful for retrieval benchmarks.

        Returns:
            {
              "question": str,
              "retrieved_chunks": list[dict],
              "reranked_chunks": list[dict],
              "retrieved_ids": list[str],    # ordered IDs for metric computation
              "timing_ms": {retrieve_ms, rerank_ms}
            }
        """
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, self.top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reranked = self.reranker.rerank(question, retrieved, self.rerank_top_n)
        rerank_ms = (time.perf_counter() - t0) * 1000

        return {
            "question": question,
            "retrieved_chunks": retrieved,
            "reranked_chunks": reranked,
            "retrieved_ids": [r["id"] for r in reranked],
            "timing_ms": {
                "retrieve_ms": round(retrieve_ms, 1),
                "rerank_ms": round(rerank_ms, 1),
            },
        }
