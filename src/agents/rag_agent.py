import re
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
        llm: OpenAILLM | None,
        top_k: int = 10,
        rerank_top_n: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

    _CITATION_PATTERN = re.compile(r"\[(\d+)]")

    def retrieve_and_rerank(self, question: str) -> dict:
        """Retrieve and rerank once so the trace can be checkpointed."""
        t_total = time.perf_counter()

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
            "retrieved_ids": [result["id"] for result in reranked],
            "contexts": [result["text"] for result in reranked],
            "timing_ms": {
                "retrieve_ms": round(retrieve_ms, 1),
                "rerank_ms": round(rerank_ms, 1),
                "retrieval_total_ms": round(
                    (time.perf_counter() - t_total) * 1000, 1
                ),
            },
        }

    def generate_from_trace(self, trace: dict) -> dict:
        """Generate from an existing retrieval trace and parse cited chunks."""
        if self.llm is None:
            raise RuntimeError("Generation requires an LLM instance.")

        t0 = time.perf_counter()
        answer = self.llm.generate_rag_answer(trace["question"], trace["contexts"])
        llm_ms = (time.perf_counter() - t0) * 1000
        if not answer.strip():
            raise RuntimeError("The generator returned an empty answer.")

        raw_indices = [int(value) for value in self._CITATION_PATTERN.findall(answer)]
        citation_indices = list(dict.fromkeys(raw_indices))
        valid_indices = [
            index for index in citation_indices if 1 <= index <= len(trace["reranked_chunks"])
        ]
        invalid_indices = [index for index in citation_indices if index not in valid_indices]
        cited_chunks = [trace["reranked_chunks"][index - 1] for index in valid_indices]

        timing = dict(trace.get("timing_ms", {}))
        timing["llm_ms"] = round(llm_ms, 1)
        timing["total_ms"] = round(
            timing.get("retrieval_total_ms", 0.0) + llm_ms, 1
        )
        return {
            **trace,
            "answer": answer,
            "citation_indices": valid_indices,
            "citation_chunk_ids": [chunk["id"] for chunk in cited_chunks],
            "invalid_citation_indices": invalid_indices,
            "cited_chunks": cited_chunks,
            "timing_ms": timing,
        }

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
              "citation_chunk_ids": list[str],
              "invalid_citation_indices": list[int],
              "timing_ms": {retrieve_ms, rerank_ms, llm_ms, total_ms}
            }
        """
        return self.generate_from_trace(self.retrieve_and_rerank(question))

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
        return self.retrieve_and_rerank(question)
