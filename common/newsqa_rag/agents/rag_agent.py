import re
import time

from newsqa_rag.retrieval import BaseRetriever
from newsqa_rag.retrieval.reranker import BaseReranker
from newsqa_rag.llm import OpenAILLM


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

    def retrieve(self, question: str) -> dict:
        """Retrieve candidates without reranking so the result can be shared."""
        t0 = time.perf_counter()
        retrieval_breakdown = {}
        if hasattr(self.retriever, "retrieve_with_timing"):
            retrieved, retrieval_breakdown = self.retriever.retrieve_with_timing(question, self.top_k)
        else:
            retrieved = self.retriever.retrieve(question, self.top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        return {
            "question": question,
            "retrieved_chunks": retrieved,
            "timing_ms": {**retrieval_breakdown, "retrieve_ms": round(retrieve_ms, 1)},
        }

    def rerank_trace(self, retrieval_trace: dict) -> dict:
        """Apply the configured reranker to a cached retrieval trace."""
        question = retrieval_trace["question"]
        retrieved = retrieval_trace["retrieved_chunks"]
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(question, retrieved, self.rerank_top_n)
        rerank_ms = (time.perf_counter() - t0) * 1000

        timing = dict(retrieval_trace.get("timing_ms", {}))
        timing["rerank_ms"] = round(rerank_ms, 1)
        timing["retrieval_total_ms"] = round(timing.get("retrieve_ms", 0.0) + rerank_ms, 1)
        return {
            **retrieval_trace,
            "reranked_chunks": reranked,
            "retrieved_ids": [result["id"] for result in reranked],
            "contexts": [result["text"] for result in reranked],
            "timing_ms": timing,
        }

    def retrieve_and_rerank(self, question: str) -> dict:
        """Retrieve and rerank once so the trace can be checkpointed."""
        return self.rerank_trace(self.retrieve(question))

    def generate_from_trace(
        self,
        trace: dict,
        *,
        system_prompt: str | None = None,
        context_depth: int | None = None,
    ) -> dict:
        """Generate from a frozen ranked trace and parse cited chunks."""
        if self.llm is None:
            raise RuntimeError("Generation requires an LLM instance.")

        ranked_chunks = trace["reranked_chunks"]
        if context_depth is not None and context_depth < 1:
            raise ValueError("context_depth must be at least 1")
        generation_chunks = ranked_chunks[:context_depth]
        if not generation_chunks:
            raise RuntimeError("Generation requires at least one ranked context.")
        contexts = [chunk["text"] for chunk in generation_chunks]

        t0 = time.perf_counter()
        if system_prompt is None:
            answer = self.llm.generate_rag_answer(trace["question"], contexts)
        else:
            answer = self.llm.generate_rag_answer(
                trace["question"], contexts, system_prompt=system_prompt
            )
        llm_ms = (time.perf_counter() - t0) * 1000
        if not answer.strip():
            raise RuntimeError("The generator returned an empty answer.")

        raw_indices = [int(value) for value in self._CITATION_PATTERN.findall(answer)]
        citation_indices = list(dict.fromkeys(raw_indices))
        valid_indices = [
            index for index in citation_indices if 1 <= index <= len(generation_chunks)
        ]
        invalid_indices = [index for index in citation_indices if index not in valid_indices]
        cited_chunks = [generation_chunks[index - 1] for index in valid_indices]

        timing = dict(trace.get("timing_ms", {}))
        timing["llm_ms"] = round(llm_ms, 1)
        timing["total_ms"] = round(
            timing.get("retrieval_total_ms", 0.0) + llm_ms, 1
        )
        return {
            **trace,
            "contexts": contexts,
            "generation_context_depth": len(generation_chunks),
            "generation_context_chunk_ids": [chunk["id"] for chunk in generation_chunks],
            "answer": answer,
            "citation_indices": valid_indices,
            "citation_chunk_ids": [chunk["id"] for chunk in cited_chunks],
            "invalid_citation_indices": invalid_indices,
            "cited_chunks": cited_chunks,
            "usage": dict(getattr(self.llm, "last_usage", {}) or {}),
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
