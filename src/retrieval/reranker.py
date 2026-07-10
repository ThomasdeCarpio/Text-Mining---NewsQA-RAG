from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[dict], top_n: int) -> list[dict]:
        """
        Rerank retrieved results.

        Args:
            query: The original query string.
            results: list of {id, text, score, metadata} from a retriever.
            top_n: Number of results to return after reranking.

        Returns:
            top_n results sorted by relevance descending.
        """
        ...

    def get_info(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}


class NoOpReranker(BaseReranker):
    """Passthrough reranker — returns the top_n results unchanged. Use as baseline."""

    def rerank(self, query: str, results: list[dict], top_n: int) -> list[dict]:
        return results[:top_n]

    def get_info(self) -> Dict[str, Any]:
        return {"type": "noop", "description": "No reranking — passthrough baseline"}


def get_reranker(config: dict) -> BaseReranker:
    """
    Factory. Reads config["retrieval"]["reranker"]["type"].
    Supported types: "noop"
    """
    reranker_cfg = config.get("retrieval", {}).get("reranker", {})
    reranker_type = reranker_cfg.get("type", "noop")

    if reranker_type == "noop":
        return NoOpReranker()
    else:
        raise ValueError(
            f"Unknown reranker type: '{reranker_type}'. Supported: 'noop'."
        )
