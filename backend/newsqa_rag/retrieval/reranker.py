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


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker loaded lazily on first use."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, results: list[dict], top_n: int) -> list[dict]:
        if not results:
            return []
        pairs = [(query, result.get("text", "")) for result in results]
        scores = self._get_model().predict(pairs)
        rescored = []
        for result, reranker_score in zip(results, scores):
            item = dict(result)
            item["retrieval_score"] = item.get("score")
            item["reranker_score"] = float(reranker_score)
            item["score"] = float(reranker_score)
            rescored.append(item)
        rescored.sort(key=lambda item: item["reranker_score"], reverse=True)
        return rescored[:top_n]

    def get_info(self) -> Dict[str, Any]:
        return {"type": "cross-encoder", "model": self.model_name}


def get_reranker(config: dict) -> BaseReranker:
    """
    Factory. Reads config["retrieval"]["reranker"]["type"].
    Supported types: "noop" and "cross-encoder"
    """
    reranker_cfg = config.get("retrieval", {}).get("reranker", {})
    reranker_type = reranker_cfg.get("type", "noop")

    if reranker_type == "noop":
        return NoOpReranker()
    if reranker_type == "cross-encoder":
        return CrossEncoderReranker(
            reranker_cfg.get(
                "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )
    else:
        raise ValueError(
            f"Unknown reranker type: '{reranker_type}'. "
            "Supported: 'noop', 'cross-encoder'."
        )
