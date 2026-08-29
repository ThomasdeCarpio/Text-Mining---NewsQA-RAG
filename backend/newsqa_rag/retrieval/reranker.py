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

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, results: list[dict], top_n: int) -> list[dict]:
        if not results:
            return []
        pairs = [(query, result.get("text", "")) for result in results]
        scores = self._get_model().predict(pairs, batch_size=self.batch_size)
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
        return {
            "type": "cross-encoder",
            "model": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
        }


class BGESequenceClassificationReranker(BaseReranker):
    """BGE encoder reranker using its native Transformers inference contract."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        device: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = min(batch_size, 8)
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._effective_device = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._effective_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            load_options = {"low_cpu_mem_usage": True}
            if str(self._effective_device).startswith("cuda"):
                load_options["torch_dtype"] = torch.float16
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **load_options,
            )
            self._model.to(self._effective_device)
            self._model.eval()
        return self._model, self._tokenizer

    def rerank(self, query: str, results: list[dict], top_n: int) -> list[dict]:
        if not results:
            return []
        import torch

        model, tokenizer = self._load()
        pairs = [[query, result.get("text", "")] for result in results]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            inputs = tokenizer(
                pairs[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._effective_device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=str(self._effective_device).startswith("cuda"),
            ):
                logits = model(**inputs, return_dict=True).logits.view(-1)
            scores.extend(logits.float().cpu().tolist())

        rescored = []
        for result, score in zip(results, scores):
            item = dict(result)
            item["retrieval_score"] = item.get("score")
            item["reranker_score"] = float(score)
            item["score"] = float(score)
            rescored.append(item)
        rescored.sort(key=lambda item: item["reranker_score"], reverse=True)
        return rescored[:top_n]

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "cross-encoder",
            "backend": "transformers-sequence-classification",
            "model": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
            "max_length": self.max_length,
        }


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
        model_name = reranker_cfg.get(
            "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        batch_size = int(reranker_cfg.get("batch_size", 32))
        device = reranker_cfg.get("device")
        if str(model_name).startswith("BAAI/bge-reranker"):
            return BGESequenceClassificationReranker(
                model_name,
                batch_size=batch_size,
                device=device,
                max_length=int(reranker_cfg.get("max_length", 512)),
            )
        return CrossEncoderReranker(
            model_name,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown reranker type: '{reranker_type}'. "
            "Supported: 'noop', 'cross-encoder'."
        )
