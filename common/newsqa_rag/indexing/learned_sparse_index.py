"""Persisted learned-sparse inverted index backed by BGE-M3 lexical weights."""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


class BGEM3SparseEncoder:
    """Lazy BGE-M3 lexical-weight encoder."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as error:
                raise RuntimeError(
                    "BGE-M3 sparse retrieval requires FlagEmbedding. "
                    "Install the project evaluation dependencies first."
                ) from error
            kwargs: dict[str, Any] = {"use_fp16": False}
            if self.device:
                kwargs["devices"] = self.device
            self._model = BGEM3FlagModel(self.model_name, **kwargs)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> list[dict[str, float]]:
        output = self._get_model().encode(
            texts,
            batch_size=batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        values = output["lexical_weights"]
        return [{str(token): float(weight) for token, weight in row.items()} for row in values]


class LearnedSparseIndex:
    """Inverted index scored by the dot product of learned lexical weights."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        encoder: BGEM3SparseEncoder | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.encoder = encoder or BGEM3SparseEncoder(model_name, device)
        self._postings: dict[str, list[tuple[str, float]]] = {}
        self._size = 0

    def build(self, chunks: list[dict], batch_size: int = 32) -> None:
        postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            weights = self.encoder.encode([row.get("text", "") for row in batch], batch_size)
            for chunk, lexical_weights in zip(batch, weights):
                for token, weight in lexical_weights.items():
                    if weight:
                        postings[token].append((chunk["id"], weight))
        self._postings = dict(postings)
        self._size = len(chunks)

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        if not self._postings:
            raise RuntimeError("Learned sparse index not built. Call build() first.")
        query_weights = self.encoder.encode([query_text], 1)[0]
        scores: dict[str, float] = defaultdict(float)
        for token, query_weight in query_weights.items():
            for chunk_id, document_weight in self._postings.get(token, ()):
                scores[chunk_id] += query_weight * document_weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [{"id": chunk_id, "score": float(score)} for chunk_id, score in ranked]

    def save(self, path: str, metadata: dict | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "schema_version": 1,
                    "model_name": self.model_name,
                    "size": self._size,
                    "postings": self._postings,
                    "metadata": metadata or {},
                },
                handle,
            )
        os.replace(temporary, target)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device: str | None = None,
        encoder: BGEM3SparseEncoder | None = None,
    ) -> "LearnedSparseIndex":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported learned sparse index schema")
        obj = cls(payload["model_name"], device=device, encoder=encoder)
        obj._size = int(payload["size"])
        obj._postings = payload["postings"]
        return obj

    @property
    def size(self) -> int:
        return self._size
