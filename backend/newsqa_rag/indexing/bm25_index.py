"""BM25 and sparse lexical index implementation with multiple scoring variants and tokenizers."""

from __future__ import annotations

import os
import pickle
import re
from typing import Any, Callable

from rank_bm25 import BM25L, BM25Okapi, BM25Plus

# Standard minimal English stopword list
DEFAULT_ENGLISH_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves",
    }
)


def _tokenize_simple(text: str) -> list[str]:
    """Basic whitespace tokenizer with lowercasing."""
    return text.lower().split()


def _tokenize_stemmed(text: str) -> list[str]:
    """Tokenize with regex word boundaries, stopword removal, and Snowball stemming."""
    try:
        from nltk.stem.snowball import SnowballStemmer
    except ImportError as error:
        raise RuntimeError("Snowball stemming requires the nltk package") from error
    stemmer = SnowballStemmer("english")

    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    filtered = [w for w in words if w not in DEFAULT_ENGLISH_STOPWORDS]
    return [stemmer.stem(w) for w in filtered]


_TOKENIZER_REGISTRY: dict[str, Callable[[str], list[str]]] = {
    "simple": _tokenize_simple,
    "stem": _tokenize_stemmed,
    "stemmed": _tokenize_stemmed,
    "snowball": _tokenize_stemmed,
}

_VARIANT_CLASS_MAP = {
    "okapi": BM25Okapi,
    "bm25okapi": BM25Okapi,
    "plus": BM25Plus,
    "bm25plus": BM25Plus,
    "bm25+": BM25Plus,
    "l": BM25L,
    "bm25l": BM25L,
}


class BM25Index:
    """Sparse lexical index over chunk texts supporting multiple scoring algorithms and tokenizers.

    Stores a position -> chunk_id mapping so query results can be joined
    with ChromaDB results in hybrid reciprocal rank fusion.
    """

    def __init__(
        self,
        variant: str = "okapi",
        tokenizer_mode: str = "simple",
    ):
        self.variant = variant.lower().strip()
        self.tokenizer_mode = tokenizer_mode.lower().strip()
        self._index: BM25Okapi | BM25Plus | BM25L | None = None
        self._id_map: list[str] = []

    def _get_tokenizer(self) -> Callable[[str], list[str]]:
        try:
            return _TOKENIZER_REGISTRY[self.tokenizer_mode]
        except KeyError as error:
            raise ValueError(f"Unknown BM25 tokenizer mode: {self.tokenizer_mode}") from error

    def build(
        self,
        chunks: list[dict],
        variant: str | None = None,
        tokenizer_mode: str | None = None,
    ) -> None:
        """Build BM25 index from chunk dictionaries.

        Args:
            chunks: List of {id, text, metadata} dicts.
            variant: Optional override for scoring algorithm ('okapi', 'plus', 'l').
            tokenizer_mode: Optional override for tokenizer ('simple', 'stem').
        """
        if variant is not None:
            self.variant = variant.lower().strip()
        if tokenizer_mode is not None:
            self.tokenizer_mode = tokenizer_mode.lower().strip()

        self._id_map = [c["id"] for c in chunks]
        tokenizer = self._get_tokenizer()
        tokenized_corpus = [tokenizer(c.get("text", "")) for c in chunks]

        try:
            cls = _VARIANT_CLASS_MAP[self.variant]
        except KeyError as error:
            raise ValueError(f"Unknown BM25 variant: {self.variant}") from error
        self._index = cls(tokenized_corpus)

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """Search BM25 index.

        Args:
            query_text: Natural language user question.
            top_k: Maximum number of chunks to return.

        Returns:
            List of {id, score} sorted by score descending (score > 0).
        """
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokenizer = self._get_tokenizer()
        tokens = tokenizer(query_text)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {"id": self._id_map[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: str) -> None:
        """Persist index to disk as pickle."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "index": self._index,
            "id_map": self._id_map,
            "variant": self.variant,
            "tokenizer_mode": self.tokenizer_mode,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load persisted index from disk with backward compatibility."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            variant = data.get("variant", "okapi")
            tokenizer_mode = data.get("tokenizer_mode", "simple")
            obj = cls(variant=variant, tokenizer_mode=tokenizer_mode)
            obj._index = data.get("index")
            obj._id_map = data.get("id_map", [])
            return obj

        obj = cls()
        obj._index = data
        return obj

    @property
    def size(self) -> int:
        return len(self._id_map)
