from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Retrieve top_k results for query.
        Returns list of {id, text, score, metadata} sorted by score descending.
        """
        ...
