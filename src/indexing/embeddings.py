# Custome embedding function implementation

from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

@register_embedding_function
class MyEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings

    @staticmethod
    def name() -> str:
        return "my-ef"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return MyEmbeddingFunction(config['model'])