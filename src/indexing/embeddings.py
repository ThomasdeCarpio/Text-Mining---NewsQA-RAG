from typing import Dict, Any, List
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function


OPENAI_MODEL_INFO = {
    "text-embedding-3-small": {
        "max_input_tokens": 8191,
        "default_dimensions": 1536,
        "use_cases": "General-purpose: search, clustering, classification. Good balance of cost and quality.",
    },
    "text-embedding-3-large": {
        "max_input_tokens": 8191,
        "default_dimensions": 3072,
        "use_cases": "High-accuracy tasks: semantic search, RAG, fine-grained similarity. Higher cost.",
    },
    "text-embedding-ada-002": {
        "max_input_tokens": 8191,
        "default_dimensions": 1536,
        "use_cases": "Legacy model. Use text-embedding-3-small/large instead.",
    },
}


@register_embedding_function
class OpenAIEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model_name: str = "text-embedding-3-small", dimensions: int = 1536):
        self.model_name = model_name
        self.dimensions = dimensions
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def __call__(self, input: Documents) -> Embeddings:
        client = self._get_client()
        batch_size = 2048
        all_embeddings: List[List[float]] = []

        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self.model_name,
                dimensions=self.dimensions,
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings

    def get_info(self) -> Dict[str, Any]:
        model_info = OPENAI_MODEL_INFO.get(self.model_name, {})
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "output_dimensions": self.dimensions,
            "max_input_tokens": model_info.get("max_input_tokens", "unknown"),
            "default_dimensions": model_info.get("default_dimensions", "unknown"),
            "use_cases": model_info.get("use_cases", "Unknown model — refer to OpenAI docs."),
        }

    @staticmethod
    def name() -> str:
        return "openai-ef"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OpenAIEmbeddingFunction":
        return OpenAIEmbeddingFunction(
            model_name=config["model_name"],
            dimensions=config["dimensions"],
        )


SENTENCE_TRANSFORMER_MODEL_INFO = {
    "all-MiniLM-L6-v2": {
        "output_dimensions": 384,
        "max_input_tokens": 256,
        "use_cases": "Lightweight general-purpose: semantic search, clustering. Fast inference, low memory.",
    },
    "all-mpnet-base-v2": {
        "output_dimensions": 768,
        "max_input_tokens": 384,
        "use_cases": "High-quality general-purpose: semantic search, RAG, classification. Better accuracy than MiniLM.",
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "output_dimensions": 384,
        "max_input_tokens": 512,
        "use_cases": "Optimized for question-answering and semantic search over passages.",
    },
    "multi-qa-mpnet-base-dot-v1": {
        "output_dimensions": 768,
        "max_input_tokens": 512,
        "use_cases": "High-quality QA retrieval. Best for question-to-passage matching.",
    },
}


@register_embedding_function
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        model = self._get_model()
        embeddings = model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def get_info(self) -> Dict[str, Any]:
        known = SENTENCE_TRANSFORMER_MODEL_INFO.get(self.model_name)
        if known:
            return {
                "provider": "sentence-transformers",
                "model_name": self.model_name,
                "output_dimensions": known["output_dimensions"],
                "max_input_tokens": known["max_input_tokens"],
                "use_cases": known["use_cases"],
            }
        model = self._get_model()
        return {
            "provider": "sentence-transformers",
            "model_name": self.model_name,
            "output_dimensions": model.get_sentence_embedding_dimension(),
            "max_input_tokens": model.max_seq_length,
            "use_cases": "Custom model — refer to Hugging Face model card.",
        }

    @staticmethod
    def name() -> str:
        return "sentence-transformer-ef"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "SentenceTransformerEmbeddingFunction":
        return SentenceTransformerEmbeddingFunction(model_name=config["model_name"])


def get_embedding_function(config: dict) -> EmbeddingFunction:
    """
    Factory function. Returns a ChromaDB-compatible EmbeddingFunction
    based on config["embedding"]["provider"].
    """
    emb_config = config["embedding"]
    provider = emb_config["provider"]

    if provider == "openai":
        return OpenAIEmbeddingFunction(
            model_name=emb_config["model_name"],
            dimensions=emb_config["dimensions"],
        )
    elif provider == "sentence-transformers":
        return SentenceTransformerEmbeddingFunction(
            model_name=emb_config["model_name"],
        )
    else:
        raise ValueError(f"Unknown embedding provider: '{provider}'. Use 'openai' or 'sentence-transformers'.")
