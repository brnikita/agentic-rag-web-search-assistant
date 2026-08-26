"""Local embedding model.

Thin LangChain adapter over `fastembed`, which runs the model on-device via
ONNX — no API key, no data leaving the machine, and it works offline once the
model is cached.

Wrapping fastembed directly (rather than using `langchain-community`, which is
being sunset) also lets us use its asymmetric helpers: BGE-family models are
trained with different prefixes for passages and queries, and using the right
one on each side measurably improves retrieval.
"""

from __future__ import annotations

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """Embeddings backed by a local fastembed ONNX model."""

    def __init__(self, model_name: str, *, threads: int | None = None) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, threads=threads)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()
