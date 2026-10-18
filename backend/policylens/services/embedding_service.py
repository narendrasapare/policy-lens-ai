from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot generate embeddings without text.")

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        embeddings = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding model returned invalid vectors.")

        return np.ascontiguousarray(vectors)

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        return self.encode_documents([query])[0]