from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Simple wrapper around SentenceTransformer to create text embeddings.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 4,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Returns a numpy array with shape (num_texts, embedding_dim).
        If normalize=True, returns L2-normalized embeddings.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        return embeddings
