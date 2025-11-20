from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss

from src.ingestion.chunking import split_into_chunks
from src.indexing.embeddings import EmbeddingModel


class VectorIndex:
    """
    Stores text chunks, builds a FAISS index, and performs similarity search.

    - We use L2-normalized embeddings.
    - FAISS IndexFlatIP (inner product) behaves like cosine similarity.
    """

    def __init__(self):
        self.text_chunks: List[Dict] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embedding_model = EmbeddingModel()

    def add_documents(self, docs: List[Tuple[str, str]]) -> None:
        """
        Add a list of documents to the index.

        docs: list of (doc_id, text) tuples.
        """
        all_chunk_texts: List[str] = []

        self.text_chunks = []
        self.index = None

        for doc_id, text in docs:
            chunks = split_into_chunks(text)
            for chunk_id, chunk_text in enumerate(chunks):
                self.text_chunks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )
                all_chunk_texts.append(chunk_text)

        if not all_chunk_texts:
            return

        # Embeddings normalized directly via EmbeddingModel
        embeddings = self.embedding_model.encode(
            all_chunk_texts,
            normalize=True,
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for the most similar chunks to the query.

        Returns a list of dictionaries with:
        - score  -> cosine-like similarity (higher is better, ~0 to 1)
        - text
        - doc_id
        - chunk_id
        """
        if self.index is None or not self.text_chunks:
            return []

        query_embedding = self.embedding_model.encode([query], normalize=True)

        # Limit top_k to available chunks
        top_k = max(1, min(top_k, len(self.text_chunks)))

        similarities, indices = self.index.search(query_embedding, top_k)

        results: List[Dict] = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.text_chunks):
                continue

            meta = self.text_chunks[idx]
            score = float(sim)
            # Clamp score to [0, 1] for nicer display
            score = max(0.0, min(1.0, score))

            results.append(
                {
                    "score": score,
                    "text": meta["text"],
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                }
            )

        return results
