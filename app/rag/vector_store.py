import threading

import numpy as np


class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.documents = []

        # The API serves requests on multiple threads; the lock stops a
        # search from reading while an upload is half-way through adding.
        self._lock = threading.Lock()

    def add(self, documents: list[str], embeddings: list[list[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")

        with self._lock:
            self.documents.extend(documents)
            self.embeddings.extend(embeddings)

    def count(self) -> int:
        return len(self.documents)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        with self._lock:
            documents = list(self.documents)
            vectors = np.array(self.embeddings)

        if not documents:
            return []

        query = np.array(query_embedding)

        similarities = np.dot(vectors, query) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [documents[i] for i in top_indices]
