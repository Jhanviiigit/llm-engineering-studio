import threading

import numpy as np


class VectorStore:
    """
    In-memory vector store. Data is lost when the process exits.

    Used for tests and quick local runs. PgVectorStore has the same
    interface and keeps data in Postgres.
    """

    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.content_hashes = set()

        # The API serves requests on multiple threads; the lock stops a
        # search from reading while an upload is half-way through adding.
        self._lock = threading.Lock()

    def has_document(self, content_hash: str) -> bool:
        with self._lock:
            return content_hash in self.content_hashes

    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        source: str | None = None,
        content_hash: str | None = None,
    ) -> bool:
        """
        Store chunks and their embeddings.

        Returns False (and stores nothing) if a document with the same
        content_hash was already added.
        """

        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")

        with self._lock:
            if content_hash is not None:
                if content_hash in self.content_hashes:
                    return False

                self.content_hashes.add(content_hash)

            self.documents.extend(documents)
            self.embeddings.extend(embeddings)

        return True

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
