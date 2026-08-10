import numpy as np


class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.documents = []

    def add(self, documents: list[str], embeddings: list[list[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")

        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        if not self.embeddings:
            return []

        vectors = np.array(self.embeddings)
        query = np.array(query_embedding)

        similarities = np.dot(vectors, query) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.documents[i] for i in top_indices]