from rag.indexer import Indexer


class FakeEmbeddingModel:
    def encode(self, chunks):
        return [[1.0, 0.0] for _ in chunks]


class FakeVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, documents, embeddings):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)


def test_indexer():
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    indexer = Indexer(
        embedding_model,
        vector_store
    )

    indexer.index("data/documents/sample.txt")

    assert len(vector_store.documents) > 0
    assert len(vector_store.documents) == len(vector_store.embeddings)