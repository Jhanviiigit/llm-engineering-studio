from rag.indexer import Indexer


class FakeEmbeddingModel:
    def encode(self, chunks):
        return [[1.0, 0.0] for _ in chunks]


class FakeVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

        self.content_hashes = set()

    def has_document(self, content_hash):
        return content_hash in self.content_hashes

    def add(self, documents, embeddings, source=None, content_hash=None):
        self.content_hashes.add(content_hash)
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        return True


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

def test_indexer_skips_already_indexed_text():
    vector_store = FakeVectorStore()

    indexer = Indexer(
        FakeEmbeddingModel(),
        vector_store
    )

    first = indexer.index_text("RAG retrieves relevant context.")
    second = indexer.index_text("RAG retrieves relevant context.")

    assert first == 1
    assert second == 0
    assert len(vector_store.documents) == 1
