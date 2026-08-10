from app.rag.retriever import Retriever


class FakeEmbeddingModel:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def search(self, query_embedding, top_k):
        assert query_embedding == [1.0, 0.0]
        assert top_k == 2

        return [
            "RAG retrieves relevant context.",
            "RAG uses embeddings for retrieval."
        ]


def test_retriever():
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedding_model,
        vector_store
    )

    results = retriever.retrieve(
        "What is RAG?",
        top_k=2
    )

    assert len(results) == 2
    assert results[0] == "RAG retrieves relevant context."