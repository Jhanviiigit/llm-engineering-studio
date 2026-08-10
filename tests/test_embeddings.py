from app.rag.embeddings import EmbeddingModel


def test_embeddings():
    model = EmbeddingModel()

    texts = [
        "RAG uses retrieved context.",
        "Large language models generate responses."
    ]

    embeddings = model.encode(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384