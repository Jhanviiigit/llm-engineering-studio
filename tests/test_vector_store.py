import numpy as np

from app.rag.vector_store import VectorStore


def test_vector_store_search():
    store = VectorStore()

    documents = [
        "RAG retrieves relevant context.",
        "Python is a programming language.",
        "Vector databases store embeddings."
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1]
    ]

    store.add(documents, embeddings)

    query_embedding = [1.0, 0.0]

    results = store.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert results[0] == "RAG retrieves relevant context."


def test_empty_vector_store():
    store = VectorStore()

    results = store.search([1.0, 0.0])

    assert results == []

def test_add_rejects_duplicate_content_hash():
    store = VectorStore()

    assert store.add(["chunk"], [[1.0, 0.0]], content_hash="abc") is True
    assert store.add(["chunk"], [[1.0, 0.0]], content_hash="abc") is False

    assert store.count() == 1
    assert store.has_document("abc")
