"""
Integration tests against a real Postgres + pgvector database.

Skipped unless TEST_DATABASE_URL is set. To run them:

    docker compose up -d
    $env:TEST_DATABASE_URL="postgresql://studio:studio@localhost:5432/studio_test"
    python -m pytest tests/test_pg_vector_store.py

These tests delete all data in the database they connect to, so they
refuse to run against any database whose name does not end in "_test".
"""

import os

import pytest


DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL not set",
)


def database_name(url):
    return url.rsplit("/", 1)[-1].split("?", 1)[0]


@pytest.fixture
def store():

    from rag.pg_vector_store import PgVectorStore

    if not database_name(DATABASE_URL).endswith("_test"):
        pytest.fail("TEST_DATABASE_URL must point to a *_test database")

    store = PgVectorStore(DATABASE_URL, dimension=2)

    with store.pool.connection() as conn:
        conn.execute("TRUNCATE documents, chunks RESTART IDENTITY CASCADE")

    yield store

    store.close()


def test_search_returns_most_similar_first(store):

    store.add(
        [
            "RAG retrieves relevant context.",
            "Python is a programming language.",
            "Vector databases store embeddings.",
        ],
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        source="test",
        content_hash="doc-1",
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert results == [
        "RAG retrieves relevant context.",
        "Vector databases store embeddings.",
    ]


def test_duplicate_document_is_not_added(store):

    assert store.add(["chunk"], [[1.0, 0.0]], content_hash="doc-1") is True
    assert store.add(["chunk"], [[1.0, 0.0]], content_hash="doc-1") is False

    assert store.count() == 1
    assert store.has_document("doc-1")


def test_data_survives_reconnect(store):

    from rag.pg_vector_store import PgVectorStore

    store.add(["persisted chunk"], [[1.0, 0.0]], content_hash="doc-1")

    reconnected = PgVectorStore(DATABASE_URL, dimension=2)

    try:
        assert reconnected.count() == 1
        assert reconnected.search([1.0, 0.0], top_k=1) == ["persisted chunk"]
    finally:
        reconnected.close()


def test_dimension_mismatch_is_reported(store):

    from rag.pg_vector_store import PgVectorStore

    with pytest.raises(RuntimeError, match="2-dimensional"):
        PgVectorStore(DATABASE_URL, dimension=384)
