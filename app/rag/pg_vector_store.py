import hashlib

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT,
    content_hash TEXT NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({dimension}) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

-- HNSW: approximate nearest-neighbour index. Finds similar vectors in
-- roughly log(N) time instead of comparing against every chunk.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


class PgVectorStore:
    """
    Vector store backed by Postgres + pgvector.

    Same interface as the in-memory VectorStore, but data survives
    restarts and is shared by every API instance that connects to the
    same database - which is what lets the API scale horizontally.
    """

    def __init__(self, database_url: str, dimension: int):
        self.dimension = dimension

        # The extension must exist before register_vector can find the
        # vector type, so create it on a plain connection first.
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # A pool reuses open connections instead of opening a new one per
        # request (each new Postgres connection costs a few milliseconds
        # and server memory).
        self.pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=10,
            configure=register_vector,
            open=True,
        )

        # Fail fast at startup if the database is unreachable
        self.pool.wait(timeout=10)

        self._create_schema()

    def _create_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(SCHEMA.format(dimension=self.dimension))

            row = conn.execute(
                """
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
                """
            ).fetchone()

        if row and row[0] != self.dimension:
            raise RuntimeError(
                f"The chunks table stores {row[0]}-dimensional embeddings "
                f"but the embedding model produces {self.dimension}. "
                "Changing the embedding model requires re-indexing into a "
                "new table or database."
            )

    def has_document(self, content_hash: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE content_hash = %s",
                (content_hash,),
            ).fetchone()

        return row is not None

    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        source: str | None = None,
        content_hash: str | None = None,
    ) -> bool:
        """
        Store a document's chunks in a single transaction.

        Returns False (and stores nothing) if a document with the same
        content_hash already exists. The UNIQUE constraint makes this
        safe even if two uploads of the same file arrive at once.
        """

        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")

        if content_hash is None:
            content_hash = hashlib.sha256(
                "\n".join(documents).encode("utf-8")
            ).hexdigest()

        with self.pool.connection() as conn, conn.transaction():

            row = conn.execute(
                """
                INSERT INTO documents (source, content_hash)
                VALUES (%s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                (source, content_hash),
            ).fetchone()

            if row is None:
                return False

            document_id = row[0]

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (document_id, i, chunk, np.array(embedding))
                        for i, (chunk, embedding) in enumerate(
                            zip(documents, embeddings)
                        )
                    ],
                )

        return True

    def count(self) -> int:
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        # <=> is pgvector's cosine distance operator (1 - cosine
        # similarity), so ascending order = most similar first.
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT content FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (np.array(query_embedding), top_k),
            ).fetchall()

        return [row[0] for row in rows]

    def close(self) -> None:
        self.pool.close()
