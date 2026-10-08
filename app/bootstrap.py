from dataclasses import dataclass

from config import get_settings
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.indexer import Indexer
from rag.retriever import Retriever
from services.rag_service import RAGService


@dataclass
class Services:
    """
    The application's long-lived components, built once at startup.
    """

    indexer: Indexer
    vector_store: VectorStore
    rag_service: RAGService

    def close(self) -> None:
        close = getattr(self.vector_store, "close", None)

        if close:
            close()


def create_vector_store(embedding_model: EmbeddingModel):

    settings = get_settings()

    if settings.database_url:
        # Imported here so the in-memory mode works without psycopg
        from rag.pg_vector_store import PgVectorStore

        return PgVectorStore(
            settings.database_url,
            dimension=embedding_model.dimension
        )

    return VectorStore()


def create_services(documents_path: str | None = None) -> Services:
    """
    Build all components and wire them together.

    This is the single place where components are created (the
    "composition root"). The vector store is chosen by configuration:
    Postgres + pgvector when DATABASE_URL is set, in-memory otherwise.
    """

    settings = get_settings()

    embedding_model = EmbeddingModel(settings.embedding_model)
    vector_store = create_vector_store(embedding_model)

    indexer = Indexer(
        embedding_model,
        vector_store
    )

    # Skipped automatically if this document is already in the store
    indexer.index(documents_path or settings.documents_path)

    retriever = Retriever(
        embedding_model,
        vector_store
    )

    return Services(
        indexer=indexer,
        vector_store=vector_store,
        rag_service=RAGService(retriever)
    )


def create_rag_service(documents_path: str | None = None) -> RAGService:

    return create_services(documents_path).rag_service
