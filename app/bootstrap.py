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


def create_services(documents_path: str | None = None) -> Services:
    """
    Build all components and wire them together.

    This is the single place where components are created (the
    "composition root"). Swapping the in-memory VectorStore for
    pgvector later only requires changing this file.
    """

    settings = get_settings()

    embedding_model = EmbeddingModel(settings.embedding_model)
    vector_store = VectorStore()

    indexer = Indexer(
        embedding_model,
        vector_store
    )

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
