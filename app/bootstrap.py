from config import get_settings
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.indexer import Indexer
from rag.retriever import Retriever
from services.rag_service import RAGService


def create_rag_service(documents_path: str | None = None) -> RAGService:
    """
    Build a RAGService with all of its dependencies.

    This is the single place where components are created and wired
    together (the "composition root"). Swapping the in-memory
    VectorStore for pgvector later only requires changing this file.
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

    return RAGService(retriever)
