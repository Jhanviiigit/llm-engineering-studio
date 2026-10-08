import hashlib

from rag.document_loader import load_text_file
from rag.chunker import chunk_text
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Indexer:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def index(self, file_path: str) -> int:
        text = load_text_file(file_path)

        return self.index_text(text, source=file_path)

    def index_text(self, text: str, source: str | None = None) -> int:
        """
        Chunk, embed and store text. Returns the number of chunks added.

        Documents are identified by a hash of their content, so indexing
        the same text twice is a no-op. With a persistent store this
        means restarting the app does not re-embed the whole corpus.
        """

        text_hash = content_hash(text)

        # Cheap check first, to skip embedding work for known documents
        if self.vector_store.has_document(text_hash):
            return 0

        chunks = chunk_text(text)

        if not chunks:
            return 0

        embeddings = self.embedding_model.encode(chunks)

        # add() re-checks atomically, in case of two concurrent uploads
        added = self.vector_store.add(
            chunks,
            embeddings,
            source=source,
            content_hash=text_hash,
        )

        return len(chunks) if added else 0
