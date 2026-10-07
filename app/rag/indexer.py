from rag.document_loader import load_text_file
from rag.chunker import chunk_text
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


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

        return self.index_text(text)

    def index_text(self, text: str) -> int:
        """
        Chunk, embed and store text. Returns the number of chunks added.
        """

        chunks = chunk_text(text)

        if not chunks:
            return 0

        embeddings = self.embedding_model.encode(chunks)

        self.vector_store.add(chunks, embeddings)

        return len(chunks)
