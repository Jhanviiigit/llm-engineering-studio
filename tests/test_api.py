import pytest
from fastapi.testclient import TestClient

from api.main import app, get_services
from bootstrap import Services
from llm.client import LLMError
from rag.vector_store import VectorStore


class FakeRAGService:

    def __init__(self, evaluation=None, error=None):
        self.evaluation = evaluation
        self.error = error
        self.calls = []

    def answer_question(
        self,
        question,
        top_k=3,
        reference_answer=None,
        evaluate=True
    ):
        self.calls.append(
            {"question": question, "top_k": top_k, "evaluate": evaluate}
        )

        if self.error:
            raise self.error

        result = {
            "response": "RAG combines retrieval with generation.",
            "retrieved_context": ["RAG combines retrieval with LLMs."],
            "model": "test-model",
            "temperature": 0.7,
            "latency": 0.1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        if evaluate:
            result["evaluation"] = self.evaluation

        return result


class FakeIndexer:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def index_text(self, text, source=None):
        chunks = [text]
        added = self.vector_store.add(
            chunks,
            [[1.0, 0.0]],
            source=source,
            content_hash=text,
        )
        return len(chunks) if added else 0


def make_client(rag_service):

    vector_store = VectorStore()

    services = Services(
        indexer=FakeIndexer(vector_store),
        vector_store=vector_store,
        rag_service=rag_service,
    )

    app.dependency_overrides[get_services] = lambda: services

    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health():

    client = make_client(FakeRAGService())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "total_chunks": 0}


def test_ask_without_evaluation_by_default():

    rag_service = FakeRAGService()
    client = make_client(rag_service)

    response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "RAG combines retrieval with generation."
    assert body["evaluation"] is None
    assert rag_service.calls[0]["evaluate"] is False
    assert rag_service.calls[0]["top_k"] == 3


def test_ask_with_evaluation():

    client = make_client(
        FakeRAGService(
            evaluation={
                "groundedness": {"score": 1.0, "reason": "Supported."}
            }
        )
    )

    response = client.post(
        "/ask",
        json={"question": "What is RAG?", "evaluate": True},
    )

    body = response.json()

    assert body["evaluation"]["groundedness"]["score"] == 1.0
    assert body["evaluation_error"] is None


def test_ask_reports_evaluation_error():

    client = make_client(
        FakeRAGService(evaluation={"error": "Evaluation failed: not json"})
    )

    response = client.post(
        "/ask",
        json={"question": "What is RAG?", "evaluate": True},
    )

    body = response.json()

    assert body["evaluation"] is None
    assert body["evaluation_error"] == "Evaluation failed: not json"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "What is RAG?", "top_k": 0},
        {"question": "What is RAG?", "top_k": 11},
        {},
    ],
)
def test_ask_rejects_invalid_input(payload):

    client = make_client(FakeRAGService())

    response = client.post("/ask", json=payload)

    assert response.status_code == 422


def test_ask_llm_failure_returns_502():

    client = make_client(
        FakeRAGService(error=LLMError("service unavailable"))
    )

    response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 502
    assert "service unavailable" in response.json()["detail"]


def test_upload_document():

    client = make_client(FakeRAGService())

    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"Vector databases store embeddings.")},
    )

    assert response.status_code == 201
    assert response.json() == {
        "filename": "notes.txt",
        "chunks_added": 1,
        "already_indexed": False,
        "total_chunks": 1,
    }


def test_upload_same_document_twice_is_not_duplicated():

    client = make_client(FakeRAGService())

    files = {"file": ("notes.txt", b"Vector databases store embeddings.")}

    client.post("/documents", files=files)
    response = client.post("/documents", files=files)

    assert response.status_code == 201
    assert response.json()["already_indexed"] is True
    assert response.json()["total_chunks"] == 1


def test_upload_rejects_non_text_file():

    client = make_client(FakeRAGService())

    response = client.post(
        "/documents",
        files={"file": ("report.pdf", b"%PDF-1.7")},
    )

    assert response.status_code == 415


def test_upload_rejects_empty_file():

    client = make_client(FakeRAGService())

    response = client.post(
        "/documents",
        files={"file": ("empty.txt", b"   ")},
    )

    assert response.status_code == 400
