from services.rag_service import RAGService


class FakeRetriever:
    def retrieve(self, question, top_k=3):
        assert question == "What is RAG?"
        assert top_k == 2

        return [
            "RAG combines retrieval with language models.",
            "RAG provides retrieved information as context."
        ]


class FakeLLMClient:
    def chat(self, prompt):
        assert "What is RAG?" in prompt
        assert "RAG combines retrieval with language models." in prompt

        return {
            "response": "RAG combines retrieval with generation.",
            "latency": 0.1,
            "model": "test-model",
            "temperature": 0.7
        }


def fake_evaluate_rag(question, context, answer, reference_answer=None):
    assert question == "What is RAG?"
    assert len(context) == 2
    assert answer == "RAG combines retrieval with generation."

    return {
        "groundedness": {
            "score": 1.0,
            "reason": "Supported by context."
        },
        "relevance": {
            "score": 1.0,
            "reason": "Answers the question."
        }
    }


def test_rag_service(monkeypatch):

    monkeypatch.setattr(
        "services.rag_service.evaluate_rag",
        fake_evaluate_rag
    )

    service = RAGService(FakeRetriever())

    service.client = FakeLLMClient()

    result = service.answer_question(
        "What is RAG?",
        top_k=2
    )

    assert result["response"] == (
        "RAG combines retrieval with generation."
    )

    assert len(result["retrieved_context"]) == 2

    assert result["evaluation"]["groundedness"]["score"] == 1.0