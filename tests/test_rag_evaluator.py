from evaluation import rag_evaluator


class FakeLLMClient:
    def chat(self, prompt, temperature=0.7):
        assert "What is RAG?" in prompt
        assert "RAG combines information retrieval" in prompt
        assert temperature == 0.0

        return {
            "response": """
{
    "groundedness": {
        "score": 1.0,
        "reason": "The answer is supported by the context."
    },
    "relevance": {
        "score": 1.0,
        "reason": "The answer directly addresses the question."
    },
    "completeness": {
        "score": 0.8,
        "reason": "The answer covers the main idea."
    },
    "conciseness": {
        "score": 1.0,
        "reason": "The answer is concise."
    }
}
"""
        }


def test_evaluate_rag(monkeypatch):

    monkeypatch.setattr(
        rag_evaluator,
        "client",
        FakeLLMClient()
    )

    result = rag_evaluator.evaluate_rag(
        question="What is RAG?",
        context=[
            "RAG combines information retrieval with large language models."
        ],
        answer="RAG combines retrieval with language models."
    )

    assert result["groundedness"]["score"] == 1.0
    assert result["relevance"]["score"] == 1.0
    assert result["completeness"]["score"] == 0.8
    assert result["conciseness"]["score"] == 1.0