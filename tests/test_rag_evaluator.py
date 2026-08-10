from evaluation import rag_evaluator


class FakeLLMClient:

    def chat(
        self,
        prompt,
        temperature=0.0,
        response_format=None
    ):

        return {
            "response": """
{
    "groundedness": {
        "score": 1.0,
        "reason": "Supported by context."
    },
    "relevance": {
        "score": 1.0,
        "reason": "Relevant to the question."
    },
    "completeness": {
        "score": 1.0,
        "reason": "Contains the required information."
    },
    "conciseness": {
        "score": 1.0,
        "reason": "Concise answer."
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
    assert result["completeness"]["score"] == 1.0
    assert result["conciseness"]["score"] == 1.0