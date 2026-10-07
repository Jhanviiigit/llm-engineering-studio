from evaluation import rag_evaluator
from llm.client import LLMError


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


def test_evaluate_rag():

    result = rag_evaluator.evaluate_rag(
        question="What is RAG?",
        context=[
            "RAG combines information retrieval with large language models."
        ],
        answer="RAG combines retrieval with language models.",
        client=FakeLLMClient()
    )

    assert result["groundedness"]["score"] == 1.0
    assert result["relevance"]["score"] == 1.0
    assert result["completeness"]["score"] == 1.0
    assert result["conciseness"]["score"] == 1.0

class InvalidJSONClient:

    def chat(self, prompt, temperature=0.0, response_format=None):

        return {"response": "not json"}


class FailingClient:

    def chat(self, prompt, temperature=0.0, response_format=None):

        raise LLMError("service unavailable")


def test_evaluate_rag_invalid_json_returns_error():

    result = rag_evaluator.evaluate_rag(
        question="What is RAG?",
        context=["RAG combines retrieval with language models."],
        answer="RAG combines retrieval with language models.",
        client=InvalidJSONClient()
    )

    assert "error" in result
    assert "groundedness" not in result


def test_evaluate_rag_llm_failure_returns_error():

    result = rag_evaluator.evaluate_rag(
        question="What is RAG?",
        context=["RAG combines retrieval with language models."],
        answer="RAG combines retrieval with language models.",
        client=FailingClient()
    )

    assert "service unavailable" in result["error"]
