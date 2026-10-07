from evaluation.batch_evaluator import BatchEvaluator


class FakeRAGService:

    def answer_question(
        self,
        question,
        top_k=3,
        reference_answer=None
    ):

        assert reference_answer is not None

        return {
            "response": f"Answer to: {question}",
            "evaluation": {
                "groundedness": {
                    "score": 1.0,
                    "reason": "Supported by context."
                },
                "relevance": {
                    "score": 1.0,
                    "reason": "Relevant."
                },
                "completeness": {
                    "score": 1.0,
                    "reason": "Complete."
                },
                "conciseness": {
                    "score": 1.0,
                    "reason": "Concise."
                },
                "correctness": {
                    "score": 1.0,
                    "reason": "Correct."
                }
            }
        }


def test_batch_evaluator():

    evaluator = BatchEvaluator(
        FakeRAGService()
    )

    results = evaluator.evaluate(
        "data/evaluation/rag_test_cases.json"
    )

    assert len(results) == 5
    assert results[0]["question"] == "What is RAG?"
    assert "reference_answer" in results[0]
    assert "evaluation" in results[0]


def test_batch_evaluator_summary():

    evaluator = BatchEvaluator(
        FakeRAGService()
    )

    results = evaluator.evaluate(
        "data/evaluation/rag_test_cases.json"
    )

    summary = evaluator.summarize(results)

    assert summary["groundedness"] == 1.0
    assert summary["relevance"] == 1.0
    assert summary["completeness"] == 1.0
    assert summary["conciseness"] == 1.0
    assert summary["correctness"] == 1.0


def test_batch_evaluator_top_k():

    evaluator = BatchEvaluator(
        FakeRAGService()
    )

    results = evaluator.evaluate(
        "data/evaluation/rag_test_cases.json",
        top_k=1
    )

    assert len(results) == 5

def test_batch_evaluator_summary_skips_failed_evaluations():

    evaluator = BatchEvaluator(
        FakeRAGService()
    )

    results = [
        {"evaluation": {"groundedness": {"score": 1.0, "reason": ""}}},
        {"evaluation": {"error": "Evaluation failed: not json"}},
        {"evaluation": {"groundedness": {"score": 0.5, "reason": ""}}},
    ]

    summary = evaluator.summarize(results)

    assert summary["groundedness"] == 0.75
