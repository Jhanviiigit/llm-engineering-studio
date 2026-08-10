from services.rag_service import RAGService
from evaluation.case_loader import load_test_cases


class BatchEvaluator:

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def evaluate(self, file_path: str,top_k: int = 3 ) -> list[dict]:

        test_cases = load_test_cases(file_path)

        results = []

        for case in test_cases:

            result = self.rag_service.answer_question(
                case["question"],
                top_k=top_k,
                reference_answer=case["reference_answer"]
            )

            results.append({
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "answer": result["response"],
                "evaluation": result["evaluation"]
            })

        return results

    def summarize(self, results: list[dict]) -> dict:

        metrics = {}

        for result in results:

            evaluation = result["evaluation"]

            for metric, details in evaluation.items():

                if not isinstance(details, dict):
                    continue

                if metric not in metrics:
                    metrics[metric] = []

                metrics[metric].append(details["score"])

        summary = {}

        for metric, scores in metrics.items():

            summary[metric] = sum(scores) / len(scores)

        return summary