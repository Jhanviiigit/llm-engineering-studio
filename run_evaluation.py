from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.indexer import Indexer
from rag.retriever import Retriever

from services.rag_service import RAGService
from evaluation.batch_evaluator import BatchEvaluator


def create_rag_service():

    embedding_model = EmbeddingModel()
    vector_store = VectorStore()

    indexer = Indexer(
        embedding_model,
        vector_store
    )

    indexer.index("data/documents/sample.txt")

    retriever = Retriever(
        embedding_model,
        vector_store
    )

    return RAGService(retriever)


def main():

    rag_service = create_rag_service()

    evaluator = BatchEvaluator(rag_service)

    experiments = [1, 3]

    experiment_results = {}

    for top_k in experiments:

        print("\n" + "=" * 60)
        print(f"Experiment: top_k = {top_k}")
        print("=" * 60)

        results = evaluator.evaluate(
            "data/evaluation/rag_test_cases.json",
            top_k=top_k
        )

        summary = evaluator.summarize(results)

        experiment_results[top_k] = summary

        print(f"\nCases evaluated : {len(results)}")

        for metric, score in summary.items():

            print(
                f"{metric.capitalize():15} : "
                f"{score:.2f}"
            )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Metric':<18}"
        f"{'top_k=1':<12}"
        f"{'top_k=3':<12}"
    )

    print("-" * 42)

    metrics = [
        "groundedness",
        "relevance",
        "completeness",
        "conciseness",
        "correctness"
    ]

    for metric in metrics:

        score_1 = experiment_results[1].get(metric, 0.0)
        score_3 = experiment_results[3].get(metric, 0.0)

        print(
            f"{metric.capitalize():<18}"
            f"{score_1:<12.2f}"
            f"{score_3:<12.2f}"
        )


if __name__ == "__main__":
    main()