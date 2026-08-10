from utils.file_manager import save_experiment
from services.translation_service import translate
from services.temperature_services import compare_temperatures
from services.paraphrase_service import paraphrase

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.indexer import Indexer
from rag.retriever import Retriever
from services.rag_service import RAGService


def display_menu():
    print("\n" + "=" * 45)
    print("         LLM Engineering Studio")
    print("=" * 45)
    print("1. Translate")
    print("2. Paraphrase")
    print("3. Temperature Experiment")
    print("4. RAG Question Answering")
    print("5. Exit")


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

    while True:

        display_menu()

        choice = input("\nSelect an option: ")

        if choice == "1":

            text = input("\nEnter text: ")
            language = input("Translate to: ")

            result = translate(text, language)

            prompt = f"Translate '{text}' to {language}"

            filepath = save_experiment(result, prompt)

            print("\n" + "=" * 45)
            print("Translation Complete")
            print("=" * 45)

            print(f"Model              : {result['model']}")
            print(f"Temperature        : {result['temperature']}")
            print(f"Latency            : {result['latency']:.2f} seconds")
            print(f"Prompt Tokens      : {result['prompt_tokens']}")
            print(f"Completion Tokens  : {result['completion_tokens']}")
            print(f"Total Tokens       : {result['total_tokens']}")

            print("\nTranslation\n")
            print(result["response"])

            print(f"\nExperiment saved to: {filepath}")

        elif choice == "2":

            text = input("\nEnter text: ")

            result = paraphrase(text)

            prompt = f"Paraphrase: {text}"

            filepath = save_experiment(result, prompt)

            print("\n" + "=" * 45)
            print("Paraphrase Complete")
            print("=" * 45)

            print(f"Model              : {result['model']}")
            print(f"Temperature        : {result['temperature']}")
            print(f"Latency            : {result['latency']:.2f} seconds")
            print(f"Prompt Tokens      : {result['prompt_tokens']}")
            print(f"Completion Tokens  : {result['completion_tokens']}")
            print(f"Total Tokens       : {result['total_tokens']}")

            print("\nParaphrased Text\n")
            print(result["response"])

            print(f"\nExperiment saved to: {filepath}")

        elif choice == "3":

            topic = input("\nEnter a topic: ")

            results = compare_temperatures(topic)

            print("\n" + "=" * 60)
            print("Temperature Comparison")
            print("=" * 60)

            for result in results:

                print(f"\nTemperature : {result['temperature']}")
                print(f"Latency     : {result['latency']:.2f} seconds")
                print(f"Tokens      : {result['total_tokens']}")

                print("\nResponse:\n")
                print(result["response"])

            print("\n" + "-" * 60)

        elif choice == "4":

            question = input("\nAsk a question about the documents: ")

            result = rag_service.answer_question(question)
            prompt = f"Answer the question using the provided documents: {question}"

            filepath = save_experiment(
            result,
            prompt,
            context=result["retrieved_context"],
            evaluation=result["evaluation"]
        )

            print("\n" + "=" * 60)
            print("RAG Answer")
            print("=" * 60)

            print(f"\nModel              : {result['model']}")
            print(f"Temperature        : {result['temperature']}")
            print(f"Latency            : {result['latency']:.2f} seconds")
            print(f"Prompt Tokens      : {result['prompt_tokens']}")
            print(f"Completion Tokens  : {result['completion_tokens']}")
            print(f"Total Tokens       : {result['total_tokens']}")

            print("\nRetrieved Context\n")

            for i, context in enumerate(
                result["retrieved_context"],
                start=1
            ):
                print(f"[{i}] {context}")

            print("\nAnswer\n")
            print(result["response"])

            print("\nEvaluation")
            print("-" * 60)

            evaluation = result["evaluation"]

            for metric, details in evaluation.items():

                if isinstance(details, dict):
                    print(f"\n{metric.capitalize()} : {details['score']:.2f}")
                    print(f"Reason           : {details['reason']}")
            print(f"\nExperiment saved to: {filepath}")

        elif choice == "5":

            print("\nGoodbye!")
            break

        else:

            print("\nInvalid choice.")


if __name__ == "__main__":
    main()