from llm.client import LLMClient, get_client
from prompts.rag import build_rag_prompt
from rag.retriever import Retriever
from evaluation.rag_evaluator import evaluate_rag


class RAGService:

    def __init__(
        self,
        retriever: Retriever,
        client: LLMClient | None = None
    ):
        self.retriever = retriever
        self.client = client or get_client()

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        reference_answer: str = None,
        evaluate: bool = True
    ) -> dict:
        """
        Retrieve context, generate an answer and (optionally) evaluate it.

        evaluate=False skips the LLM-as-judge call. Evaluation doubles
        the latency and cost of a request, so a production API would
        run it asynchronously instead of in the user's request path.
        """

        context = self.retriever.retrieve(
            question,
            top_k=top_k
        )

        prompt = build_rag_prompt(
            question,
            context
        )

        result = self.client.chat(prompt)

        result["retrieved_context"] = context

        if evaluate:
            result["evaluation"] = evaluate_rag(
                question,
                context,
                result["response"],
                reference_answer=reference_answer,
                client=self.client
            )

        return result
