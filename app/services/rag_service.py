from llm.client import LLMClient
from prompts.rag import build_rag_prompt
from rag.retriever import Retriever
from evaluation.rag_evaluator import evaluate_rag


class RAGService:

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.client = LLMClient()

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        reference_answer: str = None
    ) -> dict:

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

        evaluation = evaluate_rag(
            question,
            context,
            result["response"],
            reference_answer=reference_answer
        )

        result["evaluation"] = evaluation

        return result