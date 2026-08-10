def build_rag_prompt(question: str, context: list[str]) -> str:
    """
    Build a prompt that instructs the LLM to answer
    using only the retrieved context.
    """

    formatted_context = "\n\n".join(context)

    return f"""
You are a helpful assistant that answers questions using the provided context.

Use only the information contained in the context to answer the question.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."

Context:
{formatted_context}

Question:
{question}

Answer:
""".strip()