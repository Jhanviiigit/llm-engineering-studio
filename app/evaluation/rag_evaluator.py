import json
import logging

from llm.client import LLMClient, LLMError, get_client


logger = logging.getLogger(__name__)


def build_evaluation_prompt(
    question: str,
    context: list[str],
    answer: str,
    reference_answer: str = None
) -> str:

    formatted_context = "\n\n".join(context)

    reference_section = ""

    if reference_answer:
        reference_section = f"""
Reference Answer:
{reference_answer}
"""

    correctness_instruction = ""

    if reference_answer:
        correctness_instruction = """
5. Correctness:
Does the generated answer correctly convey the information in the reference answer?
"""

    prompt = f"""
Evaluate the quality of the following RAG response.

Question:
{question}

Retrieved Context:
{formatted_context}

Generated Answer:
{answer}

{reference_section}

Evaluate these dimensions:

1. Groundedness:
   Is the answer supported by the retrieved context?

2. Relevance:
   Does the answer directly address the question?

3. Completeness:
   Does the answer include the important information needed to answer the question?

4. Conciseness:
   Is the answer clear and free of unnecessary information?

{correctness_instruction}

Return a JSON object containing the evaluation.

Scores must be between 0.0 and 1.0.
"""

    return prompt


def evaluate_rag(
    question: str,
    context: list[str],
    answer: str,
    reference_answer: str = None,
    client: LLMClient | None = None
) -> dict:
    """
    Score a RAG answer using an LLM as the judge.

    If the judge call or its JSON output fails, an {"error": ...} dict
    is returned instead of metric scores. Returning 0.0 scores on
    failure would silently drag down benchmark averages, making a
    parsing bug look like a quality regression.
    """

    client = client or get_client()

    prompt = build_evaluation_prompt(
        question,
        context,
        answer,
        reference_answer
    )

    try:

        result = client.chat(
            prompt,
            temperature=0.0,
            response_format={
                "type": "json_object"
            }
        )

        response_text = result["response"].strip()

        if response_text.startswith("```"):
            response_text = response_text.replace("```json", "")
            response_text = response_text.replace("```", "")
            response_text = response_text.strip()

        return json.loads(response_text)

    except (LLMError, json.JSONDecodeError) as e:

        logger.warning("RAG evaluation failed: %s", e)

        return {
            "error": f"Evaluation failed: {e}"
        }
