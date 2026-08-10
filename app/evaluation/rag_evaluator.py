from llm.client import LLMClient
import json


client = LLMClient()


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

    correctness_json = ""

    if reference_answer:
        correctness_json = """,
"correctness": {
    "score": 0.0,
    "reason": ""
}"""

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
    reference_answer: str = None
) -> dict:

    prompt = build_evaluation_prompt(
        question,
        context,
        answer,
        reference_answer
    )

    result = client.chat(
        prompt,
        temperature=0.0,
        response_format={
            "type": "json_object"
        }
    )

    try:

        response_text = result["response"].strip()

        if response_text.startswith("```"):
            response_text = response_text.replace("```json", "")
            response_text = response_text.replace("```", "")
            response_text = response_text.strip()

        evaluation = json.loads(response_text)

        return evaluation

    except Exception as e:

        print("\nEvaluation parsing error:", e)

        return {
            "groundedness": {
                "score": 0.0,
                "reason": "Evaluation parsing failed."
            },
            "relevance": {
                "score": 0.0,
                "reason": "Evaluation parsing failed."
            },
            "completeness": {
                "score": 0.0,
                "reason": "Evaluation parsing failed."
            },
            "conciseness": {
                "score": 0.0,
                "reason": "Evaluation parsing failed."
            }
        }