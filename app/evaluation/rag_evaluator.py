import json
import logging

from llm.client import LLMClient, LLMError, get_client


logger = logging.getLogger(__name__)


BASE_METRICS = [
    "groundedness",
    "relevance",
    "completeness",
    "conciseness",
]


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

    metrics = list(BASE_METRICS)

    if reference_answer:
        metrics.append("correctness")

    json_schema = json.dumps(
        {
            metric: {"score": 0.0, "reason": "..."}
            for metric in metrics
        },
        indent=2
    )

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

Return only a JSON object with exactly this structure:

{json_schema}

Scores must be between 0.0 and 1.0.
"""

    return prompt


def parse_evaluation(response_text: str) -> dict:
    """
    Parse the judge's JSON into {metric: {"score", "reason"}}.

    Different models format JSON differently (code fences, capitalised
    keys, bare numbers instead of objects), so the output is normalised
    before it is used. Raises ValueError if no scores can be found.
    """

    response_text = response_text.strip()

    # Take the outermost {...}, dropping code fences or surrounding text
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object in judge response")

    data = json.loads(response_text[start:end + 1])

    evaluation = {}

    for key, value in data.items():

        metric = key.strip().lower()

        if isinstance(value, (int, float)):
            value = {"score": value, "reason": ""}

        if isinstance(value, dict) and "score" in value:
            evaluation[metric] = {
                "score": float(value["score"]),
                "reason": str(value.get("reason", ""))
            }

    if not evaluation:
        raise ValueError("No metric scores in judge response")

    return evaluation


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

        # Reasoning models spend part of max_tokens thinking before
        # they write the JSON, so the judge needs a larger budget.
        result = client.chat(
            prompt,
            temperature=0.0,
            max_tokens=2000,
            response_format={
                "type": "json_object"
            }
        )

        return parse_evaluation(result["response"])

    # json.JSONDecodeError is a subclass of ValueError
    except (LLMError, ValueError) as e:

        logger.warning("RAG evaluation failed: %s", e)

        return {
            "error": f"Evaluation failed: {e}"
        }
