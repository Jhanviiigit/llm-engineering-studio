from llm.client import LLMClient, get_client
from prompts.story import build_story_prompt


def compare_temperatures(
    topic: str,
    temperatures: list[float] | None = None,
    client: LLMClient | None = None
) -> list[dict]:

    client = client or get_client()

    prompt = build_story_prompt(topic)

    temperatures = temperatures or [0.0, 0.5]

    results = []

    for temp in temperatures:

        result = client.chat(
            prompt=prompt,
            temperature=temp,
        )

        results.append(result)

    return results
