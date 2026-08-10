from llm.client import LLMClient
from prompts.story import build_story_prompt

client = LLMClient()


def compare_temperatures(topic: str):

    print("Entered compare_temperatures()")

    prompt = build_story_prompt(topic)

    temperatures = [0.0, 0.5]

    results = []

    for temp in temperatures:
        print(f"Calling model with temperature {temp}")

        result = client.chat(
            prompt=prompt,
            temperature=temp,
        )

        print("Received response")

        results.append(result)

    print("Returning results")

    return results