from llm.client import LLMClient, get_client
from prompts.translation import build_translation_prompt


def translate(
    text: str,
    language: str,
    client: LLMClient | None = None
) -> dict:

    client = client or get_client()

    prompt = build_translation_prompt(text, language)

    return client.chat(prompt)
