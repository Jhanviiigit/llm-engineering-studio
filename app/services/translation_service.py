from llm.client import LLMClient
from prompts.translation import build_translation_prompt


client = LLMClient()


def translate(text: str, language: str) -> str:

    prompt = build_translation_prompt(text, language)

    return client.chat(prompt)