from llm.client import LLMClient, get_client
from prompts.paraphrase import build_paraphrase_prompt


def paraphrase(text: str, client: LLMClient | None = None) -> dict:

    client = client or get_client()

    prompt = build_paraphrase_prompt(text)

    return client.chat(prompt)
