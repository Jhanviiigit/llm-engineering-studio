from llm.client import LLMClient
from prompts.paraphrase import build_paraphrase_prompt

client = LLMClient()


def paraphrase(text: str):

    prompt = build_paraphrase_prompt(text)

    return client.chat(prompt)