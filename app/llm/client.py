import time
from functools import lru_cache

from openai import OpenAI

from config import get_settings
from parsers.response_parser import parse_response


class LLMError(Exception):
    """
    Raised when a call to the LLM fails.

    Raising (instead of returning the error as response text) lets
    callers tell failures apart from real answers, so errors can be
    shown, logged, retried or alerted on.
    """


class LLMClient:
    """
    Client for interacting with OpenRouter using the OpenAI-compatible SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ):
        settings = get_settings()

        api_key = api_key or settings.openrouter_api_key

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Please check your .env file."
            )

        self.default_model = default_model or settings.llm_model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or settings.llm_base_url,
        )

    def chat(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        response_format: dict | None = None,
    ) -> dict:
        """
        Send a prompt to the LLM and return the response
        along with useful metadata.

        response_format can be used to request structured output,
        such as a JSON object.
        """

        model = model or self.default_model

        request = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add structured output configuration only when requested
        if response_format is not None:
            request["response_format"] = response_format

        start = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                **request
            )
        except Exception as e:
            raise LLMError(f"LLM request failed: {e}") from e

        latency = time.perf_counter() - start

        choice = response.choices[0]

        # Reasoning models can spend the whole max_tokens budget
        # "thinking" and return no answer text at all.
        if not choice.message.content:
            raise LLMError(
                "LLM returned an empty response "
                f"(finish_reason={choice.finish_reason}). "
                "Try a higher max_tokens or a different model."
            )

        parsed = parse_response(response)

        parsed["latency"] = latency
        # Routers (e.g. openrouter/free) may serve a different model
        # than the one requested - record the one that actually answered.
        parsed["model"] = getattr(response, "model", None) or model
        parsed["temperature"] = temperature

        return parsed


@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    """
    Return a shared LLMClient, created on first use.

    Creating the client lazily (rather than at import time) means
    modules can be imported - e.g. by tests - without an API key.
    """

    return LLMClient()
