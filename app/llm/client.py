import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from parsers.response_parser import parse_response


# Load environment variables
load_dotenv()


class LLMClient:
    """
    Client for interacting with OpenRouter using the OpenAI-compatible SDK.
    """

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Please check your .env file."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def chat(
        self,
        prompt: str,
        model: str = "openrouter/free",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> dict:
        """
        Send a prompt to the LLM and return the response
        along with useful metadata.
        """

        start = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            end = time.perf_counter()
            latency = end - start

            parsed = parse_response(response)

            parsed["latency"] = latency
            parsed["model"] = model
            parsed["temperature"] = temperature

            return parsed

        except Exception as e:
            return {
                "response": f"Error: {e}",
                "latency": 0,
                "model": model,
                "temperature": temperature,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }