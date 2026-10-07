import pytest

from llm.client import LLMClient, LLMError


class FakeCompletions:

    def create(self, **request):
        raise RuntimeError("connection refused")


class FakeChat:

    completions = FakeCompletions()


class FakeOpenAI:

    chat = FakeChat()


def test_missing_api_key_raises(monkeypatch):

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError):
        LLMClient()


def test_failed_request_raises_llm_error():

    client = LLMClient(api_key="test-key")
    client.client = FakeOpenAI()

    with pytest.raises(LLMError, match="connection refused"):
        client.chat("Hello")
