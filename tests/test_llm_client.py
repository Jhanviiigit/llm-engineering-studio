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


class EmptyMessage:

    content = None


class EmptyChoice:

    message = EmptyMessage()
    finish_reason = "length"


class EmptyResponse:

    choices = [EmptyChoice()]


class EmptyCompletions:

    def create(self, **request):
        return EmptyResponse()


class EmptyChat:

    completions = EmptyCompletions()


class EmptyOpenAI:

    chat = EmptyChat()


def test_empty_response_raises_llm_error():

    client = LLMClient(api_key="test-key")
    client.client = EmptyOpenAI()

    with pytest.raises(LLMError, match="finish_reason=length"):
        client.chat("Hello")
