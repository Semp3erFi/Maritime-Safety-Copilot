from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from backend import llm


def test_embedding_cache_reuses_unchanged_content_and_invalidates_changes(monkeypatch):
    for name, value in {
        "ENDPOINT": "https://example.openai.azure.com",
        "API_VERSION": "test",
        "EMBEDDING_DEPLOYMENT": "embedding-test",
    }.items():
        monkeypatch.setenv(f"AZURE_OPENAI_{name}", value)
    calls = []

    def embed(texts):
        calls.append(texts)
        return [[1, 0] for _ in texts]

    monkeypatch.setattr(llm, "embed_texts", embed)
    llm.embed_documents(["first"])
    llm.embed_documents(["first"])
    llm.embed_documents(["edited"])
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "another-deployment")
    llm.embed_documents(["edited"])
    assert len(calls) == 3


def test_placeholder_credentials_are_rejected(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key")
    with pytest.raises(llm.ConfigurationError, match="AZURE_OPENAI_API_KEY"):
        llm.require_env("AZURE_OPENAI_API_KEY")


def test_azure_provider_error_is_sanitized(monkeypatch):
    def fail(**kwargs):
        raise APIConnectionError(
            message="private-provider-detail", request=httpx.Request("POST", "https://example.com")
        )

    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-test")
    monkeypatch.setattr(
        llm,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))),
    )
    with pytest.raises(llm.ProviderError) as error:
        llm.generate_answer([{"role": "user", "content": "test"}])
    assert "private-provider-detail" not in str(error.value)


@pytest.mark.parametrize(
    "finish_reason,content,error_message",
    [
        ("stop", "", "no text"),
        ("length", "unfinished answer", "completion budget"),
        ("length", None, "completion budget"),
    ],
)
def test_empty_and_truncated_completion_are_errors(
    monkeypatch, finish_reason, content, error_message
):
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-test")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))
        ]
    )
    monkeypatch.setattr(
        llm,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
        ),
    )
    with pytest.raises(llm.ProviderError, match=error_message):
        llm.generate_answer([{"role": "user", "content": "test"}])


def test_short_qa_has_reasoning_headroom_and_returns_completed_answer(monkeypatch):
    from backend.prompts import build_messages

    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-test")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="  A concise demo answer.  "),
                )
            ]
        )

    monkeypatch.setattr(
        llm,
        "get_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    messages = build_messages("What are the COLREG crossing rules?", [], [], "general")
    assert llm.generate_answer(messages) == "A concise demo answer."
    assert len(calls) == 1
    assert calls[0]["max_completion_tokens"] == 4096
    assert calls[0]["messages"] == messages
    assert "200 words" in messages[0]["content"]
