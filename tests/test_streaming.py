import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import llm, main, tracing
from frontend.streaming import StreamError, answer_tokens


def events(client, **payload):
    response = client.post("/ask/stream", json={"question": "What is SOLAS IX?", **payload})
    assert response.status_code == 200
    assert "application/x-ndjson" in response.headers["content-type"]
    return [json.loads(line) for line in response.text.splitlines()]


@pytest.mark.parametrize("mode", ["knowledge_base", "general"])
def test_streamed_answer_and_final_sources(monkeypatch, mode):
    async def tokens(messages):
        if mode == "knowledge_base":
            assert json.loads(messages[-1]["content"])["retrieved_excerpts"]
        yield "Demo "
        yield "answer [S1]."

    monkeypatch.setattr(llm, "stream_answer", tokens)
    with TestClient(main.app) as client:
        result = events(client, mode=mode)
    assert [e["type"] for e in result] == ["delta", "delta", "done"]
    final = result[-1]["response"]
    assert final["answer"] == "Demo answer [S1]."
    assert bool(final["sources"]) == (mode == "knowledge_base")
    if final["sources"]:
        assert final["sources"][0]["source"] == "solas_chapter_ix_summary.md"
        assert final["sources"][0]["metadata"]["document_type"] == "educational_summary"


@pytest.mark.parametrize("payload", [{"retrieval_enabled": False}, {"question": "SOLAS XII"}])
def test_no_evidence_does_not_stream_from_provider(monkeypatch, payload):
    monkeypatch.setattr(llm, "stream_answer", lambda _: pytest.fail("No provider call expected"))
    with TestClient(main.app) as client:
        result = events(client, **payload)
    assert len(result) == 1 and result[0]["type"] == "done"
    assert "insufficient" in result[0]["response"]["answer"]


def test_failed_citation_check_replaces_provisional_text(monkeypatch):
    async def tokens(messages):
        yield "Unverified answer [S99]."

    monkeypatch.setattr(llm, "stream_answer", tokens)
    with TestClient(main.app) as client:
        result = events(client)
    assert result[0]["text"] == "Unverified answer [S99]."
    assert "insufficient" in result[-1]["response"]["answer"]
    assert result[-1]["response"]["warnings"]


def test_partial_failure_has_error_event_and_no_done(monkeypatch):
    async def tokens(messages):
        yield "Partial "
        raise llm.ProviderError("Azure chat stream failed or was interrupted. Please retry.")

    monkeypatch.setattr(llm, "stream_answer", tokens)
    with TestClient(main.app) as client:
        result = events(client)
    assert [e["type"] for e in result] == ["delta", "error"]
    assert result[-1]["status_code"] == 502


def test_missing_configuration_and_validation():
    with TestClient(main.app) as client:
        assert events(client)[0]["status_code"] == 503
        assert client.post("/ask/stream", json={"question": " "}).status_code == 422


def test_generator_is_incremental_and_closes_on_disconnect(monkeypatch):
    consumed, closed = [], []

    async def tokens(messages):
        try:
            consumed.append(1)
            yield "First "
            consumed.append(2)
            yield "second [S1]."
        finally:
            closed.append(True)

    monkeypatch.setattr(llm, "stream_answer", tokens)

    async def run():
        generator = main.answer_events(main.AskRequest(question="SOLAS IX"))
        first = json.loads(await anext(generator))
        assert first == {"type": "delta", "text": "First "}
        assert consumed == [1]
        await generator.aclose()
        assert closed == [True]

    asyncio.run(run())


@pytest.mark.parametrize(
    "finish,expected_error",
    [
        ("stop", None),
        ("length", "completion budget"),
        (None, "before completion"),
        ("content_filter", "filtered"),
    ],
)
def test_azure_stream_validation_and_cleanup(monkeypatch, finish, expected_error):
    closed, settings = [], []

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def __aiter__(self):
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="First "), finish_reason=None)
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="answer."), finish_reason=finish)
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def create(self, **kwargs):
            settings.append(kwargs)
            return FakeStream()

    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "test")
    monkeypatch.setattr(llm, "client_settings", dict)
    monkeypatch.setattr(llm, "AsyncAzureOpenAI", FakeClient)

    async def collect():
        return [token async for token in llm.stream_answer([])]

    if expected_error:
        with pytest.raises(llm.ProviderError, match=expected_error):
            asyncio.run(collect())
    else:
        assert asyncio.run(collect()) == ["First ", "answer."]
    assert closed == [True, True]
    assert settings[0]["stream"] is True
    assert settings[0]["max_completion_tokens"] == 4096


@pytest.mark.parametrize("ending", [[], [{"type": "error", "detail": "Provider interrupted"}]])
def test_frontend_detects_incomplete_stream(monkeypatch, ending):
    @contextmanager
    def stream(*args, **kwargs):
        yield httpx.Response(
            200,
            request=httpx.Request("POST", "http://test/ask/stream"),
            content="\n".join(
                json.dumps(e) for e in [{"type": "delta", "text": "Partial"}, *ending]
            ),
        )

    monkeypatch.setattr(httpx, "stream", stream)
    completed = {}
    tokens = answer_tokens("http://test/ask/stream", {}, completed)
    assert next(tokens) == "Partial"
    with pytest.raises(StreamError):
        next(tokens)
    assert completed == {}


def test_streaming_records_completed_langfuse_output(monkeypatch):
    langfuse = pytest.importorskip("langfuse")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    sdk = langfuse.Langfuse(
        public_key="pk-lf-stream-test",
        secret_key="sk-lf-stream-test",
        base_url="http://127.0.0.1:1",
        span_exporter=exporter,
    )
    monkeypatch.setattr(tracing, "get_tracer", lambda: sdk)

    async def tokens(messages):
        yield "Demo "
        yield "evidence [S1]."

    monkeypatch.setattr(llm, "stream_answer", tokens)
    with TestClient(main.app) as client:
        result = events(client)
        sdk.flush()
        spans = {s.name: s for s in exporter.get_finished_spans()}
        for name in ("retrieval", "llm"):
            assert spans[name].parent.span_id == spans["ask"].context.span_id
        for name in ("ask", "llm"):
            assert any("Demo evidence [S1]." in str(v) for v in spans[name].attributes.values())
        assert result[-1]["type"] == "done"
