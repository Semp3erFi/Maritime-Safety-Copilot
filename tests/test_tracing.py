from contextlib import contextmanager

import pytest

from backend import tracing


def test_tracing_optional_with_missing_or_partial_settings(monkeypatch):
    assert tracing.get_tracer() is None
    tracing.get_tracer.cache_clear()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-key")
    assert tracing.get_tracer() is None


def test_observations_nest_and_capture_updates(monkeypatch):
    events = []

    class FakeTracer:
        @contextmanager
        def start_as_current_observation(self, **values):
            events.append(("start", values["name"]))
            yield self
            events.append(("end", values["name"]))

        def update(self, **values):
            events.append(("update", values))

    monkeypatch.setattr(tracing, "get_tracer", lambda: FakeTracer())
    with tracing.trace("ask") as root:
        with tracing.trace("retrieval", kind="retriever"):
            pass
        tracing.update_span(root, output="answer")
    assert events == [
        ("start", "ask"),
        ("start", "retrieval"),
        ("end", "retrieval"),
        ("update", {"output": "answer"}),
        ("end", "ask"),
    ]


def test_telemetry_failure_does_not_swallow_business_error(monkeypatch):
    class BrokenTracer:
        def start_as_current_observation(self, **values):
            raise RuntimeError("tracing failed")

    monkeypatch.setattr(tracing, "get_tracer", lambda: BrokenTracer())
    with tracing.trace("works"):
        result = 42
    assert result == 42
    with pytest.raises(ValueError, match="business error"):
        with tracing.trace("fails"):
            raise ValueError("business error")


def test_installed_sdk_records_pipeline_without_network(monkeypatch):
    langfuse = pytest.importorskip("langfuse")
    from fastapi.testclient import TestClient
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from backend import llm, main

    exporter = InMemorySpanExporter()
    sdk = langfuse.Langfuse(
        public_key="pk-lf-offline-test",
        secret_key="sk-lf-offline-test",
        base_url="http://127.0.0.1:1",
        span_exporter=exporter,
    )
    monkeypatch.setattr(tracing, "get_tracer", lambda: sdk)
    monkeypatch.setattr(llm, "generate_answer", lambda _: "Demo evidence [S1].")
    with TestClient(main.app) as client:
        assert client.post("/ask", json={"question": "SOLAS IX"}).status_code == 200
        sdk.flush()
        spans = {span.name: span for span in exporter.get_finished_spans()}
        assert {"ask", "retrieval", "llm"} <= spans.keys()
        root = spans["ask"]
        for name in ("retrieval", "llm"):
            assert spans[name].parent.span_id == root.context.span_id
            assert spans[name].end_time >= spans[name].start_time
        assert any("Demo evidence" in str(value) for value in root.attributes.values())


def test_configured_but_uninstalled_sdk_is_optional(monkeypatch):
    import sys

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-offline-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-offline-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setitem(sys.modules, "langfuse", None)
    assert tracing.get_tracer() is None


@pytest.mark.parametrize("failure_stage", ["start", "update", "exit", "shutdown"])
def test_tracing_failures_do_not_break_ask_response(monkeypatch, failure_stage):
    from fastapi.testclient import TestClient

    from backend import llm, main

    class BrokenTracer:
        def start_as_current_observation(self, **values):
            if failure_stage == "start":
                raise RuntimeError("Telemetry unavailable")
            return self

        def __enter__(self):
            return self

        def update(self, **values):
            if failure_stage == "update":
                raise RuntimeError("Telemetry update failed")

        def __exit__(self, *args):
            if failure_stage == "exit":
                raise RuntimeError("Telemetry completion failed")

        def shutdown(self):
            if failure_stage == "shutdown":
                raise RuntimeError("Telemetry flush failed")

    monkeypatch.setattr(tracing, "get_tracer", lambda: BrokenTracer())
    monkeypatch.setattr(llm, "generate_answer", lambda _: "Demo evidence [S1].")
    with TestClient(main.app) as client:
        response = client.post("/ask", json={"question": "What is SOLAS IX?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "Demo evidence [S1]."
