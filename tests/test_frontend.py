import json
from contextlib import contextmanager
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def test_chat_sources_history_and_clear(monkeypatch):
    requests = []

    def post(url, *, json, timeout):
        requests.append(json)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "answer": "A demo answer [S1].",
                "mode": "knowledge_base",
                "warnings": [],
                "sources": [
                    {
                        "citation": "S1",
                        "chunk_id": "demo:1",
                        "source": "demo.md",
                        "title": "Curated document",
                        "metadata": {"topic": "safety_management", "keywords": ["drills"]},
                        "section": "Demo section",
                        "excerpt": "Original demo excerpt.",
                        "score": 0.8,
                        "similarity": None,
                        "keyword_score": 0.8,
                    }
                ],
                "technical_details": {
                    "retrieval_method": "exact",
                    "chunks_retrieved": 1,
                    "latency_ms": 12.3,
                },
            },
        )

    @contextmanager
    def stream(method, url, **kwargs):
        assert method == "POST" and url.endswith("/ask/stream")
        result = post(url, **kwargs).json()
        events = [
            {"type": "delta", "text": "A demo "},
            {"type": "delta", "text": "answer [S1]."},
            {"type": "done", "response": result},
        ]
        yield httpx.Response(
            200,
            request=httpx.Request(method, url),
            content="\n".join(json.dumps(event) for event in events),
        )

    monkeypatch.setattr(httpx, "stream", stream)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    app.chat_input[0].set_value("SOLAS IX").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("[S1]" in item.value for item in app.markdown)
    assert any(item.value == "Original demo excerpt." for item in app.text)
    assert json.loads(app.json[0].value)["keywords"] == ["drills"]
    assert requests[0]["history"] == []
    app.chat_input[0].set_value("Explain SOLAS IX again").run()
    assert len(requests[1]["history"]) == 2
    app.button(key="try_SOLAS IX").click().run()
    assert requests[2]["question"] == "What is SOLAS IX?"
    assert len(requests[2]["history"]) == 4
    assert requests[2]["mode"] == "knowledge_base"
    app.run()
    assert len(requests) == 3  # Widget reruns must not resubmit a clicked suggestion.
    app.sidebar.button[0].click().run()
    assert not app.chat_message


def test_backend_connection_failure_is_friendly(monkeypatch):
    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "stream", fail)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.chat_input[0].set_value("SOLAS IX").run()
    assert not app.exception
    assert "Cannot reach the backend" in app.error[0].value


def test_stream_failure_clears_partial_text_and_does_not_save_history(monkeypatch):
    @contextmanager
    def stream(method, url, **kwargs):
        events = [
            {"type": "delta", "text": "Incomplete test answer"},
            {"type": "error", "detail": "Provider interrupted. Please retry."},
        ]
        yield httpx.Response(
            200,
            request=httpx.Request(method, url),
            content="\n".join(json.dumps(event) for event in events),
        )

    monkeypatch.setattr(httpx, "stream", stream)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.chat_input[0].set_value("SOLAS IX").run()
    assert not app.exception
    assert "Provider interrupted" in app.error[0].value
    assert not any("Incomplete test answer" in item.value for item in app.markdown)
    assert app.session_state["chats"]["knowledge_base"] == []


def test_modes_have_separate_histories_and_general_disables_retrieval(monkeypatch):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.radio[0].set_value("General LLM").run()
    assert not app.exception
    assert app.sidebar.toggle[0].disabled
    assert app.sidebar.slider[0].disabled
    assert app.session_state["chats"] == {"knowledge_base": [], "general": []}


def test_sidebar_upload_saves_once_and_explains_refresh(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import streamlit as st

    from backend.retrieval import load_chunks
    from frontend import uploads

    original_save = uploads.save_markdown
    monkeypatch.setattr(
        uploads, "save_markdown", lambda name, data: original_save(name, data, tmp_path)
    )
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda *args, **kwargs: SimpleNamespace(
            name="../../New note.md",
            getvalue=lambda: b"---\ntitle: Uploaded note\n---\n# Topic\nNew evidence.",
        ),
    )
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not list(tmp_path.iterdir())
    app.button(key="save_knowledge").click().run()
    assert not app.exception
    assert "next semantic/hybrid retrieval" in app.success[0].value
    assert load_chunks(tmp_path)[0].title == "Uploaded note"
    app.run()
    assert len(list(tmp_path.glob("*.md"))) == 1
