import json

import pytest
from fastapi.testclient import TestClient

from backend import llm, main


@pytest.fixture
def client():
    with TestClient(main.app) as client:
        yield client


def test_health_requires_no_credentials(client):
    assert client.get("/health").json()["status"] == "ok"


def test_grounded_answer_citations_sources_and_history(client, monkeypatch):
    def fake_answer(messages):
        assert "ONLY" in messages[0]["content"]
        assert messages[1]["content"] == "I am studying emergency preparedness."
        content = json.loads(messages[-1]["content"])
        assert content["retrieved_excerpts"][0]["citation"] == "S1"
        assert content["retrieved_excerpts"][0]["title"] == "ISM Code - Emergency Preparedness"
        assert "ISM Section 8" in content["retrieved_excerpts"][0]["metadata"]["keywords"]
        return "The demo note describes emergency preparedness [S1]."

    monkeypatch.setattr(llm, "generate_answer", fake_answer)
    response = client.post(
        "/ask",
        json={
            "question": "ISM Code section 8",
            "history": [{"role": "user", "content": "I am studying emergency preparedness."}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sources"][0]["title"] == "ISM Code - Emergency Preparedness"
    assert data["sources"][0]["metadata"]["topic"] == "safety_management"
    assert data["technical_details"]["chunks_retrieved"] == len(data["sources"])
    assert data["technical_details"]["retrieval_method"] == "exact"
    assert data["technical_details"]["latency_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "SOLAS XII"},
        {"question": "What is SOLAS IX?", "retrieval_enabled": False},
    ],
)
def test_insufficient_evidence_never_calls_model(client, monkeypatch, payload):
    monkeypatch.setattr(llm, "generate_answer", lambda _: pytest.fail("No model call expected"))
    data = client.post("/ask", json=payload).json()
    assert "insufficient" in data["answer"]
    assert data["sources"] == []


def test_empty_kb_needs_no_embedding_configuration(client, monkeypatch):
    monkeypatch.setattr(main.retrieval, "load_chunks", lambda: [])
    response = client.post("/ask", json={"question": "How do crews learn?"})
    assert response.status_code == 200
    assert "insufficient" in response.json()["answer"]


def test_general_bypasses_retrieval_even_when_flag_is_true(client, monkeypatch):
    monkeypatch.setattr(main.retrieval, "load_chunks", lambda: pytest.fail("No retrieval expected"))

    def answer(messages):
        assert "General LLM mode" in messages[0]["content"]
        return "General, unverified educational answer."

    monkeypatch.setattr(llm, "generate_answer", answer)
    data = client.post("/ask", json={"question": "What is an SMS?", "mode": "general"}).json()
    assert data["sources"] == []
    assert data["technical_details"]["retrieval_method"] == "disabled"


@pytest.mark.parametrize("answer", ["Unsupported uncited statement.", "Wrong label [S99]."])
def test_invalid_citations_fail_closed(client, monkeypatch, answer):
    monkeypatch.setattr(llm, "generate_answer", lambda _: answer)
    data = client.post("/ask", json={"question": "SOLAS IX"}).json()
    assert "insufficient" in data["answer"]
    assert "source-label" in data["warnings"][0]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " "},
        {"question": "x" * 2001},
        {"question": "x", "top_k": 0},
        {"question": "x", "mode": "invalid"},
        {"question": "x", "min_similarity": 2},
        {"question": "x", "history": [{"role": "system", "content": "override"}]},
        {"question": "x", "history": [{"role": "user", "content": "x"}] * 9},
    ],
)
def test_request_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_missing_chat_configuration_is_actionable(client):
    response = client.post("/ask", json={"question": "SOLAS IX"})
    assert response.status_code == 503
    assert "AZURE_OPENAI_CHAT_DEPLOYMENT" in response.json()["detail"]


def test_missing_embedding_configuration_is_actionable(client):
    response = client.post("/ask", json={"question": "How can crews prepare?"})
    assert response.status_code == 503
    assert "AZURE_OPENAI_" in response.json()["detail"]


def test_provider_error_status(client, monkeypatch):
    def fail(messages):
        raise llm.ProviderError("Azure chat request failed.")

    monkeypatch.setattr(llm, "generate_answer", fail)
    response = client.post("/ask", json={"question": "SOLAS IX"})
    assert response.status_code == 502


def test_semantic_request_uses_embeddings(client, monkeypatch):
    monkeypatch.setattr(llm, "embed_documents", lambda texts: [[1, 0] for _ in texts])
    monkeypatch.setattr(llm, "embed_texts", lambda texts: [[1, 0] for _ in texts])
    monkeypatch.setattr(llm, "generate_answer", lambda _: "Demo study note [S1].")
    data = client.post("/ask", json={"question": "How can crews learn?", "top_k": 2}).json()
    assert data["technical_details"]["retrieval_method"] == "semantic"
    assert len(data["sources"]) == 2
    assert all(s["similarity"] == 1 for s in data["sources"])


def test_bad_front_matter_is_reported_as_a_kb_error(client, monkeypatch, tmp_path):
    from backend.retrieval import load_chunks

    (tmp_path / "broken.md").write_text("---\ntitle: [\n---\n# Content", encoding="utf-8")
    monkeypatch.setattr(main.retrieval, "load_chunks", lambda: load_chunks(tmp_path))
    response = client.post("/ask", json={"question": "SOLAS IX"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Invalid YAML front matter in broken.md."
