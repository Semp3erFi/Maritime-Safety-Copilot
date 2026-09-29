"""Document loading and smoke checks against the currently curated knowledge base."""

import pytest

from backend.retrieval import (
    DATA_DIR,
    KnowledgeBaseError,
    load_chunks,
    parse_document,
    retrieve,
    searchable_text,
)


def test_front_matter_is_preserved_on_each_body_chunk(tmp_path):
    path = tmp_path / "arbitrary_filename.md"
    path.write_text(
        '\ufeff---\ntitle: "Curated: entry"\nkeywords: [entry, MSC.581(110)]\n'
        "reviewed: 2026-09-10\nrevision: 2\nverified: false\n"
        "custom: {nested: [first, second]}\nsource: publisher\n---\n"
        "# Markdown title\n## Before entry\none two three four five",
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path, chunk_words=3)
    assert [c.text for c in chunks] == ["one two three", "four five"]
    for chunk in chunks:
        assert chunk.source == path.name  # YAML cannot overwrite the actual filename.
        assert chunk.title == "Curated: entry"
        assert chunk.section == "Before entry"
        assert chunk.metadata == {
            "title": "Curated: entry",
            "keywords": ["entry", "MSC.581(110)"],
            "reviewed": "2026-09-10",
            "revision": 2,
            "verified": False,
            "custom": {"nested": ["first", "second"]},
            "source": "publisher",
        }
        assert "MSC.581(110)" in searchable_text(chunk)
        assert "keywords:" not in chunk.text


def test_discovery_excludes_documentation_and_title_falls_back(tmp_path):
    (tmp_path / "ReAdMe.MD").write_text("# Guidance\nNot knowledge.", encoding="utf-8")
    (tmp_path / "setup_notes.md").write_text(
        "---\ndocument_type: documentation\n---\n# Setup\nNot knowledge.", encoding="utf-8"
    )
    (tmp_path / "unexpected.MD").write_text("No heading here.", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    chunks = load_chunks(tmp_path)
    assert {c.source: c.title for c in chunks} == {
        "unexpected.MD": "unexpected",
    }
    assert all(c.metadata == {} for c in chunks)
    (tmp_path / "added.md").write_text(
        "---\ntitle: New addition\n---\nFresh text.", encoding="utf-8"
    )
    assert {c.title for c in load_chunks(tmp_path)} == {"unexpected", "New addition"}
    (tmp_path / "unexpected.MD").unlink()
    assert "unexpected.MD" not in {c.source for c in load_chunks(tmp_path)}


@pytest.mark.parametrize(
    "front_matter",
    [
        "title: [",
        "[one, two]",
        "title: 12",
        "custom: !!python/object/apply:os.system ['bad']",
    ],
)
def test_invalid_or_unsafe_yaml_reports_filename(tmp_path, front_matter):
    path = tmp_path / "bad.md"
    path.write_text(f"---\n{front_matter}\n---\nBody.", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="bad.md"):
        load_chunks(tmp_path)


def test_unclosed_and_empty_front_matter(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("---\ntitle: Missing delimiter", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="Unclosed"):
        load_chunks(tmp_path)
    path.write_text("---\n---\n# Plain title\nBody.", encoding="utf-8")
    assert load_chunks(tmp_path)[0].title == "Plain title"
    path.write_text("---\ntitle: Only metadata\n---", encoding="utf-8")
    assert load_chunks(tmp_path) == []  # Metadata alone is not an evidence excerpt.


def test_current_corpus_uses_every_nonempty_knowledge_markdown_file():
    paths = [
        p
        for p in DATA_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".md"
        and p.stem.casefold() != "readme"
        and str(parse_document(p)[0].get("document_type", "")).casefold() != "documentation"
    ]
    chunks = load_chunks()
    expected = {p.name for p in paths if parse_document(p)[1].strip()}
    assert {c.source for c in chunks} == expected
    for path in paths:
        metadata, body = parse_document(path)
        matches = [c for c in chunks if c.source == path.name]
        assert all(c.metadata == metadata for c in matches)
        if metadata.get("title"):
            assert all(c.title == metadata["title"] for c in matches)
        assert all(c.text in " ".join(body.split()) for c in matches)


def test_solas_ix_top_source_is_curated_summary():
    chunks = load_chunks()
    assert all(c.source.casefold() != "readme.md" for c in chunks)
    method, sources = retrieve("What is SOLAS IX?", chunks)
    assert method == "exact"
    assert sources[0]["source"] == "solas_chapter_ix_summary.md"
    assert sources[0]["metadata"]["document_type"] == "educational_summary"


@pytest.mark.parametrize(
    "question,expected_title",
    [
        ("ISM Code section 8", "ISM Code - Emergency Preparedness"),
        ("IGF Code bunkering", "IGF Code Basics"),
        ("MSC.581(110)", "Enclosed Space Entry Safety"),
        ("SOLAS Regulation XI-1/7", "Enclosed Space Entry Safety"),
        ("A.1206(34)", "Port State Control Basics"),
        ("permit to work checklist", "Permit to Work System"),
        ("risk assessment residual risk", "Maritime Risk Assessment"),
        (
            "SOLAS IX Document of Compliance DOC certification",
            "SOLAS Chapter IX - Management for the Safe Operation of Ships",
        ),
    ],
)
def test_curated_keyword_retrieval(question, expected_title):
    _, sources = retrieve(question, load_chunks(), method="exact")
    assert sources[0]["title"] == expected_title
    assert sources[0]["metadata"]["document_type"] == "educational_summary"
    assert (DATA_DIR / sources[0]["source"]).is_file()


def test_semantic_pipeline_embeds_new_body_and_metadata():
    chunks = load_chunks()
    captured = []

    def embed_corpus(texts):
        captured.extend(texts)
        return [[1, 0] if '"topic": "alternative_fuels"' in text else [0, 1] for text in texts]

    method, sources = retrieve(
        "How are gas fuel transfers planned?",
        chunks,
        embed=lambda texts: [[1, 0]],
        embed_corpus=embed_corpus,
    )
    assert method == "semantic"
    assert len(captured) == len(chunks)
    assert any("Bunkering low-flashpoint fuels" in text for text in captured)
    assert all(s["title"] == "IGF Code Basics" for s in sources)
    assert all(s["similarity"] == 1 for s in sources)


def test_embeddings_refresh_after_document_or_metadata_changes(tmp_path, monkeypatch):
    from backend import llm

    for name, value in {
        "ENDPOINT": "https://example.openai.azure.com",
        "API_VERSION": "test",
        "EMBEDDING_DEPLOYMENT": "test",
    }.items():
        monkeypatch.setenv(f"AZURE_OPENAI_{name}", value)
    calls = []

    def embed(texts):
        calls.append(texts)
        return [[1, 0] for _ in texts]

    monkeypatch.setattr(llm, "embed_texts", embed)
    path = tmp_path / "custom.md"

    def refresh():
        llm.embed_documents([searchable_text(c) for c in load_chunks(tmp_path)])

    path.write_text("---\ntitle: Notes\nkeywords: [old]\n---\nBody.", encoding="utf-8")
    refresh()
    refresh()
    assert len(calls) == 1
    path.write_text("---\ntitle: Notes\nkeywords: [new]\n---\nBody.", encoding="utf-8")
    refresh()
    assert len(calls) == 2 and '"new"' in calls[-1][0]
    path.write_text("---\ntitle: Notes\nkeywords: [new]\n---\nRevised body.", encoding="utf-8")
    refresh()
    path.rename(tmp_path / "renamed.md")
    refresh()
    assert len(calls) == 4 and "renamed.md" in calls[-1][0]
    added = tmp_path / "new.md"
    added.write_text("# Added\nAnother document.", encoding="utf-8")
    refresh()
    assert len(calls) == 5 and len(calls[-1]) == 2
    added.unlink()
    # Reusing the matching earlier corpus is valid; the removed document must disappear.
    assert len(load_chunks(tmp_path)) == 1
    refresh()
    assert llm._cached_embeddings.cache_info().hits >= 2
