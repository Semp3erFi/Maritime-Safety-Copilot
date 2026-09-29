import pytest

from backend.retrieval import (
    Chunk,
    cosine_scores,
    identifiers,
    keyword_score,
    load_chunks,
    retrieve,
    route_query,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is SOLAS IX?", "exact"),
        ("SOLAS chapter 9", "exact"),
        ("ISM Code section 8", "exact"),
        ("Section 8 of the ISM Code", "exact"),
        ("International Safety Management Code", "exact"),
        ("COLREG Rule 5", "exact"),
        ("IGF Code bunkering", "exact"),
        ("MSC.581(110)", "exact"),
        ("A.1206(34)", "exact"),
        ("SOLAS Regulation XI-1/7", "exact"),
        ("How does ISM Code section 8 help a crew prepare?", "hybrid"),
        ("Compare SOLAS IX with ISM Code", "hybrid"),
        ("How can a team learn from a near miss?", "semantic"),
    ],
)
def test_routing(query, expected):
    assert route_query(query) == expected


def test_multiple_and_reverse_identifiers():
    assert identifiers("ISM Code sections 8 and 9") == {"ism section 8", "ism section 9"}
    assert identifiers("section 8.1 of the ISM Code") == {"ism section 8.1"}
    assert identifiers("SOLAS chapter IX") == identifiers("SOLAS 9")


def test_exact_search_needs_no_embeddings_and_returns_metadata(tmp_path):
    (tmp_path / "custom_name.md").write_text(
        "---\ntitle: Emergency planning\nkeywords: [ISM Section 8]\n---\n"
        "# Different heading\nAn educational summary of preparedness.",
        encoding="utf-8",
    )
    method, sources = retrieve("ISM Code section 8", load_chunks(tmp_path))
    assert method == "exact"
    assert sources[0]["source"] == "custom_name.md"
    assert sources[0]["title"] == "Emergency planning"
    assert sources[0]["metadata"]["keywords"] == ["ISM Section 8"]
    assert sources[0]["citation"] == "S1"
    assert "educational summary" in sources[0]["excerpt"]
    assert sources[0]["similarity"] is None
    assert sources[0]["score"] > 0


def test_missing_identifier_does_not_match_a_different_section(tmp_path):
    (tmp_path / "sections.md").write_text(
        "# Test notes\n## ISM Code section 8\nPreparedness.\n## ISM Code section 10\nMaintenance.",
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path)
    assert retrieve("ISM Code section 1", chunks)[1] == []
    assert retrieve("ISM Code section 8.1", chunks)[1] == []
    assert retrieve("SOLAS XII", chunks)[1] == []
    assert retrieve("ISM Code section 10", chunks)[1][0]["section"] == "ISM Code section 10"
    assert identifiers("SOLAS XI-1/7") != identifiers("SOLAS XI-1/8")


def test_keyword_matching_for_named_regulation():
    chunks = load_chunks()
    assert max(keyword_score("International Safety Management Code", c) for c in chunks) > 0.5


def test_cosine_is_scale_independent_and_handles_zero_vectors():
    assert cosine_scores([2, 0], [[10, 0], [0, 7], [-1, 0], [0, 0]]) == [1, 0, -1, 0]
    assert cosine_scores([0, 0], [[1, 2]]) == [0]


@pytest.mark.parametrize("query,docs", [([1, 2], [[1]]), ([float("nan")], [[1]])])
def test_reject_invalid_embeddings(query, docs):
    with pytest.raises(ValueError):
        cosine_scores(query, docs)


def sample_chunks():
    return [
        Chunk("a", "a.md", "Learning", "ISM Code section 9", "Near-miss review"),
        Chunk("b", "b.md", "Maintenance", "ISM Code section 10", "Equipment checks"),
    ]


def test_semantic_ranking_threshold_and_top_k():
    def embed(texts):
        return [[1, 0] for _ in texts]

    def corpus(texts):
        return [[1, 0], [0, 1]]

    method, results = retrieve(
        "learning from an incident", sample_chunks(), embed=embed, embed_corpus=corpus, top_k=1
    )
    assert method == "semantic"
    assert len(results) == 1 and results[0]["chunk_id"] == "a"
    assert results[0]["similarity"] == 1
    assert (
        retrieve("unrelated", sample_chunks(), embed=lambda texts: [[0, 0] for _ in texts])[1] == []
    )


def test_hybrid_combines_scores_and_preserves_exact_evidence():
    method, results = retrieve(
        "How does ISM Code section 9 support learning?",
        sample_chunks(),
        embed=lambda texts: [[1, 0] for _ in texts],
        embed_corpus=lambda texts: [[0, 1], [-1, 0]],
    )
    assert method == "hybrid"
    assert [item["chunk_id"] for item in results] == ["a"]
    assert results[0]["score"] == pytest.approx(0.4 * results[0]["keyword_score"], abs=0.0001)


def test_empty_corpus_and_empty_question_never_call_embedder():
    def fail(texts):
        pytest.fail("Embeddings should not be requested")

    assert retrieve("question", [], embed=fail)[1] == []
    assert retrieve(" ", sample_chunks(), embed=fail)[1] == []


def test_chunk_boundaries_metadata_and_content_changes(tmp_path):
    path = tmp_path / "demo.md"
    path.write_text("# Title\n## ISM Code section 8\none two three four five", encoding="utf-8")
    chunks = load_chunks(tmp_path, chunk_words=3)
    assert [c.text for c in chunks] == ["one two three", "four five"]
    assert all(c.section == "ISM Code section 8" and c.title == "Title" for c in chunks)
    assert len({c.id for c in chunks}) == 2
    path.write_text("# Changed\nnew note", encoding="utf-8")
    assert load_chunks(tmp_path)[0].text == "new note"


def test_corpus_embedding_count_must_match():
    with pytest.raises(ValueError, match="count"):
        retrieve("learning", sample_chunks(), embed=lambda texts: [[1, 0]])
