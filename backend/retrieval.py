"""Small, inspectable search: identifier matching, keyword overlap and cosine similarity."""

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import yaml

SearchMethod = Literal["exact", "semantic", "hybrid"]
Embedder = Callable[[list[str]], list[list[float]]]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STOPWORDS = set(
    "a an the is are of for to in on and or what which does do tell me about "
    "explain how why with can should under code section chapter".split()
)
ALIASES = {
    "international safety management code": "ism code",
    "international convention for the safety of life at sea": "solas",
    "international regulations for preventing collisions at sea": "colregs",
    "international code of safety for ships using gases or other low-flashpoint fuels": "igf code",
}


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


class KnowledgeBaseError(ValueError):
    """A local document cannot be parsed; report the filename, not its contents."""


def _json_date(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError("Front matter must use JSON-compatible values or dates")


def parse_document(path: Path) -> tuple[dict, str]:
    """Read optional YAML front matter safely, separately from the Markdown body."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next((i for i in range(1, len(lines)) if lines[i].strip() in {"---", "..."}), None)
    if end is None:
        raise KnowledgeBaseError(f"Unclosed YAML front matter in {path.name}.")
    try:
        metadata = yaml.safe_load("\n".join(lines[1:end]))
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
            raise ValueError("Expected a mapping with string keys")
        if metadata.get("title") is not None and not isinstance(metadata["title"], str):
            raise ValueError("title must be a string")
        # Preserve typed lists, mappings and scalars; dates become ISO strings for the API.
        metadata = json.loads(json.dumps(metadata, default=_json_date, allow_nan=False))
    except (yaml.YAMLError, TypeError, ValueError, RecursionError) as exc:
        raise KnowledgeBaseError(f"Invalid YAML front matter in {path.name}.") from exc
    return metadata, "\n".join(lines[end + 1 :])


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower().replace("–", "-").replace("—", "-")).strip()
    for full, short in ALIASES.items():
        text = text.replace(full, short)
    text = re.sub(r"\bsolas\s+(?:chapter\s+)?9\b", "solas ix", text)
    return text


def identifiers(text: str) -> set[str]:
    text = normalize(text)
    found = set()
    for match in re.finditer(
        r"\bsolas\s+(?:(?:chapter|regulation)\s+)?"
        r"([ivxlcdm]+(?:-\d+)?|\d+)(/\d+(?:\.\d+)*)?\b",
        text,
    ):
        found.add(f"solas {match[1]}{match[2] or ''}")
    for match in re.finditer(r"\b(?:msc|mepc|a)\.\s*\d+\(\d+\)", text):
        found.add(re.sub(r"\s+", "", match[0]))
    for match in re.finditer(r"\bism\s*(?:code\s*)?(?:section\s*|§\s*)?(\d+(?:\.\d+)*)\b", text):
        found.add(f"ism section {match[1]}")
    for match in re.finditer(
        r"\bsection\s+(\d+(?:\.\d+)*)\s+of\s+(?:the\s+)?ism(?:\s+code)?\b", text
    ):
        found.add(f"ism section {match[1]}")
    for match in re.finditer(
        r"\bism\s+(?:code\s+)?sections?\s+(\d+(?:\.\d+)*(?:(?:\s*,\s*|\s+and\s+)\d+(?:\.\d+)*)*)",
        text,
    ):
        found.update(f"ism section {number}" for number in re.findall(r"\d+(?:\.\d+)*", match[1]))
    for match in re.finditer(r"\b(?:colregs?\s+)?rule\s+(\d+)\b", text):
        found.add(f"rule {match[1]}")
    return found


def named_codes(text: str) -> set[str]:
    text = normalize(text)
    return {
        name
        for name in ("ism code", "igf code", "solas", "colregs", "marpol", "stcw")
        if re.search(rf"\b{re.escape(name)}\b", text)
    }


def route_query(query: str) -> SearchMethod:
    if identifiers(query) or named_codes(query):
        if re.search(r"\b(how|why|compare|apply|relationship|relate|practical)\b", query.lower()):
            return "hybrid"
        return "exact"
    return "semantic"


def load_chunks(directory: Path = DATA_DIR, chunk_words: int = 180) -> list[Chunk]:
    """Discover knowledge files dynamically, excluding README and explicitly marked documentation."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be positive")
    chunks = []
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.stem.casefold() == "readme":
            continue
        metadata, text = parse_document(path)
        if str(metadata.get("document_type", "")).casefold() == "documentation":
            continue
        title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
        title = (metadata.get("title") or "").strip()
        title = title or (title_match[1] if title_match else path.stem.replace("_", " "))
        section, buffer = title, []
        blocks = []
        for line in text.splitlines():
            if re.match(r"^#{1,6} ", line):
                if buffer:
                    blocks.append((section, "\n".join(buffer)))
                section, buffer = line.lstrip("# ").strip(), []
            else:
                buffer.append(line)
        blocks.append((section, "\n".join(buffer)))
        number = 0
        for heading, body in blocks:
            words = body.split()
            for start in range(0, len(words), chunk_words):
                number += 1
                chunks.append(
                    Chunk(
                        id=f"{path.stem}:{number}",
                        source=path.name,
                        title=title,
                        section=heading,
                        text=" ".join(words[start : start + chunk_words]),
                        metadata=metadata,
                    )
                )
    return chunks


def searchable_text(chunk: Chunk) -> str:
    # Metadata-only keywords/references are searchable and participate in the embedding cache key.
    metadata = json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True)
    return f"{chunk.source}\n{chunk.title}\n{chunk.section}\n{metadata}\n{chunk.text}"


def keyword_score(query: str, chunk: Chunk) -> float:
    text = searchable_text(chunk)
    requested = identifiers(query)
    matched = requested & identifiers(text)
    # Never confuse section 1 with section 10 or use a generic Code hit for a missing ID.
    if requested and not matched:
        return 0.0
    terms = set(re.findall(r"[a-z0-9]+", normalize(query))) - STOPWORDS
    doc_terms = set(re.findall(r"[a-z0-9]+", normalize(text)))
    local_terms = set(
        re.findall(r"[a-z0-9]+", normalize(f"{chunk.title}\n{chunk.section}\n{chunk.text}"))
    )
    # Shared document keywords help discover a file, while body/heading hits rank its chunks.
    overlap = (0.7 * len(terms & local_terms) + 0.3 * len(terms & doc_terms)) / max(len(terms), 1)
    if requested:
        return 0.7 * len(matched) / len(requested) + 0.3 * overlap
    if named_codes(query) & named_codes(text):
        return 0.5 + 0.5 * overlap
    return overlap


def cosine_scores(query_vector: list[float], document_vectors: list[list[float]]) -> list[float]:
    if not document_vectors:
        return []
    query = np.asarray(query_vector, dtype=float)
    docs = np.asarray(document_vectors, dtype=float)
    if query.ndim != 1 or docs.ndim != 2 or docs.shape[1] != query.size or query.size == 0:
        raise ValueError("Embedding dimensions do not match")
    if not np.isfinite(query).all() or not np.isfinite(docs).all():
        raise ValueError("Embeddings must be finite")
    denominator = np.linalg.norm(docs, axis=1) * np.linalg.norm(query)
    scores = np.divide(docs @ query, denominator, out=np.zeros(len(docs)), where=denominator > 0)
    return np.clip(scores, -1.0, 1.0).tolist()


def retrieve(
    query: str,
    chunks: list[Chunk],
    *,
    embed: Embedder | None = None,
    embed_corpus: Embedder | None = None,
    top_k: int = 3,
    min_similarity: float = 0.3,
    method: SearchMethod | None = None,
) -> tuple[SearchMethod, list[dict]]:
    method = method or route_query(query)
    if not chunks or not query.strip():
        return method, []
    lexical = [keyword_score(query, chunk) for chunk in chunks]
    semantic: list[float | None] = [None] * len(chunks)
    if method != "exact":
        if embed is None:
            raise ValueError("Semantic and hybrid search require an embedding function")
        vectors = (embed_corpus or embed)([searchable_text(chunk) for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match corpus")
        query_vectors = embed([query])
        if len(query_vectors) != 1:
            raise ValueError("Expected one query embedding")
        semantic = cosine_scores(query_vectors[0], vectors)
    results = []
    for chunk, lex, cos in zip(chunks, lexical, semantic):
        sem_match = cos is not None and cos >= min_similarity
        lex_match = lex > 0
        if method == "exact" and not lex_match:
            continue
        if method == "semantic" and not sem_match:
            continue
        if method == "hybrid" and not (lex_match or sem_match):
            continue
        score = lex if method == "exact" else cos
        if method == "hybrid":
            score = 0.6 * max(cos or 0, 0) + 0.4 * lex
        results.append(
            {
                "chunk_id": chunk.id,
                "source": chunk.source,
                "title": chunk.title,
                "metadata": chunk.metadata,
                "section": chunk.section,
                "excerpt": chunk.text,
                "score": round(score, 4),
                "similarity": round(cos, 4) if cos is not None else None,
                "keyword_score": round(lex, 4),
            }
        )
    results.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    return method, [dict(item, citation=f"S{i}") for i, item in enumerate(results[:top_k], 1)]
