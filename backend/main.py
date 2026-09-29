"""Run from Day18: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000."""

import asyncio
import json
import os
import re
from contextlib import aclosing, asynccontextmanager
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend import llm, retrieval
from backend.prompts import INSUFFICIENT_EVIDENCE, build_messages
from backend.tracing import shutdown_tracing, trace, update_span


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    shutdown_tracing()


app = FastAPI(title="maritime-safety-copilot", version="0.1.0", lifespan=lifespan)


class HistoryMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["knowledge_base", "general"] = "knowledge_base"
    retrieval_enabled: bool = True
    top_k: int = Field(default=3, ge=1, le=6)
    min_similarity: float = Field(default=0.3, ge=0, le=1)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=8)


class Source(BaseModel):
    citation: str
    chunk_id: str
    source: str
    title: str
    metadata: dict = Field(default_factory=dict)
    section: str
    excerpt: str
    score: float
    similarity: float | None
    keyword_score: float


class TechnicalDetails(BaseModel):
    retrieval_method: Literal["exact", "semantic", "hybrid", "disabled"]
    chunks_retrieved: int
    latency_ms: float


class AskResponse(BaseModel):
    answer: str
    mode: Literal["knowledge_base", "general"]
    sources: list[Source]
    technical_details: TechnicalDetails
    warnings: list[str] = Field(default_factory=list)


@app.exception_handler(llm.ConfigurationError)
def configuration_error(request: Request, exc: llm.ConfigurationError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(llm.ProviderError)
def provider_error(request: Request, exc: llm.ProviderError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
def health():
    """Liveness only; deliberately makes no model or telemetry network calls."""
    return {"status": "ok", "app": "maritime-safety-copilot"}


def prepare_answer(request: AskRequest):
    sources = []
    method = "disabled"
    enabled = request.mode == "knowledge_base" and request.retrieval_enabled
    with trace(
        "retrieval",
        kind="retriever",
        input={
            "question": request.question,
            "enabled": enabled,
            "top_k": request.top_k,
            "min_similarity": request.min_similarity,
        },
    ) as span:
        if enabled:
            try:
                chunks = retrieval.load_chunks()
            except retrieval.KnowledgeBaseError as exc:
                raise HTTPException(503, str(exc)) from exc
            except (OSError, UnicodeError) as exc:
                raise HTTPException(503, "Could not read the local data/ knowledge base.") from exc
            try:
                method, sources = retrieval.retrieve(
                    request.question,
                    chunks,
                    embed=llm.embed_texts,
                    embed_corpus=llm.embed_documents,
                    top_k=request.top_k,
                    min_similarity=request.min_similarity,
                )
            except ValueError as exc:
                raise llm.ProviderError(
                    "Invalid embedding response; check the embedding deployment."
                ) from exc
        update_span(span, output={"method": method, "sources": sources})
    return method, sources


def finish_answer(request, answer, method, sources, started):
    warnings = []
    if request.mode == "knowledge_base" and not sources:
        answer = INSUFFICIENT_EVIDENCE
        warnings.append(
            "Retrieval is off."
            if not request.retrieval_enabled
            else "No matching demo evidence found."
        )
    if request.mode == "knowledge_base":
        labels = set(re.findall(r"\[(S\d+)\]", answer))
        allowed = {source["citation"] for source in sources}
        if labels - allowed or (not labels and "insufficient" not in answer.lower()):
            answer = INSUFFICIENT_EVIDENCE
            warnings.append("The generated answer failed the source-label check.")
    response = AskResponse(
        answer=answer,
        mode=request.mode,
        sources=sources,
        warnings=warnings,
        technical_details=TechnicalDetails(
            retrieval_method=method,
            chunks_retrieved=len(sources),
            latency_ms=round((perf_counter() - started) * 1000, 1),
        ),
    )
    return response


def answer_messages(request, sources):
    return build_messages(
        request.question, [item.model_dump() for item in request.history], sources, request.mode
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    started = perf_counter()
    with trace("ask", input=request.model_dump()) as root:
        method, sources = prepare_answer(request)
        answer = INSUFFICIENT_EVIDENCE
        if request.mode == "general" or sources:
            messages = answer_messages(request, sources)
            with trace(
                "llm",
                kind="generation",
                input=messages,
                model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "unconfigured"),
            ) as span:
                answer = llm.generate_answer(messages)
                update_span(span, output=answer)
        response = finish_answer(request, answer, method, sources, started)
        update_span(root, output=response.model_dump())
        return response


def stream_event(event_type, **payload):
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


async def answer_events(request: AskRequest):
    started = perf_counter()
    try:
        # These contexts stay in a single async task across every yielded delta.
        with trace("ask", input=request.model_dump(), metadata={"streaming": True}) as root:
            try:
                method, sources = await asyncio.to_thread(prepare_answer, request)
                answer = INSUFFICIENT_EVIDENCE
                if request.mode == "general" or sources:
                    messages = answer_messages(request, sources)
                    with trace(
                        "llm",
                        kind="generation",
                        input=messages,
                        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "unconfigured"),
                    ) as span:
                        parts = []
                        try:
                            async with aclosing(llm.stream_answer(messages)) as tokens:
                                async for token in tokens:
                                    parts.append(token)
                                    yield stream_event("delta", text=token)
                            answer = "".join(parts).strip()
                            update_span(span, output=answer)
                        except BaseException:
                            update_span(
                                span,
                                output="".join(parts),
                                level="ERROR",
                                status_message="Stream incomplete; output is partial",
                            )
                            raise
                response = finish_answer(request, answer, method, sources, started)
                update_span(root, output=response.model_dump())
            except (asyncio.CancelledError, GeneratorExit):
                update_span(
                    root, level="WARNING", status_message="Client disconnected before completion"
                )
                raise
        # The final event is authoritative: the citation check may replace provisional text.
        yield stream_event("done", response=response.model_dump())
    except (llm.ConfigurationError, llm.ProviderError, HTTPException) as exc:
        status = (
            exc.status_code
            if isinstance(exc, HTTPException)
            else (503 if isinstance(exc, llm.ConfigurationError) else 502)
        )
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        yield stream_event("error", status_code=status, detail=detail)
    except Exception:
        yield stream_event(
            "error", status_code=500, detail="The response stream failed. Please retry."
        )


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    return StreamingResponse(
        answer_events(request),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
