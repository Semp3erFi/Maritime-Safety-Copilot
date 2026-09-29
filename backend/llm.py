"""The only Azure/model integration. No credentials or embeddings are written to disk."""

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI, OpenAIError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigurationError(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or "your-" in value.lower():
        raise ConfigurationError(f"Set {name} in Day18/.env or the process environment.")
    return value


def client_settings() -> dict:
    endpoint = require_env("AZURE_OPENAI_ENDPOINT")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ConfigurationError("AZURE_OPENAI_ENDPOINT must be an HTTPS resource URL.")
    return dict(
        azure_endpoint=endpoint,
        api_key=require_env("AZURE_OPENAI_API_KEY"),
        api_version=require_env("AZURE_OPENAI_API_VERSION"),
        timeout=30.0,
        max_retries=0,
    )


@lru_cache(maxsize=1)
def get_client() -> AzureOpenAI:
    return AzureOpenAI(**client_settings())


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    deployment = require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    try:
        response = get_client().embeddings.create(model=deployment, input=texts)
    except OpenAIError as exc:
        raise ProviderError(
            "Azure embedding request failed. Check endpoint, API version, deployment, "
            "credentials and quota."
        ) from exc
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


@lru_cache(maxsize=4)
def _cached_embeddings(
    texts: tuple[str, ...], endpoint: str, deployment: str, api_version: str
) -> tuple[tuple[float, ...], ...]:
    # Configuration and corpus contents form the cache key. Never persist the index.
    return tuple(tuple(vector) for vector in embed_texts(list(texts)))


def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors = _cached_embeddings(
        tuple(texts),
        require_env("AZURE_OPENAI_ENDPOINT"),
        require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        require_env("AZURE_OPENAI_API_VERSION"),
    )
    return [list(vector) for vector in vectors]


def generate_answer(messages: list[dict[str, str]]) -> str:
    deployment = require_env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    try:
        response = get_client().chat.completions.create(
            model=deployment,
            messages=messages,
            # Includes reasoning tokens on reasoning models, not just visible answer text.
            # Concision is controlled by the prompt rather than a tight generation budget.
            max_completion_tokens=4096,
        )
    except OpenAIError as exc:
        # Do not return provider exception bodies, which can contain sensitive details.
        raise ProviderError(
            "Azure chat request failed. Check endpoint, API version, deployment, "
            "credentials and quota."
        ) from exc
    if not response.choices:
        raise ProviderError("The model returned no answer. Please try again.")
    choice = response.choices[0]
    return completed_answer(choice.message.content, choice.finish_reason)


def completed_answer(answer: str | None, finish_reason: str | None) -> str:
    if finish_reason == "length":
        raise ProviderError(
            "The model exhausted its completion budget before finishing. "
            "No incomplete answer is shown. Please retry; if this persists, "
            "review the deployment's completion budget in backend/llm.py."
        )
    if finish_reason == "content_filter":
        raise ProviderError("The response was filtered. Please rephrase the question.")
    if not answer or not answer.strip():
        raise ProviderError("The model returned no text (possibly filtered). Try rephrasing.")
    return answer.strip()


async def stream_answer(messages: list[dict[str, str]]):
    """Yield Azure text deltas; close the upstream connection on completion or cancellation."""
    deployment = require_env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    parts, finish_reason = [], None
    try:
        async with AsyncAzureOpenAI(**client_settings()) as client:
            async with await client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_completion_tokens=4096,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    if not chunk.choices:
                        continue  # Azure can send annotation/usage-only chunks.
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
                    if choice.delta.content:
                        parts.append(choice.delta.content)
                        yield choice.delta.content
    except OpenAIError as exc:
        raise ProviderError("Azure chat stream failed or was interrupted. Please retry.") from exc
    if finish_reason is None:
        raise ProviderError("The model stream ended before completion. Please retry.")
    completed_answer("".join(parts), finish_reason)
