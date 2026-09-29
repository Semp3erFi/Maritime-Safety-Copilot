"""Keep the entire test suite offline, independent of any real .env settings."""

import os

import pytest

from backend import llm, tracing


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch):
    cached_functions = (llm.get_client, llm._cached_embeddings, tracing.get_tracer)
    for name in os.environ:
        if name.startswith(("AZURE_OPENAI_", "LANGFUSE_")):
            monkeypatch.delenv(name, raising=False)
    for function in cached_functions:
        function.cache_clear()
    yield
    for function in cached_functions:
        function.cache_clear()
