"""Optional, best-effort Langfuse spans. Telemetry failure must not break an answer."""

import logging
import os
from contextlib import contextmanager
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tracer():
    settings = [
        os.getenv(name, "").strip()
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    ]
    if not all(settings):
        if any(settings):
            logger.warning("Langfuse disabled: configure both keys and LANGFUSE_BASE_URL.")
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings[0], secret_key=settings[1], base_url=settings[2], timeout=3
        )
    except Exception:
        logger.warning("Langfuse unavailable; continuing without tracing.")
        return None


def update_span(span, **values):
    if span is not None:
        try:
            span.update(**values)
        except Exception:
            logger.warning("Could not update Langfuse observation.")


@contextmanager
def trace(name: str, *, kind: str = "span", **values):
    manager, span = None, None
    try:
        tracer = get_tracer()
        if tracer is not None:
            manager = tracer.start_as_current_observation(name=name, as_type=kind, **values)
            span = manager.__enter__()
    except Exception:
        manager = None
        logger.warning("Could not start Langfuse observation.")
    try:
        yield span
    except Exception as exc:
        # Record the error type, not provider bodies, environment values or credentials.
        update_span(span, level="ERROR", status_message=type(exc).__name__)
        raise
    finally:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                logger.warning("Could not finish Langfuse observation.")


def shutdown_tracing():
    tracer = get_tracer()
    if tracer is not None:
        try:
            tracer.shutdown()
        except Exception:
            logger.warning("Could not flush Langfuse observations at shutdown.")
