"""Consume the backend's newline-delimited streaming protocol without model imports."""

import json

import httpx


class StreamError(RuntimeError):
    pass


def answer_tokens(url: str, payload: dict, completed: dict):
    with httpx.stream("POST", url, json=payload, timeout=httpx.Timeout(120, connect=5)) as response:
        if response.is_error:
            response.read()
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["type"] == "delta":
                if not isinstance(event["text"], str):
                    raise StreamError("Invalid response fragment. Please retry.")
                yield event["text"]
            elif event["type"] == "done":
                result = event["response"]
                if not isinstance(result["answer"], str) or not isinstance(result["sources"], list):
                    raise StreamError("Invalid completed response. Please retry.")
                if not isinstance(result["technical_details"], dict):
                    raise StreamError("Missing response details. Please retry.")
                completed.update(result)
                return
            elif event["type"] == "error":
                raise StreamError(event.get("detail", "The response stream failed. Please retry."))
            else:
                raise StreamError("Unexpected response event. Please retry.")
    raise StreamError("The response stream ended before completion. Please retry.")
