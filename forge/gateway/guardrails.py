"""Request guardrails applied before any model sees the input."""

from __future__ import annotations

import re

from fastapi import HTTPException

# crude but cheap signals; a production deployment would add a classifier model here
_SECRETS = re.compile(r"(AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9]{32,})")


def check_request(body: dict, max_chars: int) -> None:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise HTTPException(400, "messages must be a non-empty list")
    text = "".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
    if len(text) > max_chars:
        raise HTTPException(413, f"prompt exceeds {max_chars} characters")
    if _SECRETS.search(text):
        raise HTTPException(400, "request appears to contain a credential; refusing to forward it to a model")
    mt = body.get("max_tokens")
    if mt is not None and (not isinstance(mt, int) or not 1 <= mt <= 4096):
        raise HTTPException(400, "max_tokens must be an integer in [1, 4096]")
