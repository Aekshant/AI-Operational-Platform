"""Retry with exponential backoff and full jitter (AWS architecture blog style)."""

from __future__ import annotations

import random


def backoff(attempt: int, base: float = 0.1, cap: float = 2.0) -> float:
    return random.uniform(0, min(cap, base * 2**attempt))


class RetryableError(Exception):
    """Failure before the first byte reached the client, so it's safe to retry elsewhere."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
