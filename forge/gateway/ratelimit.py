"""Per-tenant token-bucket rate limiting.

In-process buckets are exact for one gateway replica. With N replicas, either give
each replica rps/N (what the Helm chart does) or back this with Redis (same interface).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    rate: float
    burst: int
    tokens: float = field(default=-1)
    updated: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.tokens < 0:
            self.tokens = float(self.burst)

    def try_take(self, n: float = 1.0) -> tuple[bool, float]:
        """Returns (allowed, seconds until enough tokens)."""
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= n:
            self.tokens -= n
            return True, 0.0
        return False, (n - self.tokens) / self.rate if self.rate > 0 else float("inf")


class RateLimiter:
    def __init__(self, limits: dict[str, tuple[float, int, int]], default: tuple[float, int, int] = (5, 10, 0)):
        self.limits = limits
        self.default = default
        self.buckets: dict[str, TokenBucket] = {}

    def priority(self, tenant: str) -> int:
        return self.limits.get(tenant, self.default)[2]

    def check(self, tenant: str) -> tuple[bool, float]:
        b = self.buckets.get(tenant)
        if b is None:
            rate, burst, _ = self.limits.get(tenant, self.default)
            b = self.buckets[tenant] = TokenBucket(rate, burst)
        return b.try_take()
