"""Admission control: a bulkhead per backend with a bounded, priority-aware queue.

    in-flight < max_inflight          -> admitted immediately
    queue has room for this priority  -> wait (up to queue_timeout) for a slot
    otherwise                         -> shed (429 + Retry-After)

Lower-priority tenants see a smaller effective queue, so under pressure the free tier
is shed first and paying tenants keep their latency. That's load shedding by policy,
not by luck.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

# fraction of the queue each priority class may occupy
QUEUE_SHARE = {0: 0.5, 1: 0.8, 2: 1.0}


class Shed(Exception):
    def __init__(self, reason: str, retry_after: float = 1.0):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


class Bulkhead:
    def __init__(self, max_inflight: int, max_queue: int, queue_timeout_s: float):
        self.max_inflight = max_inflight
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self.sem = asyncio.Semaphore(max_inflight)
        self.inflight = 0
        self.queued = 0

    @asynccontextmanager
    async def admit(self, priority: int):
        limit = int(self.max_queue * QUEUE_SHARE.get(priority, 0.5))
        if self.inflight >= self.max_inflight and self.queued >= limit:
            raise Shed("queue_full", retry_after=1.0 + self.queued / max(1, self.max_inflight))
        self.queued += 1
        try:
            await asyncio.wait_for(self.sem.acquire(), timeout=self.queue_timeout_s)
        except asyncio.TimeoutError:
            raise Shed("queue_timeout", retry_after=2.0) from None
        finally:
            self.queued -= 1
        self.inflight += 1
        try:
            yield
        finally:
            self.inflight -= 1
            self.sem.release()
