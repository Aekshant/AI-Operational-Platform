"""Circuit breaker over a rolling window of call outcomes.

CLOSED --(failure ratio >= threshold over >= min_calls)--> OPEN
OPEN   --(open_s elapsed)--> HALF_OPEN (one probe at a time)
HALF_OPEN --probe ok--> CLOSED ; --probe fails--> OPEN

Slow calls (TTFT over the SLO budget) count as failures. A model that answers
in 8 seconds is down from the user's point of view.
"""

from __future__ import annotations

import time
from collections import deque

CLOSED, HALF_OPEN, OPEN = "closed", "half_open", "open"
STATE_VALUE = {CLOSED: 0, HALF_OPEN: 1, OPEN: 2}


class CircuitBreaker:
    def __init__(self, threshold: float = 0.5, min_calls: int = 10, open_s: float = 15, window: int = 50):
        self.threshold = threshold
        self.min_calls = min_calls
        self.open_s = open_s
        self.outcomes: deque[bool] = deque(maxlen=window)
        self.state = CLOSED
        self.opened_at = 0.0
        self.probing = False

    def allow(self) -> bool:
        if self.state == OPEN and time.monotonic() - self.opened_at >= self.open_s:
            self.state, self.probing = HALF_OPEN, False
        if self.state == CLOSED:
            return True
        if self.state == HALF_OPEN and not self.probing:
            self.probing = True
            return True
        return False

    def record(self, ok: bool) -> None:
        if self.state == HALF_OPEN:
            self.probing = False
            if ok:
                self.state = CLOSED
                self.outcomes.clear()
            else:
                self._open()
            return
        self.outcomes.append(ok)
        if len(self.outcomes) >= self.min_calls:
            failure_ratio = self.outcomes.count(False) / len(self.outcomes)
            if failure_ratio >= self.threshold:
                self._open()

    def _open(self) -> None:
        self.state = OPEN
        self.opened_at = time.monotonic()
        self.outcomes.clear()
