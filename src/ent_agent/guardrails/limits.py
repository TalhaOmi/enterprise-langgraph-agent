"""Cost & availability controls (OWASP LLM10 unbounded consumption).

In-memory implementations so the repo runs anywhere. For multi-replica deployments back
these with Redis (same interfaces) - otherwise each replica enforces its own limits.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RateLimiter:
    """Sliding window: at most `limit` events per `window` seconds per key."""

    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self._clock = limit, window, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


class TokenBudget:
    """Daily token budget per key (user/tenant). Checked before a run, charged after."""

    def __init__(self, daily_limit: int, clock: Callable[[], float] = time.time):
        self.daily_limit, self._clock = daily_limit, clock
        self._used: dict[tuple[str, int], int] = defaultdict(int)
        self._lock = threading.Lock()

    def _day(self) -> int:
        return int(self._clock() // 86_400)

    def has_budget(self, key: str) -> bool:
        with self._lock:
            return self._used[(key, self._day())] < self.daily_limit

    def charge(self, key: str, tokens: int) -> None:
        with self._lock:
            self._used[(key, self._day())] += max(0, tokens)

    def used(self, key: str) -> int:
        with self._lock:
            return self._used[(key, self._day())]


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """Stop hammering a failing model provider; fail fast, recover after `reset_seconds`."""

    def __init__(self, failure_threshold: int, reset_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.reset, self._clock = failure_threshold, reset_seconds, clock
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            if self._opened_at is not None:
                if self._clock() - self._opened_at < self.reset:
                    raise CircuitOpenError("circuit open")
                self._opened_at = None  # half-open: let one attempt through
                self._failures = self.threshold - 1
        try:
            result = fn()
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self.threshold:
                    self._opened_at = self._clock()
            raise
        with self._lock:
            self._failures = 0
        return result
