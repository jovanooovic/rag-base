"""A sliding-window limiter for login attempts.

In-process and therefore per-worker: two uvicorn workers mean twice the
allowance, and a restart forgets everything. That is a real limitation, not a
detail -- it is written into the README rather than left for someone to
discover. It is still worth having, because the alternative is an endpoint that
will verify passwords as fast as anyone can send them.

Keyed on both the address being attacked and the source doing the attacking:
one alone leaves an obvious hole. Per-email only, and one attacker locks out
any account they choose by failing logins on purpose. Per-IP only, and a
botnet spreads a single account's attempts across a thousand addresses.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, *, max_attempts: int = 10, window_seconds: float = 300.0,
                 max_keys: int = 10_000):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        cutoff = now - self.window
        while hits and hits[0] < cutoff:
            hits.popleft()
        if not hits:
            self._hits.pop(key, None)
            return deque()
        return hits

    def check(self, key: str, *, now: float | None = None) -> bool:
        """True when the caller is still under the limit. Does not record."""
        now = time.monotonic() if now is None else now
        return len(self._prune(key, now)) < self.max_attempts

    def record(self, key: str, *, now: float | None = None) -> None:
        """Count one failed attempt.

        Only failures are recorded. Counting successes too would log a busy
        legitimate user out of their own account.
        """
        now = time.monotonic() if now is None else now
        if len(self._hits) >= self.max_keys and key not in self._hits:
            # Bounded so a flood of unique keys cannot exhaust memory. Dropping
            # the oldest bucket is a small unfairness under attack; unbounded
            # growth is an outage.
            self._hits.pop(next(iter(self._hits)), None)
        self._prune(key, now)
        self._hits[key].append(now)

    def reset(self, key: str) -> None:
        """Called after a successful login, so one bad day at the keyboard does
        not keep counting against someone who then got it right."""
        self._hits.pop(key, None)

    def clear(self) -> None:
        self._hits.clear()
