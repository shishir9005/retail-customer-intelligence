"""A tiny sliding-window rate limiter, used to protect the free Groq quota from visitors."""
from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowLimiter:
    """Allow at most `max_calls` per `window_s` seconds, shared by every visitor of the app."""

    def __init__(self, max_calls: int, window_s: float, clock=time.monotonic):
        self.max_calls, self.window_s, self._clock = max_calls, window_s, clock
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Record a call and return True if allowed; return False (recording nothing) if over the limit."""
        now = self._clock()
        with self._lock:
            while self._calls and now - self._calls[0] >= self.window_s:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True
