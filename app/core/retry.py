from __future__ import annotations

import logging
import random
import time
from typing import Callable, Iterable, TypeVar

from .errors import RetryableError

log = logging.getLogger(__name__)
T = TypeVar("T")


def call(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: Iterable[type[BaseException]] = (RetryableError,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`, retrying with full-jitter exponential backoff.

    Full jitter rather than plain exponential: when a provider rate-limits you,
    every worker retries at the same moment otherwise and you re-trigger the limit.
    """
    retry_on = tuple(retry_on)
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203
            last = exc
            if attempt == attempts:
                break
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay = random.uniform(0, delay)
            log.warning("attempt %s/%s failed (%s); retrying in %.2fs", attempt, attempts, exc, delay)
            sleep(delay)
    assert last is not None
    raise last
