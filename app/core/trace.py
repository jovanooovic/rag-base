from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    started_at: float
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return round(((self.ended_at or time.time()) - self.started_at) * 1000, 2)


class Trace:
    """A dead-simple nested-span tracer that writes one JSON file per run.

    Deliberately not OpenTelemetry: every client has a different observability
    stack and none of them want a vendor pinned on day one. This gives you a
    complete, greppable record of a run with zero infrastructure, and it is
    ~80 lines to swap for OTel when a client asks.
    """

    def __init__(self, run_id: str | None = None, *, enabled: bool = True, out_dir: str | Path = "./traces"):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.enabled = enabled
        self.out_dir = Path(out_dir)
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self.counters: dict[str, float] = {"llm_calls": 0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}

    @contextmanager
    def span(self, name: str, /, **attributes: Any) -> Iterator[Span]:
        """`name` is positional-only so an attribute may also be called "name".

        Not pedantry: `trace.span("tool", name=tool.name)` is the natural way to
        write it, and without this it raises TypeError at runtime.
        """
        parent = self._stack[-1].span_id if self._stack else None
        s = Span(name=name, span_id=uuid.uuid4().hex[:8], parent_id=parent,
                 started_at=time.time(), attributes=dict(attributes))
        self.spans.append(s)
        self._stack.append(s)
        try:
            yield s
        except BaseException as exc:
            s.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.ended_at = time.time()
            self._stack.pop()

    def count(self, **deltas: float) -> None:
        for k, v in deltas.items():
            self.counters[k] = self.counters.get(k, 0) + v

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "counters": self.counters,
            "spans": [{**asdict(s), "duration_ms": s.duration_ms} for s in self.spans],
        }

    def save(self) -> Path | None:
        if not self.enabled:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def pretty(self) -> str:
        depth: dict[str | None, int] = {None: 0}
        lines = [f"run {self.run_id}  " + "  ".join(f"{k}={v}" for k, v in self.counters.items())]
        for s in self.spans:
            d = depth.get(s.parent_id, 0)
            depth[s.span_id] = d + 1
            mark = "x" if s.error else "-"
            lines.append(f"{'  ' * d}{mark} {s.name} ({s.duration_ms}ms)" + (f" !! {s.error}" if s.error else ""))
        return "\n".join(lines)
