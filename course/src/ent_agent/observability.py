"""Module 18: see what the system does. Traces (what ran, how long), token and cost tracking, and privacy.

Privacy rule (same as the audit log): never write raw prompts or answers to telemetry. Write a short
fingerprint and the length, so you can correlate events without storing customer text.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_span", default=None)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def estimate_tokens(text: str) -> int:
    """Rough guess (about 4 characters per token). Real providers report exact counts: prefer those."""
    return max(1, len(text) // 4)


class Tracer:
    """Records spans (a named piece of work with a start, a duration and a status) to a list and optionally a JSONL file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs: object) -> Iterator[dict]:
        parent = _current.get()
        rec = {
            "trace_id": parent["trace_id"] if parent else uuid.uuid4().hex[:16],
            "span_id": uuid.uuid4().hex[:8],
            "parent_id": parent["span_id"] if parent else None,
            "name": name,
            "attrs": dict(attrs),
            "status": "ok",
        }
        token = _current.set(rec)
        start = time.perf_counter()
        try:
            yield rec
        except Exception as e:
            rec["status"] = "error"
            rec["attrs"]["error_type"] = type(e).__name__  # the type only, never the message (it may hold user text)
            raise
        finally:
            rec["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)
            _current.reset(token)
            self.spans.append(rec)
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")


@dataclass
class CostTracker:
    """Adds up tokens and money. `prices` maps model name to (USD per million input tokens, USD per million output tokens).

    YOU supply the prices from your provider's current price page. They change, so none are hard-coded here.
    """

    prices: dict[str, tuple[float, float]]
    daily_budget_usd: float | None = None
    calls: list[dict] = field(default_factory=list)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p_in, p_out = self.prices[model]  # KeyError on an unpriced model is deliberate: unknown cost must not pass silently
        cost = input_tokens * p_in / 1e6 + output_tokens * p_out / 1e6
        self.calls.append({"model": model, "in": input_tokens, "out": output_tokens, "usd": cost})
        return cost

    @property
    def total_usd(self) -> float:
        return sum(c["usd"] for c in self.calls)

    @property
    def over_budget(self) -> bool:
        return self.daily_budget_usd is not None and self.total_usd > self.daily_budget_usd


class TracedLLM:
    """Wraps any object with .invoke(prompt) -> str. Adds a span and a cost record to every call."""

    def __init__(self, llm, model: str, tracer: Tracer, costs: CostTracker) -> None:  # noqa: ANN001
        self.llm, self.model, self.tracer, self.costs = llm, model, tracer, costs

    def invoke(self, prompt: str) -> str:
        with self.tracer.span("llm.call", model=self.model, prompt_fp=fingerprint(prompt), prompt_chars=len(prompt)) as sp:
            out = self.llm.invoke(prompt)
            t_in, t_out = estimate_tokens(prompt), estimate_tokens(out)
            sp["attrs"].update(output_fp=fingerprint(out), output_chars=len(out), in_tokens=t_in, out_tokens=t_out)
            sp["attrs"]["usd"] = self.costs.record(self.model, t_in, t_out)
            return out
