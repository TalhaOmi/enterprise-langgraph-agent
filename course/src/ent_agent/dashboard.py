"""Module 18 part 2: turn a trace file into one self-contained HTML dashboard (no server, no JavaScript needed)."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. p in 0..100."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def summarise(spans: list[dict]) -> dict:
    llm = [s for s in spans if s["name"] == "llm.call"]
    by_model: dict[str, float] = {}
    for s in llm:
        m = s["attrs"].get("model", "?")
        by_model[m] = by_model.get(m, 0.0) + float(s["attrs"].get("usd", 0.0))
    roots = [s for s in spans if s["parent_id"] is None]
    durations = [s["duration_ms"] for s in roots]
    return {
        "requests": len(roots),
        "errors": sum(1 for s in spans if s["status"] == "error"),
        "p50_ms": percentile(durations, 50),
        "p95_ms": percentile(durations, 95),
        "llm_calls": len(llm),
        "total_usd": sum(by_model.values()),
        "usd_by_model": by_model,
        "slowest": sorted(spans, key=lambda s: -s["duration_ms"])[:5],
    }


def render_dashboard(trace_file: Path, out_file: Path) -> dict:
    spans = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    s = summarise(spans)
    tiles = "".join(
        f"<div class='t'><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>"
        for k, v in [
            ("requests", s["requests"]),
            ("errors", s["errors"]),
            ("p50 ms", round(s["p50_ms"], 1)),
            ("p95 ms", round(s["p95_ms"], 1)),
            ("LLM calls", s["llm_calls"]),
            ("cost USD", f"{s['total_usd']:.4f}"),
        ]
    )
    models = "".join(f"<tr><td>{html.escape(m)}</td><td>{c:.4f}</td></tr>" for m, c in sorted(s["usd_by_model"].items()))
    slow = "".join(
        f"<tr><td>{html.escape(x['name'])}</td><td>{x['duration_ms']:.1f}</td><td>{html.escape(x['status'])}</td></tr>" for x in s["slowest"]
    )
    out_file.write_text(
        "<!doctype html><meta charset=utf-8><title>Agent dashboard</title>"
        "<style>body{font:15px system-ui;margin:24px;max-width:760px}.g{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}"
        ".t{border:1px solid #8884;border-radius:8px;padding:12px}.t b{display:block;font-size:22px}.t span{opacity:.7}"
        "table{border-collapse:collapse;width:100%;margin:8px 0 20px}td,th{border-bottom:1px solid #8884;padding:6px;text-align:left}</style>"
        f"<h1>Agent dashboard</h1><div class=g>{tiles}</div><h2>Cost by model (USD)</h2><table>{models}</table>"
        f"<h2>Slowest spans</h2><table><tr><th>span</th><th>ms</th><th>status</th></tr>{slow}</table>",
        encoding="utf-8",
    )
    return s
