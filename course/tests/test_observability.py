import json

import pytest

from ent_agent.dashboard import percentile, render_dashboard, summarise
from ent_agent.observability import CostTracker, Tracer, TracedLLM, estimate_tokens, fingerprint
from ent_agent.rag import FakeLLM

PRICES = {"test-model": (3.0, 15.0)}  # made-up numbers for tests, NOT real prices


def test_spans_nest_and_share_a_trace_id():
    t = Tracer()
    with t.span("request"):
        with t.span("retrieve"):
            pass
    child, root = t.spans[0], t.spans[1]
    assert child["parent_id"] == root["span_id"] and child["trace_id"] == root["trace_id"] and root["parent_id"] is None
    assert root["duration_ms"] >= child["duration_ms"] >= 0


def test_error_spans_record_type_but_not_message():
    t = Tracer()
    with pytest.raises(ValueError):
        with t.span("boom"):
            raise ValueError("customer secret 4111 1111 1111 1111")
    assert t.spans[0]["status"] == "error" and t.spans[0]["attrs"]["error_type"] == "ValueError"
    assert "4111" not in json.dumps(t.spans)


def test_spans_are_written_as_jsonl(tmp_path):
    f = tmp_path / "t.jsonl"
    t = Tracer(f)
    with t.span("a"):
        pass
    assert json.loads(f.read_text().splitlines()[0])["name"] == "a"


def test_cost_math_and_budget():
    c = CostTracker(PRICES, daily_budget_usd=0.01)
    usd = c.record("test-model", 1_000_000, 100_000)
    assert usd == pytest.approx(3.0 + 1.5) and c.over_budget


def test_unpriced_model_fails_loudly():
    with pytest.raises(KeyError):
        CostTracker(PRICES).record("mystery", 1, 1)


def test_traced_llm_logs_fingerprints_not_text():
    t, c = Tracer(), CostTracker(PRICES)
    prompt = '<doc id="d#0" source="s">\nTop secret refund rule.\n</doc>'
    out = TracedLLM(FakeLLM(), "test-model", t, c).invoke(prompt)
    blob = json.dumps(t.spans)
    assert "Top secret" not in blob and fingerprint(prompt) in blob and out
    assert t.spans[0]["attrs"]["in_tokens"] == estimate_tokens(prompt) and c.total_usd > 0


def test_percentile_nearest_rank():
    assert percentile([10, 20, 30, 40, 50], 50) == 30 and percentile([10, 20, 30, 40, 50], 95) == 50 and percentile([], 50) == 0.0


def test_dashboard_renders_numbers_and_escapes_html(tmp_path):
    t = Tracer(tmp_path / "t.jsonl")
    costs = CostTracker(PRICES)
    llm = TracedLLM(FakeLLM(), "test-model", t, costs)
    for _ in range(3):
        with t.span("request"):
            llm.invoke('<doc id="d#0" source="s">\nhello world.\n</doc>')
    with (tmp_path / "t.jsonl").open("a") as f:  # a slow span with a hostile name, written by hand so it ranks in the top 5
        f.write(json.dumps({"trace_id": "x", "span_id": "y", "parent_id": None, "name": "<script>alert(1)</script>", "attrs": {}, "status": "ok", "duration_ms": 9999.0}) + "\n")
    s = render_dashboard(tmp_path / "t.jsonl", tmp_path / "d.html")
    page = (tmp_path / "d.html").read_text()
    assert s["requests"] == 4 and s["llm_calls"] == 3 and "<script>alert" not in page and "&lt;script&gt;" in page
    assert summarise([])["requests"] == 0
