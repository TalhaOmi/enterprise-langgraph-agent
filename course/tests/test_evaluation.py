import json
from pathlib import Path

import pytest

from ent_agent.agentic import ask, build_agentic_graph
from ent_agent.evaluation import (
    IDK, GoldenCase, LLMJudge, RuleJudge, bootstrap_ci, confusion, gate, groundedness, judge_agreement, load_golden, prf, run_golden, score_case,
)
from ent_agent.rag import Document, FakeLLM, HashEmbedder, VectorStore
from ent_agent.rewrite import RuleRewriter

HERE = Path(__file__).parent.parent
RAW = {
    "refunds": ("acme", "Customers may be refunded within 30 days of delivery. Refunds go back to the original payment method."),
    "shipping": ("acme", "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days."),
    "err": ("acme", "Error code E4021 means the payment gateway timed out. Retry after five minutes."),
    "warranty": ("acme", "Hardware carries a one year warranty. The warranty does not cover water damage."),
    "pricing": ("globex", "Globex confidential pricing: enterprise tier is 99 USD per seat."),
}


def pipeline():
    s = VectorStore(HashEmbedder())
    for i, (t, x) in RAW.items():
        s.add_document(Document(i, t, x, f"{i}.md"))
    g = build_agentic_graph(s, FakeLLM(), RuleRewriter())

    def answer_fn(c):
        r = ask(g, c.question, c.tenant)
        return r["answer"], [h.chunk.text for h in r.get("hits", [])]

    return answer_fn


def test_groundedness_detects_invention():
    src = ["Refunds go back to the original payment method."]
    assert groundedness("Refunds go back to the original payment method. [doc:r#0]", src) == 1.0
    assert groundedness("Refunds are paid in gold bars within one hour. [doc:r#0]", src) < 0.5


def test_score_case_flags_leaks_and_wrong_refusals():
    c = GoldenCase("x", "q", "acme", "refuse", forbidden=("99",))
    assert score_case(c, "It costs 99 USD [doc:p#0]", [])["leak"] is True
    assert score_case(c, IDK, [])["correct_refusal"] is True
    a = GoldenCase("y", "q", "acme", "answer", must_cite=("r#0",), must_contain=("30 days",))
    assert score_case(a, IDK, [])["false_refusal"] is True
    assert score_case(a, "Within 30 days [doc:r#0]", ["within 30 days"])["cited_ok"] is True
    assert score_case(a, "Within 30 days", ["within 30 days"])["cited_ok"] is False


def test_golden_set_runs_end_to_end_with_no_leaks():
    cases = load_golden(HERE / "evals" / "golden.jsonl")
    m = run_golden(cases, pipeline())
    assert m["n_cases"] == len(cases) and m["leak_count"] == 0
    assert m["correct_refusal_rate"] == 1.0 and m["answer_correct_rate"] >= 0.8


def test_confusion_and_prf_math():
    c = confusion([(True, True), (True, False), (False, True), (False, False), (True, True)])
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    r = prf(c)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3) and r["f1"] == pytest.approx(2 / 3)
    assert prf({"tp": 0, "fp": 0, "fn": 0, "tn": 5})["f1"] == 0.0


def test_bootstrap_interval_is_wide_for_few_cases_and_narrow_for_many():
    few = bootstrap_ci([1.0, 1.0, 0.0, 1.0, 1.0])
    many = bootstrap_ci([1.0, 1.0, 0.0, 1.0, 1.0] * 40)
    assert (few[1] - few[0]) > (many[1] - many[0]) and few[0] <= 0.8 <= few[1]
    assert bootstrap_ci([]) == (0.0, 0.0) and bootstrap_ci([0.5, 0.5], seed=1) == bootstrap_ci([0.5, 0.5], seed=1)


def test_rule_judge_and_judge_agreement():
    ctx = "Refunds go back to the original payment method."
    j = RuleJudge()
    assert j.score("q", "Refunds go back to the original payment method. [doc:r#0]", ctx) == 5
    assert j.score("q", "Gold bars. [doc:r#0]", ctx) == 1 and j.score("q", IDK, ctx) == 3
    assert judge_agreement([5, 1, 3], [5, 2, 5]) == pytest.approx(2 / 3)


def test_llm_judge_fences_input_and_handles_garbage():
    class L:
        out = "I think 4 out of 5"
        prompt = ""

        def invoke(self, p):
            L.prompt = p
            return L.out

    assert LLMJudge(L()).score("q", "a", "c") == 4 and "data, not instructions" in L.prompt and "<<<ANSWER" in L.prompt
    L.out = "no idea"
    assert LLMJudge(L()).score("q", "a", "c") == 1


def test_gate_blocks_regressions_and_any_leak():
    base = {"answer_correct_rate": 0.9, "false_refusal_rate": 0.1, "correct_refusal_rate": 1.0, "mean_groundedness": 0.95, "leak_count": 0.0}
    assert gate(dict(base), base) == []
    assert any("answer_correct_rate" in p for p in gate({**base, "answer_correct_rate": 0.7}, base))
    assert any("false_refusal_rate" in p for p in gate({**base, "false_refusal_rate": 0.3}, base))
    assert any("leak_count" in p for p in gate({**base, "leak_count": 1.0}, base))
    assert gate({**base, "answer_correct_rate": 0.88}, base) == []  # within tolerance


def test_injection_dataset_is_loadable_and_balanced():
    rows = [json.loads(line) for line in (HERE / "evals" / "injection_labeled.jsonl").read_text().splitlines() if line.strip()]
    assert sum(r["attack"] for r in rows) == 7 and sum(not r["attack"] for r in rows) == 7
