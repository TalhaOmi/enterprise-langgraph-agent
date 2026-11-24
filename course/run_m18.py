"""Module 18 demo: trace a few requests, track cost, write a dashboard. Run: python run_m18.py  then open dashboard.html"""

from pathlib import Path

from ent_agent.dashboard import render_dashboard
from ent_agent.observability import CostTracker, TracedLLM, Tracer
from ent_agent.rag import Document, FakeLLM, HashEmbedder, VectorStore, build_prompt

trace_file = Path("traces.jsonl")
trace_file.unlink(missing_ok=True)
tracer = Tracer(trace_file)
PRICES = {"demo-model": (3.0, 15.0)}  # example numbers only: use your provider's current prices
costs = CostTracker(PRICES, daily_budget_usd=1.00)
llm = TracedLLM(FakeLLM(), "demo-model", tracer, costs)

store = VectorStore(HashEmbedder())
store.add_document(Document("refunds", "acme", "Customers may be refunded within 30 days of delivery. Refunds go back to the original payment method.", "refunds.md"))

for question in ["which payment method do refunds go back to", "refund days after delivery", "zzzz nothing here"]:
    with tracer.span("request"):
        with tracer.span("retrieve"):
            hits = store.search(question, "acme", frozenset())
        answer = llm.invoke(build_prompt(question, hits))
        print(f"{question!r} -> {answer[:60]}")

summary = render_dashboard(trace_file, Path("dashboard.html"))
print(f"\nrequests={summary['requests']} llm_calls={summary['llm_calls']} p95={summary['p95_ms']:.2f}ms cost=${costs.total_usd:.6f} over_budget={costs.over_budget}")
print("Wrote dashboard.html and traces.jsonl")
