"""Module 22 demo: golden-set quality, injection-detector precision/recall with uncertainty, and the CI gate."""

import json
import sys
from pathlib import Path

from ent_agent.agentic import ask, build_agentic_graph
from ent_agent.evaluation import bootstrap_ci, confusion, gate, load_golden, prf, run_golden
from ent_agent.rag import _INSTRUCTION_LIKE, Document, FakeLLM, HashEmbedder, VectorStore
from ent_agent.rewrite import RuleRewriter

CORPUS = {
    "refunds": ("acme", "Customers may be refunded within 30 days of delivery. Refunds go back to the original payment method."),
    "shipping": ("acme", "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days."),
    "err": ("acme", "Error code E4021 means the payment gateway timed out. Retry after five minutes."),
    "warranty": ("acme", "Hardware carries a one year warranty. The warranty does not cover water damage."),
    "pricing": ("globex", "Globex confidential pricing: enterprise tier is 99 USD per seat."),
}
store = VectorStore(HashEmbedder())
for doc_id, (tenant, text) in CORPUS.items():
    store.add_document(Document(doc_id, tenant, text, f"{doc_id}.md"))
graph = build_agentic_graph(store, FakeLLM(), RuleRewriter())


def answer_fn(case):
    r = ask(graph, case.question, case.tenant)
    return r["answer"], [h.chunk.text for h in r.get("hits", [])]


# 1. golden set
cases = load_golden(Path("evals/golden.jsonl"))
metrics = run_golden(cases, answer_fn)
print("Golden set:", {k: round(v, 2) for k, v in metrics.items()})

# 2. injection detector on a labelled set
rows = [json.loads(line) for line in Path("evals/injection_labeled.jsonl").read_text().splitlines() if line.strip()]
pairs = [(bool(_INSTRUCTION_LIKE.search(r["text"])), r["attack"]) for r in rows]
c = confusion(pairs)
print("\nInjection regex:", c, {k: round(v, 2) for k, v in prf(c).items()})
caught = [1.0 if p else 0.0 for p, t in pairs if t]
lo, hi = bootstrap_ci(caught)
print(f"Attack recall {sum(caught) / len(caught):.2f} with 95% interval {lo:.2f} to {hi:.2f}  (n={len(caught)} attacks: wide on purpose)")
for r, (p, t) in zip(rows, pairs):
    if p != t:
        print("  MISSED" if t else "  FALSE ALARM", "->", r["text"])

# 3. regression gate against a saved baseline
base_file = Path("evals/baseline.json")
if not base_file.exists():
    base_file.write_text(json.dumps(metrics, indent=2))
    print("\nNo baseline yet: saved this run as evals/baseline.json")
problems = gate(metrics, json.loads(base_file.read_text()))
print("\nGate:", "PASS" if not problems else "FAIL: " + "; ".join(problems))
sys.exit(1 if problems else 0)
