"""Module 17 demo: compare vector-only, keyword-only and hybrid on a labelled set, then run the agentic graph."""

from ent_agent.agentic import ask, build_agentic_graph
from ent_agent.hybrid import hybrid_search, keyword_search
from ent_agent.rag import Document, FakeLLM, HashEmbedder, VectorStore
from ent_agent.rewrite import RuleRewriter
from ent_agent.retrieval_eval import Case, evaluate

DOCS = {
    "refunds": "Customers may be refunded within 30 days of delivery. Refunds go back to the original payment method.",
    "pto": "Employees get 15 days of paid time off leave per year. Request leave from your manager.",
    "err": "Error code E4021 means the payment gateway timed out. Retry after five minutes.",
    "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "returns": "To return an item, request a return label within 14 days. Returned items must be unused.",
    "warranty": "Hardware carries a one year warranty. The warranty does not cover water damage.",
    "security": "Report a lost badge to the security desk immediately. Badges are deactivated within one hour.",
    "expenses": "Submit expense reports by the fifth of the month. Receipts are required over 25 dollars.",
}
store = VectorStore(HashEmbedder())
for i, t in DOCS.items():
    store.add_document(Document(i, "acme", t, f"{i}.md"))

CASES = [
    Case("what does E4021 mean", "acme", frozenset({"err"})),
    Case("payment gateway timed out", "acme", frozenset({"err"})),
    Case("how long do I have to get my money back", "acme", frozenset({"refunds"})),
    Case("refund payment method", "acme", frozenset({"refunds"})),
    Case("days off per year", "acme", frozenset({"pto"})),
    Case("paid time off leave", "acme", frozenset({"pto"})),
    Case("how fast is express delivery", "acme", frozenset({"shipping"})),
    Case("lost badge", "acme", frozenset({"security"})),
    Case("when are receipts needed for expenses", "acme", frozenset({"expenses"})),
    Case("does warranty cover water damage", "acme", frozenset({"warranty"})),
]
N = frozenset()
systems = {
    "vector only": lambda c: store.search(c.query, c.tenant_id, N, k=3, min_score=0.0),
    "keyword (BM25) only": lambda c: keyword_search(store, c.query, c.tenant_id, N, k=3),
    "hybrid (RRF)": lambda c: hybrid_search(store, c.query, c.tenant_id, N, k=3),
}
for k in (1, 3):
    print(f"\nk={k}  (n={len(CASES)} cases)")
    print(f"{'system':22} recall  mrr   hit")
    for name, fn in systems.items():
        m = evaluate(fn, CASES, k=k)
        print(f"{name:22} {m['recall@k']:.2f}    {m['mrr']:.2f}  {m['hit_rate@k']:.2f}")

g = build_agentic_graph(store, FakeLLM(), RuleRewriter())
for q in ["Refunds go back to which payment method", "hey can you tell me about PTO please", "quantum banana spaceship"]:
    r = ask(g, q, "acme")
    print(f"\nQ: {q}\n  trace: {r['trace']}\n  answer: {r['answer']}")
