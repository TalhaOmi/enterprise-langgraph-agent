"""Module 20 demo: supervisor + two specialists + private memory. Run: python run_m20.py"""

from ent_agent.memory import MemoryStore
from ent_agent.multi_agent import ask_multi, build_multi_agent_graph
from ent_agent.rag import Document, FakeLLM, HashEmbedder, VectorStore

ORDERS = {"acme": {"ORD-1001": "shipped"}, "globex": {"ORD-2002": "processing"}}
store = VectorStore(HashEmbedder())
store.add_document(Document("refunds", "acme", "Customers may be refunded within 30 days of delivery. Refunds go back to the original payment method.", "refunds.md"))
memory = MemoryStore()
graph = build_multi_agent_graph(store, ORDERS, FakeLLM(), memory)

for who, q in [
    ("alice", "refund policy: which payment method, and status of ORD-1001"),
    ("alice", "status of ORD-2002"),  # that order belongs to another tenant
    ("alice", "remember that I prefer short answers, my email is alice@example.com"),
    ("alice", "what is the refund policy"),
    ("bob", "what is the refund policy"),
]:
    r = ask_multi(graph, q, "acme", who)
    print(f"\n[{who}] {q}\n  trace : {r['trace']}\n  answer: {r['answer'][:110]}\n  memory seen by the model: {bool(r.get('memory_note'))}")
print("\nalice's stored memories:", memory.list("acme", "alice"))
