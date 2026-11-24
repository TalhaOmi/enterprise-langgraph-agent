"""Module 4 demo: ingest, retrieve, answer, check citations. Run: python run_m4.py"""
from ent_agent.rag import Document, FakeLLM, HashEmbedder, VectorStore, build_prompt, check_citations

store = VectorStore(HashEmbedder())
store.add_document(Document("refunds", "acme", "Customers may be refunded within 30 days of delivery. Refunds go back to the original payment method.", "refunds.md"))
store.add_document(Document("pricing", "globex", "Globex confidential pricing: enterprise tier is 99 USD per seat.", "pricing.md"))

for tenant, q in [("acme", "which payment method do refunds go back to"), ("acme", "Globex pricing per seat")]:
    hits = store.search(q, tenant, frozenset())
    print(f"\n[{tenant}] {q}")
    for h in hits:
        print(f"  {h.score:.2f} {h.chunk.id}: {h.chunk.text[:60]}")
    answer = FakeLLM().invoke(build_prompt(q, hits))
    print("  answer:", answer, "| citations ok:", check_citations(answer, hits)[0])
