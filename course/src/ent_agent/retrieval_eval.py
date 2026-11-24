"""Module 17 part 5: measure retrieval, do not guess. Each case says which documents SHOULD be found."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .rag import Hit


@dataclass(frozen=True)
class Case:
    query: str
    tenant_id: str
    relevant: frozenset[str]  # doc ids that count as correct


def recall_at_k(ranked_docs: list[str], relevant: frozenset[str], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    return len(set(ranked_docs[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked_docs: list[str], relevant: frozenset[str]) -> float:
    """1/rank of the first relevant document, 0 if none. MRR is the mean of this over cases."""
    for i, d in enumerate(ranked_docs, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def evaluate(search: Callable[[Case], list[Hit]], cases: list[Case], k: int = 3) -> dict[str, float]:
    recalls, rrs, hits = [], [], []
    for c in cases:
        docs: list[str] = []
        for h in search(c):
            if h.chunk.doc_id not in docs:  # several chunks of one document count once
                docs.append(h.chunk.doc_id)
        recalls.append(recall_at_k(docs, c.relevant, k))
        rrs.append(reciprocal_rank(docs, c.relevant))
        hits.append(1.0 if set(docs[:k]) & c.relevant else 0.0)
    n = len(cases) or 1
    return {"recall@k": sum(recalls) / n, "mrr": sum(rrs) / n, "hit_rate@k": sum(hits) / n, "cases": float(len(cases))}
