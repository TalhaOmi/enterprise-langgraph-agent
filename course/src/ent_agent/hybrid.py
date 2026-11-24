"""Module 17 part 1: hybrid search = keyword (BM25) + vector, merged with reciprocal rank fusion.

Why: vectors find similar MEANING, keywords find exact TERMS (ids, names, error codes). Each misses
what the other catches. The enterprise rules still apply: tenant and role filtering happens first,
and the integrity check runs on every hit.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .rag import Hit, VectorStore

_WORD = re.compile(r"[a-z0-9]{2,}")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def bm25_scores(query: str, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Classic BM25: rare query words count more; long documents are slightly penalised."""
    n = len(docs)
    if n == 0:
        return []
    avg = sum(len(d) for d in docs) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    out = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for w in set(words(query)):
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            score += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        out.append(score)
    return out


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: score = sum of 1/(k + rank) over each ranking. No score scaling needed."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def keyword_search(store: VectorStore, query: str, tenant_id: str, roles: frozenset[str], k: int = 10) -> list[Hit]:
    allowed = [c for c in store._chunks if c.tenant_id == tenant_id and (not c.allowed_roles or c.allowed_roles & roles)]
    scores = bm25_scores(query, [words(c.text) for c in allowed])
    ranked = sorted(zip(scores, allowed, strict=True), key=lambda t: (-t[0], t[1].id))
    return [
        Hit(c, s) for s, c in ranked[:k] if s > 0 and hashlib.sha256(c.text.encode()).hexdigest() == c.sha256
    ]


def hybrid_search(store: VectorStore, query: str, tenant_id: str, roles: frozenset[str], k: int = 3, pool: int = 10) -> list[Hit]:
    vec = store.search(query, tenant_id, roles, k=pool, min_score=0.0)
    kw = keyword_search(store, query, tenant_id, roles, k=pool)
    fused = rrf([[h.chunk.id for h in vec], [h.chunk.id for h in kw]])
    by_id = {h.chunk.id: h for h in vec + kw}
    ranked = sorted(fused, key=lambda i: (-fused[i], i))[:k]
    return [Hit(by_id[i].chunk, fused[i]) for i in ranked]
