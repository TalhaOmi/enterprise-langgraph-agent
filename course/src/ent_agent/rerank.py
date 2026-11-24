"""Module 17 part 2: reranking. Retrieve many cheaply, then re-order the few best with a slower, sharper scorer."""

from __future__ import annotations

from typing import Protocol

from .hybrid import words
from .rag import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class CoverageReranker:
    """Offline stand-in: prefers chunks that contain MORE of the distinct query words, then shorter chunks."""

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        q = set(words(query))

        def key(h: Hit) -> tuple[float, int, str]:
            covered = len(q & set(words(h.chunk.text))) / (len(q) or 1)
            return (-covered, len(h.chunk.text), h.chunk.id)

        return sorted(hits, key=key)[:k]


class CrossEncoderReranker:
    """Real reranker (not tested by me): a cross-encoder reads query and chunk TOGETHER, which is more accurate
    than comparing two separate vectors. Needs `pip install sentence-transformers` and a model download."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        scores = self._model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(zip(scores, hits, strict=True), key=lambda t: -float(t[0]))
        return [Hit(h.chunk, float(s)) for s, h in order[:k]]
