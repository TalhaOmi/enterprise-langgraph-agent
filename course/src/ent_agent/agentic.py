"""Module 17 part 4: agentic RAG as a LangGraph graph.

retrieve -> grade --enough--> answer
               |
               +--weak, tries left--> rewrite -> retrieve ...
               +--weak, out of tries--> give_up ("I do not know")

The loop is BOUNDED (max_attempts). Without a bound, an agent that keeps rewriting can run forever and cost money.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .hybrid import hybrid_search, words
from .rag import Hit, VectorStore, build_prompt, check_citations
from .rewrite import FILLER, Rewriter


class RagState(TypedDict, total=False):
    question: str
    tenant_id: str
    roles: frozenset[str]
    query: str
    attempt: int
    hits: list[Hit]
    answer: str
    trace: list[str]


def coverage(query: str, hits: list[Hit]) -> float:
    """How much of the question's meaningful words the best chunk contains (0 to 1)."""
    q = {w for w in words(query) if w not in FILLER}
    if not q or not hits:
        return 0.0
    return max(len(q & set(words(h.chunk.text))) / len(q) for h in hits)


def build_agentic_graph(store: VectorStore, llm, rewriter: Rewriter, max_attempts: int = 3, min_coverage: float = 0.5):  # noqa: ANN001
    def retrieve(s: RagState) -> RagState:
        hits = hybrid_search(store, s["query"], s["tenant_id"], s["roles"], k=3)
        return {"hits": hits, "trace": [*s.get("trace", []), f"retrieve({s['query']!r}) -> {len(hits)} hits"]}

    def grade(s: RagState) -> str:
        if coverage(s["query"], s.get("hits", [])) >= min_coverage:
            return "answer"
        tries_left = s["attempt"] + 1 < min(max_attempts, len(rewriter.variants(s["question"])))
        return "rewrite" if tries_left else "give_up"  # no new phrasing left means retrying would change nothing

    def rewrite(s: RagState) -> RagState:
        variants = rewriter.variants(s["question"])
        nxt = s["attempt"] + 1
        query = variants[min(nxt, len(variants) - 1)]
        return {"query": query, "attempt": nxt, "trace": [*s["trace"], f"rewrite -> {query!r}"]}

    def answer(s: RagState) -> RagState:
        text = llm.invoke(build_prompt(s["question"], s["hits"]))
        ok, _ = check_citations(text, s["hits"])
        return {"answer": text if ok else "I do not know.", "trace": [*s["trace"], f"answer (citations ok: {ok})"]}

    def give_up(s: RagState) -> RagState:
        return {"answer": "I do not know.", "trace": [*s["trace"], "give_up"]}

    g = StateGraph(RagState)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite", rewrite)
    g.add_node("answer", answer)
    g.add_node("give_up", give_up)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", grade, ["answer", "rewrite", "give_up"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", END)
    g.add_edge("give_up", END)
    return g.compile()


def ask(graph, question: str, tenant_id: str, roles: frozenset[str] = frozenset()) -> RagState:  # noqa: ANN001
    return graph.invoke(
        {"question": question, "query": question, "tenant_id": tenant_id, "roles": roles, "attempt": 0, "trace": []},
        {"recursion_limit": 25},
    )
