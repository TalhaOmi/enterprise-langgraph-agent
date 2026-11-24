"""Module 20 part 2: a supervisor with two specialist agents, as a LangGraph graph.

START -> remember_or_recall -> supervisor -> dispatch <-> (orders | policy) -> compose -> END

Enterprise rules:
  * The supervisor only PLANS which specialist runs. It cannot call tools itself.
  * Each specialist has its own narrow job: `orders` reads orders of the caller's tenant; `policy` runs RAG.
  * tenant_id and user_id live in the graph state, set by the API from the verified token. No model output can change them.
  * Hand-offs are bounded (max_handoffs), so two agents cannot bounce work between each other forever.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .hybrid import hybrid_search
from .memory import MemoryStore
from .rag import VectorStore, build_prompt, check_citations

_ORDER = re.compile(r"\bORD-\d{3,}\b")
_REMEMBER = re.compile(r"^\s*remember that\s+(.+)$", re.IGNORECASE | re.DOTALL)
_POLICY_WORDS = ("policy", "refund", "shipping", "return", "warranty", "leave", "how long", "how many")


class MAState(TypedDict, total=False):
    question: str
    tenant_id: str
    user_id: str
    roles: frozenset[str]
    memory_note: str
    plan: list[str]
    handoffs: int
    parts: list[str]
    answer: str
    trace: list[str]


def plan_for(question: str) -> list[str]:
    """Rule-based supervisor. Swap for an LLM router later; keep the output a list of allowed names only."""
    plan = []
    if _ORDER.search(question):
        plan.append("orders")
    if any(w in question.lower() for w in _POLICY_WORDS) or not plan:
        plan.append("policy")
    return plan


def build_multi_agent_graph(store: VectorStore, orders: dict[str, dict[str, str]], llm, memory: MemoryStore, max_handoffs: int = 3):  # noqa: ANN001
    def entry(s: MAState) -> MAState:
        m = _REMEMBER.match(s["question"])
        if m:
            key = memory.remember(s["tenant_id"], s["user_id"], m.group(1))
            msg = "Saved." if key else "I could not save that."
            return {"answer": msg, "plan": [], "parts": [], "trace": ["memory.remember"]}
        return {"memory_note": memory.recall_for_prompt(s["tenant_id"], s["user_id"]), "trace": ["memory.recall"]}

    def after_entry(s: MAState) -> str:
        return "end" if s.get("answer") else "supervisor"

    def supervisor(s: MAState) -> MAState:
        plan = plan_for(s["question"])[:max_handoffs]
        return {"plan": plan, "handoffs": 0, "parts": [], "trace": [*s["trace"], f"supervisor plan={plan}"]}

    def dispatch(s: MAState) -> str:
        if s["plan"] and s["handoffs"] < max_handoffs:
            return s["plan"][0]
        return "compose"

    def orders_agent(s: MAState) -> MAState:
        ids = _ORDER.findall(s["question"])
        mine = orders.get(s["tenant_id"], {})  # only this tenant's orders are even reachable
        lines = [f"{i}: {mine[i]}" if i in mine else f"{i}: not found" for i in ids]
        return {"parts": [*s["parts"], "; ".join(lines)], "plan": s["plan"][1:], "handoffs": s["handoffs"] + 1, "trace": [*s["trace"], "orders"]}

    def policy_agent(s: MAState) -> MAState:
        hits = hybrid_search(store, s["question"], s["tenant_id"], s["roles"], k=3)
        note = s.get("memory_note", "")
        text = llm.invoke((note + "\n" if note else "") + build_prompt(s["question"], hits))  # the note is already fenced and sanitised
        ok, _ = check_citations(text, hits)
        return {"parts": [*s["parts"], text if ok else "I do not know."], "plan": s["plan"][1:], "handoffs": s["handoffs"] + 1, "trace": [*s["trace"], "policy"]}

    def compose(s: MAState) -> MAState:
        return {"answer": " | ".join(s["parts"]) or "I do not know.", "trace": [*s["trace"], "compose"]}

    g = StateGraph(MAState)
    for name, fn in [("entry", entry), ("supervisor", supervisor), ("orders", orders_agent), ("policy", policy_agent), ("compose", compose)]:
        g.add_node(name, fn)
    g.add_edge(START, "entry")
    g.add_conditional_edges("entry", after_entry, {"end": END, "supervisor": "supervisor"})
    g.add_conditional_edges("supervisor", dispatch, ["orders", "policy", "compose"])
    g.add_conditional_edges("orders", dispatch, ["orders", "policy", "compose"])
    g.add_conditional_edges("policy", dispatch, ["orders", "policy", "compose"])
    g.add_edge("compose", END)
    return g.compile()


def ask_multi(graph, question: str, tenant_id: str, user_id: str, roles: frozenset[str] = frozenset()) -> MAState:  # noqa: ANN001
    return graph.invoke({"question": question, "tenant_id": tenant_id, "user_id": user_id, "roles": roles, "trace": []}, {"recursion_limit": 25})
