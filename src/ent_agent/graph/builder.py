"""Assemble the LangGraph state machine and the checkpointer."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..audit import AuditLog
from ..config import Settings
from ..llm import get_llm
from ..services import Services
from ..tools.registry import langchain_tools
from .nodes import AgentNodes
from .state import AgentState


def make_canary(settings: Settings) -> str:
    seed = settings.jwt_secret.get_secret_value() + "canary"
    return "CNRY-" + hashlib.sha256(seed.encode()).hexdigest()[:12]


def make_checkpointer(settings: Settings, stack: ExitStack) -> BaseCheckpointSaver:
    """Durable state is what makes approvals survive restarts and deploys.

    In-memory is for dev/tests only (Settings refuses to start in prod without a URL).
    """
    if not settings.checkpoint_url:
        return InMemorySaver()
    from langgraph.checkpoint.postgres import PostgresSaver  # pip install ".[postgres]"

    saver = stack.enter_context(PostgresSaver.from_conn_string(settings.checkpoint_url))
    saver.setup()  # creates tables on first run (idempotent)
    return saver


def build_graph(
    settings: Settings,
    services: Services,
    audit: AuditLog,
    checkpointer: BaseCheckpointSaver,
    llm: BaseChatModel | None = None,
):
    canary = make_canary(settings)
    base = llm or get_llm(settings, canary)
    nodes = AgentNodes(settings, services, base.bind_tools(langchain_tools()), audit, canary)

    g = StateGraph(AgentState)
    g.add_node("input_guard", nodes.input_guard)
    g.add_node("refuse", nodes.refuse)
    g.add_node("agent", nodes.agent)
    g.add_node("authorize_tools", nodes.authorize_tools)
    g.add_node("approval_gate", nodes.approval_gate)
    g.add_node("run_tools", nodes.run_tools)
    g.add_node("output_guard", nodes.output_guard)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: "refuse" if s.get("blocked") else "agent", ["refuse", "agent"])
    g.add_edge("refuse", END)

    def after_agent(s: dict) -> str:
        if s.get("blocked"):
            return END
        return "authorize_tools" if s["messages"][-1].tool_calls else "output_guard"

    g.add_conditional_edges("agent", after_agent, ["authorize_tools", "output_guard", END])

    def after_authorize(s: dict) -> str:
        if s.get("pending_calls"):
            return "approval_gate"
        return "run_tools" if s.get("ready_calls") else "agent"

    g.add_conditional_edges("authorize_tools", after_authorize, ["approval_gate", "run_tools", "agent"])
    g.add_conditional_edges(
        "approval_gate", lambda s: "run_tools" if s.get("ready_calls") else "agent", ["run_tools", "agent"]
    )
    g.add_edge("run_tools", "agent")
    g.add_conditional_edges(
        "output_guard", lambda s: "agent" if s.get("blocked") == "output_retry" else END, ["agent", END]
    )
    return g.compile(checkpointer=checkpointer)
