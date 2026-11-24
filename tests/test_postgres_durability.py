"""Durability integration test: an approval must survive a full process restart.

Skipped unless AGENT_TEST_POSTGRES_URL is set, e.g.
    AGENT_TEST_POSTGRES_URL=postgresql://user:pass@localhost:5432/agent make test
"""

import os
import uuid
from contextlib import ExitStack

import pytest
from langgraph.types import Command

from ent_agent.audit import AuditLog
from ent_agent.config import Settings
from ent_agent.graph.builder import build_graph, make_checkpointer
from ent_agent.services import Services
from tests.conftest import agent_principal

URL = os.environ.get("AGENT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="AGENT_TEST_POSTGRES_URL not set")


def _boot(tmp_path, services):
    """Simulate a fresh process: new settings, new checkpointer connection, new compiled graph."""
    settings = Settings(
        env="test",
        checkpoint_url=URL,
        audit_log_path=str(tmp_path / "audit.jsonl"),
        kill_switch_file=str(tmp_path / ".kill"),
    )
    stack = ExitStack()
    graph = build_graph(settings, services, AuditLog(settings.audit_log_path), make_checkpointer(settings, stack))
    return graph, stack


def test_pending_refund_survives_restart_and_runs_exactly_once(tmp_path):
    services = Services.demo()
    p = agent_principal()
    cfg = {"configurable": {"thread_id": f"acme:alice:{uuid.uuid4().hex}"}, "recursion_limit": 40}

    graph1, stack1 = _boot(tmp_path, services)
    graph1.invoke(
        {
            "messages": [("user", "Please refund 20.00 for order ORD-1001")],
            "principal": p.to_dict(),
            "request_id": "r1",
        },
        cfg,
    )
    assert graph1.get_state(cfg).next == ("approval_gate",)
    stack1.close()  # "process dies": connection closed, in-memory graph discarded

    graph2, stack2 = _boot(tmp_path, services)  # "new process" reads state back from Postgres
    try:
        assert graph2.get_state(cfg).next == ("approval_gate",)
        assert str(services.orders.get("acme", "ORD-1001")["refunded"]) == "0"
        graph2.invoke(Command(resume={"approved": True, "approver": "bob"}), cfg)
        assert not graph2.get_state(cfg).next
        assert str(services.orders.get("acme", "ORD-1001")["refunded"]) == "20.00"
    finally:
        stack2.close()
