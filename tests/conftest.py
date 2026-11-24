import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ent_agent.audit import AuditLog
from ent_agent.auth import Principal
from ent_agent.config import Settings
from ent_agent.graph.builder import build_graph
from ent_agent.services import Services


@pytest.fixture
def settings(tmp_path):
    return Settings(
        env="test",
        audit_log_path=str(tmp_path / "audit.jsonl"),
        kill_switch_file=str(tmp_path / ".kill"),
        rate_limit_per_minute=1000,
    )


@pytest.fixture
def services():
    return Services.demo()


@pytest.fixture
def audit(settings):
    return AuditLog(settings.audit_log_path)


@pytest.fixture
def graph(settings, services, audit):
    return build_graph(settings, services, audit, InMemorySaver())


def agent_principal(user="alice", tenant="acme", roles=("support_agent",)):
    return Principal(user, tenant, frozenset(roles))


def run(graph, text, principal=None, thread="t1", rid="req-1"):
    principal = principal or agent_principal()
    cfg = {"configurable": {"thread_id": f"{principal.tenant_id}:{principal.user_id}:{thread}"}, "recursion_limit": 40}
    graph.invoke({"messages": [("user", text)], "principal": principal.to_dict(), "request_id": rid}, cfg)
    return cfg, graph.get_state(cfg)


def last_text(state):
    return str(state.values["messages"][-1].content)
