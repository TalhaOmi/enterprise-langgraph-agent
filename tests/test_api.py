import pytest
from fastapi.testclient import TestClient

from ent_agent.api import create_app
from ent_agent.auth import issue_token
from ent_agent.config import Settings


@pytest.fixture
def ctx(settings, services, audit):
    from langgraph.checkpoint.memory import InMemorySaver

    from ent_agent.graph.builder import build_graph

    graph = build_graph(settings, services, audit, InMemorySaver())
    app = create_app(settings, services, graph, audit)
    return TestClient(app), settings, services


def hdr(settings, user="alice", tenant="acme", roles=("support_agent",)):
    return {"Authorization": f"Bearer {issue_token(settings, user, tenant, list(roles))}"}


def test_auth_required(ctx):
    c, s, _ = ctx
    assert c.post("/v1/chat", json={"message": "hi"}).status_code == 401
    assert c.post("/v1/chat", json={"message": "hi"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    expired = {"Authorization": "Bearer " + issue_token(s, "a", "acme", ["support_agent"], ttl=-1)}
    assert c.post("/v1/chat", json={"message": "hi"}, headers=expired).status_code == 401


def test_chat_and_security_headers(ctx):
    c, s, _ = ctx
    r = c.post("/v1/chat", json={"message": "status of order ORD-1001"}, headers=hdr(s))
    assert r.status_code == 200 and r.json()["status"] == "completed"
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Request-ID"]


def test_full_approval_flow_with_separation_of_duties(ctx):
    c, s, services = ctx
    r = c.post(
        "/v1/chat", json={"message": "refund 20.00 for order ORD-1001", "thread_id": "t9"}, headers=hdr(s)
    ).json()
    assert r["status"] == "pending_approval" and r["actions"][0]["tool"] == "issue_refund"
    aid = r["approval_id"]
    # requester cannot approve (even if they had the role), non-approvers cannot, other tenants cannot
    assert (
        c.post(
            f"/v1/approvals/{aid}", json={"approve": True}, headers=hdr(s, "alice", roles=("support_agent", "approver"))
        ).status_code
        == 403
    )
    assert c.post(f"/v1/approvals/{aid}", json={"approve": True}, headers=hdr(s, "carol")).status_code == 403
    assert (
        c.post(
            f"/v1/approvals/{aid}", json={"approve": True}, headers=hdr(s, "mallory", "globex", ("approver",))
        ).status_code
        == 404
    )
    assert str(services.orders.get("acme", "ORD-1001")["refunded"]) == "0"
    ok = c.post(f"/v1/approvals/{aid}", json={"approve": True}, headers=hdr(s, "bob", roles=("approver",)))
    assert ok.status_code == 200 and ok.json()["status"] == "completed"
    assert str(services.orders.get("acme", "ORD-1001")["refunded"]) == "20.00"
    # the approval is single-use
    assert (
        c.post(f"/v1/approvals/{aid}", json={"approve": True}, headers=hdr(s, "bob", roles=("approver",))).status_code
        == 404
    )


def test_expired_approval_is_treated_as_rejection(ctx, monkeypatch):
    c, s, services = ctx
    c.app.state.settings.__dict__["approval_timeout_seconds"] = -1
    r = c.post("/v1/chat", json={"message": "refund 20.00 for order ORD-1001"}, headers=hdr(s)).json()
    out = c.post(
        f"/v1/approvals/{r['approval_id']}", json={"approve": True}, headers=hdr(s, "bob", roles=("approver",))
    )
    assert out.status_code == 200
    assert str(services.orders.get("acme", "ORD-1001")["refunded"]) == "0"


def test_threads_are_namespaced_per_user(ctx):
    c, s, _ = ctx
    c.post("/v1/chat", json={"message": "status of order ORD-1001", "thread_id": "shared"}, headers=hdr(s, "alice"))
    r = c.post("/v1/chat", json={"message": "hello", "thread_id": "shared"}, headers=hdr(s, "mallory"))
    assert "ORD-1001" not in r.text  # mallory gets a fresh thread, not alice's history


def test_thread_id_validation(ctx):
    c, s, _ = ctx
    assert c.post("/v1/chat", json={"message": "hi", "thread_id": "../../etc"}, headers=hdr(s)).status_code == 422


def test_kill_switch_returns_503(ctx):
    c, s, _ = ctx
    open(s.kill_switch_file, "w").close()
    assert c.post("/v1/chat", json={"message": "hi"}, headers=hdr(s)).status_code == 503
    assert c.get("/readyz").status_code == 503
    assert c.get("/healthz").status_code == 200  # liveness stays up


def test_rate_limit(settings, services, audit):
    s = Settings(**{**settings.model_dump(), "rate_limit_per_minute": 2})
    c = TestClient(create_app(s, services, audit=audit))
    codes = [c.post("/v1/chat", json={"message": "hi"}, headers=hdr(s)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_input_size_limit_enforced_by_schema(ctx):
    c, s, _ = ctx
    assert c.post("/v1/chat", json={"message": "a" * 25_000}, headers=hdr(s)).status_code == 422


def test_chat_ui_is_served_with_strict_csp(ctx):
    c, _, _ = ctx
    page = c.get("/ui")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    csp = page.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp
    assert "<script>" not in page.text  # no inline script: the CSP would block it anyway
    assert c.get("/ui/chat.js").status_code == 200 and c.get("/ui/chat.css").status_code == 200
    assert "innerHTML" not in c.get("/ui/chat.js").text  # model output must only ever be inserted as text


def test_chat_ui_needs_no_token_but_the_api_still_does(ctx):
    c, _, _ = ctx
    assert c.get("/ui").status_code == 200 and c.post("/v1/chat", json={"message": "hi"}).status_code == 401
