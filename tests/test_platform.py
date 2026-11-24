"""Auth, config hardening, audit chain, limits, policy."""

import json

import pytest

from ent_agent.audit import AuditLog
from ent_agent.auth import AuthError, issue_token, verify_token
from ent_agent.config import Settings
from ent_agent.guardrails.limits import CircuitBreaker, CircuitOpenError, RateLimiter, TokenBudget
from ent_agent.guardrails.policy import authorize
from ent_agent.tools.registry import REGISTRY, RefundArgs
from tests.conftest import agent_principal


def test_prod_refuses_insecure_defaults():
    with pytest.raises(ValueError, match="Refusing to start in prod"):
        Settings(env="prod")
    ok = Settings(
        env="prod", jwt_secret="x" * 40, llm_model="anthropic:claude-sonnet-5-5", checkpoint_url="postgresql://u:p@h/db"
    )
    assert ok.env == "prod"


def test_jwt_roundtrip_and_rejections(settings):
    tok = issue_token(settings, "alice", "acme", ["support_agent"])
    p = verify_token(settings, tok)
    assert (p.user_id, p.tenant_id, p.roles) == ("alice", "acme", frozenset({"support_agent"}))
    with pytest.raises(AuthError):
        verify_token(settings, issue_token(settings, "a", "acme", [], ttl=-5))  # expired
    with pytest.raises(AuthError):
        verify_token(settings, tok + "x")  # bad signature
    other = settings.model_copy(update={"jwt_audience": "someone-else"})
    with pytest.raises(AuthError):
        verify_token(other, tok)  # wrong audience


def test_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "a.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.record("evt", n=i)
    assert AuditLog.verify(path) == (True, None)
    lines = path.read_text().splitlines()
    entry = json.loads(lines[2])
    entry["n"] = 999
    lines[2] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n")
    assert AuditLog.verify(path) == (False, 3)
    # deletion is detected too
    path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    assert AuditLog.verify(path)[0] is False


def test_audit_resumes_chain_after_restart(tmp_path):
    path = tmp_path / "a.jsonl"
    AuditLog(path).record("one")
    AuditLog(path).record("two")
    assert AuditLog.verify(path) == (True, None)


def test_rate_limiter_window():
    t = [0.0]
    rl = RateLimiter(2, 60, clock=lambda: t[0])
    assert rl.allow("u") and rl.allow("u") and not rl.allow("u")
    assert rl.allow("other")
    t[0] = 61
    assert rl.allow("u")


def test_token_budget_resets_daily():
    t = [0.0]
    b = TokenBudget(100, clock=lambda: t[0])
    b.charge("u", 100)
    assert not b.has_budget("u")
    t[0] = 86_400
    assert b.has_budget("u")


def test_circuit_breaker_opens_and_recovers():
    t = [0.0]
    cb = CircuitBreaker(2, 30, clock=lambda: t[0])

    def boom():
        raise RuntimeError("x")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: 1)
    t[0] = 31
    assert cb.call(lambda: 1) == 1


def test_policy_default_deny_and_risk_tiers(settings):
    refund = REGISTRY["issue_refund"]
    args = RefundArgs(order_id="ORD-1001", amount="10.00", reason="because")
    assert not authorize(refund, args, agent_principal(roles=("viewer",)), settings).allowed
    assert not authorize(refund, args, agent_principal(roles=("approver",)), settings).allowed
    assert not authorize(refund, args, agent_principal(roles=()), settings).allowed
    d = authorize(refund, args, agent_principal(), settings)
    assert d.allowed and d.needs_approval
    big = RefundArgs(order_id="ORD-1001", amount="501.00", reason="because")
    assert not authorize(refund, big, agent_principal(), settings).allowed


def test_tool_args_are_strict():
    with pytest.raises(Exception):  # noqa: B017 - unknown field
        RefundArgs(order_id="ORD-1001", amount="1", reason="abc", tenant_id="globex")
    with pytest.raises(Exception):  # noqa: B017 - bad id / negative amount
        RefundArgs(order_id="1001; DROP TABLE", amount="-5", reason="abc")
