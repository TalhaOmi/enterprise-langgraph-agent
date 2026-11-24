"""Policy-as-code: who may call which tool, with which arguments, and what needs a human.

One place, reviewable by security/compliance, unit-tested. Default deny.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel

from ..auth import Principal
from ..config import Settings
from ..tools.registry import Risk, ToolSpec

# Capability-based access: the role decides which tools the model may even *attempt*.
ROLE_TOOLS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"search_knowledge_base"}),
    "support_agent": frozenset({
        "search_knowledge_base", "get_order_status", "create_support_ticket", "issue_refund", "fetch_url",
    }),
    "approver": frozenset(),  # approvers approve; they do not drive the agent's tools
}  # fmt: skip


@dataclass(frozen=True)
class Decision:
    allowed: bool
    needs_approval: bool = False
    reason: str = ""


ArgPolicy = Callable[[BaseModel, Principal, Settings], str | None]


def _refund_limit(args: BaseModel, _p: Principal, settings: Settings) -> str | None:
    amount: Decimal = args.amount  # type: ignore[attr-defined]
    if amount > Decimal(str(settings.refund_hard_limit)):
        return f"refund above hard limit of {settings.refund_hard_limit}"
    return None


ARG_POLICIES: dict[str, list[ArgPolicy]] = {"issue_refund": [_refund_limit]}


def authorize(spec: ToolSpec, args: BaseModel, principal: Principal, settings: Settings) -> Decision:
    allowed_tools: set[str] = set()
    for role in principal.roles:
        allowed_tools |= ROLE_TOOLS.get(role, frozenset())
    if spec.name not in allowed_tools:
        return Decision(False, reason="role not permitted to use this tool")
    for policy in ARG_POLICIES.get(spec.name, []):
        problem = policy(args, principal, settings)
        if problem:
            return Decision(False, reason=problem)
    if spec.risk is Risk.HIGH:
        return Decision(True, needs_approval=True, reason="high-risk action requires human approval")
    return Decision(True)
