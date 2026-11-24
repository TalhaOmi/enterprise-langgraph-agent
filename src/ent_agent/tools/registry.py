"""Tool registry: every tool declares a risk tier, a strict args schema, and a handler.

Rules enforced by design (OWASP LLM06 excessive agency):
  * strict schemas: `extra="forbid"`, bounded strings, regex-validated ids, Decimal money
  * identity (tenant/user) is injected by the runtime from the verified token - it is NOT a
    tool argument, so the model cannot ask for another tenant's data
  * side effects are idempotent via `ctx.idempotency_key` (the tool_call id)
  * tool output is untrusted data going back into the model: it is length-capped
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ..auth import Principal
from ..config import Settings
from ..guardrails.net import UnsafeURL, validate_url
from ..rag.store import render_for_llm
from ..services import Services


class Risk(StrEnum):
    READ = "read"
    WRITE = "write"
    HIGH = "high"  # irreversible / financial / external effect => human approval


@dataclass(frozen=True)
class ToolContext:
    principal: Principal
    idempotency_key: str
    services: Services
    settings: Settings


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


OrderId = Field(pattern=r"^ORD-\d{4,8}$", description="Order id like ORD-1001")


class SearchArgs(_Strict):
    query: str = Field(min_length=2, max_length=200)


class OrderArgs(_Strict):
    order_id: str = OrderId


class TicketArgs(_Strict):
    summary: str = Field(min_length=5, max_length=300)
    priority: Literal["low", "normal", "high"] = "normal"


class RefundArgs(_Strict):
    order_id: str = OrderId
    amount: Decimal = Field(gt=0, le=10_000, decimal_places=2)
    reason: str = Field(min_length=3, max_length=200)


class FetchArgs(_Strict):
    url: str = Field(max_length=500)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    risk: Risk
    handler: Callable[[BaseModel, ToolContext], str]

    def as_langchain_tool(self) -> StructuredTool:
        """Schema-only tool for `bind_tools`; execution goes through our guarded runtime."""

        def _never_called(**_: object) -> str:  # pragma: no cover
            raise RuntimeError("tools must run through the guarded tool node")

        return StructuredTool.from_function(
            func=_never_called, name=self.name, description=self.description, args_schema=self.args_model
        )


def _search(args: SearchArgs, ctx: ToolContext) -> str:
    docs = ctx.services.store.search(args.query, ctx.principal)
    return render_for_llm(docs)


def _order_status(args: OrderArgs, ctx: ToolContext) -> str:
    order = ctx.services.orders.get(ctx.principal.tenant_id, args.order_id)
    if order is None:
        return "Order not found."
    return f"Order {args.order_id}: status={order['status']}, total={order['total']}, refunded={order['refunded']}"


def _ticket(args: TicketArgs, ctx: ToolContext) -> str:
    t = ctx.services.tickets.create(
        ctx.principal.tenant_id, ctx.principal.user_id, args.summary, args.priority, ctx.idempotency_key
    )
    return f"Created ticket {t['ticket_id']} (priority {t['priority']})."


def _refund(args: RefundArgs, ctx: ToolContext) -> str:
    try:
        r = ctx.services.orders.refund(ctx.principal.tenant_id, args.order_id, args.amount, ctx.idempotency_key)
    except (LookupError, ValueError) as exc:
        return f"Refund failed: {exc}"
    return f"Refund {r['refund_id']} issued: {r['amount']} for {r['order_id']}."


def _fetch(args: FetchArgs, ctx: ToolContext) -> str:
    try:
        url = validate_url(args.url, ctx.settings.allowed_fetch_domains)
    except UnsafeURL as exc:
        return f"Fetch blocked: {exc}"
    try:
        with httpx.Client(timeout=5.0, follow_redirects=False) as client:  # redirects could escape the allowlist
            resp = client.get(url, headers={"Accept": "text/plain, text/html"})
        resp.raise_for_status()
    except httpx.HTTPError:
        return "Fetch failed."
    return "UNTRUSTED WEB CONTENT:\n" + resp.text[:20_000].replace("<", "‹")


REGISTRY: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        ToolSpec("search_knowledge_base", "Search company policy documents. Cite results as [doc:<id>].",
                 SearchArgs, Risk.READ, _search),
        ToolSpec("get_order_status", "Look up an order by id for the current customer tenant.",
                 OrderArgs, Risk.READ, _order_status),
        ToolSpec("create_support_ticket", "Open a support ticket.", TicketArgs, Risk.WRITE, _ticket),
        ToolSpec("issue_refund", "Refund an order. Requires human approval.", RefundArgs, Risk.HIGH, _refund),
        ToolSpec("fetch_url", "Fetch an allow-listed https URL.", FetchArgs, Risk.READ, _fetch),
    ]
}  # fmt: skip


def langchain_tools() -> list[StructuredTool]:
    return [s.as_langchain_tool() for s in REGISTRY.values()]
