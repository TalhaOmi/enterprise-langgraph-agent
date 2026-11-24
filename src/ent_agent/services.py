"""Business back-ends the tools talk to. In-memory stand-ins; replace with real clients.

Important properties to preserve when you replace them:
  * everything is tenant-scoped (the tenant comes from the verified Principal)
  * side-effecting operations are IDEMPOTENT (keyed by tool_call id) because LangGraph may
    re-run a node after an interrupt/resume or a crash recovery.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from decimal import Decimal

from .rag.store import KnowledgeStore, seed_demo


@dataclass
class OrderRepo:
    _orders: dict[tuple[str, str], dict] = field(default_factory=dict)
    _refunds: dict[str, dict] = field(default_factory=dict)  # idempotency_key -> refund
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def seed(self) -> None:
        self._orders[("acme", "ORD-1001")] = {
            "status": "delivered",
            "total": Decimal("120.00"),
            "refunded": Decimal("0"),
        }
        self._orders[("acme", "ORD-1002")] = {"status": "shipped", "total": Decimal("35.50"), "refunded": Decimal("0")}
        self._orders[("globex", "ORD-9001")] = {
            "status": "delivered",
            "total": Decimal("999.00"),
            "refunded": Decimal("0"),
        }

    def get(self, tenant: str, order_id: str) -> dict | None:
        return self._orders.get((tenant, order_id))

    def refund(self, tenant: str, order_id: str, amount: Decimal, idem_key: str) -> dict:
        with self._lock:
            if idem_key in self._refunds:  # replay -> same result, no second refund
                return self._refunds[idem_key]
            order = self._orders.get((tenant, order_id))
            if order is None:
                raise LookupError("order not found")
            if order["status"] != "delivered":
                raise ValueError("only delivered orders can be refunded")
            if order["refunded"] + amount > order["total"]:
                raise ValueError("refund exceeds order total")
            order["refunded"] += amount
            record = {"order_id": order_id, "amount": str(amount), "refund_id": f"RF-{len(self._refunds) + 1:04d}"}
            self._refunds[idem_key] = record
            return record


@dataclass
class TicketRepo:
    _tickets: dict[str, dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, tenant: str, user: str, summary: str, priority: str, idem_key: str) -> dict:
        with self._lock:
            if idem_key in self._tickets:
                return self._tickets[idem_key]
            ticket = {"ticket_id": f"TCK-{len(self._tickets) + 1:04d}", "tenant": tenant, "by": user,
                      "summary": summary, "priority": priority}  # fmt: skip
            self._tickets[idem_key] = ticket
            return ticket


@dataclass
class Services:
    store: KnowledgeStore
    orders: OrderRepo
    tickets: TicketRepo

    @classmethod
    def demo(cls) -> Services:
        store, orders = KnowledgeStore(), OrderRepo()
        seed_demo(store)
        orders.seed()
        return cls(store, orders, TicketRepo())
