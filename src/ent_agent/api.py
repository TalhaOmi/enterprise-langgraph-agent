"""HTTP API (FastAPI). This is the enforcement perimeter around the graph.

Per request, in order: authenticate -> kill switch -> rate limit -> token budget -> run graph.
Approvals are a separate endpoint restricted to the `approver` role, same tenant, never the
requester (separation of duties), with an expiry.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from langgraph.types import Command
from pydantic import BaseModel, Field

from .audit import AuditLog
from .auth import AuthError, Principal, verify_token
from .config import Settings
from .controls import kill_switch_active
from .graph.builder import build_graph, make_checkpointer
from .graph.nodes import _text
from .guardrails.limits import RateLimiter, TokenBudget
from .logging_setup import configure_logging
from .services import Services

_THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_STATIC = Path(__file__).parent / "static"
_UI_FILES = {
    "": ("chat.html", "text/html"),
    "/chat.js": ("chat.js", "text/javascript"),
    "/chat.css": ("chat.css", "text/css"),
}
# Strict CSP for the chat page: only our own script and styles, no inline code, no framing, same-origin requests only.
_UI_CSP = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = None


class ChatOut(BaseModel):
    request_id: str
    thread_id: str
    status: str  # completed | blocked | pending_approval
    answer: str | None = None
    approval_id: str | None = None
    actions: list[dict] | None = None


class ApprovalIn(BaseModel):
    approve: bool
    reason: str = Field(default="", max_length=300)


@dataclass
class PendingApproval:
    approval_id: str
    thread_key: str
    tenant_id: str
    requester: str
    created_at: float
    actions: list[dict]


class ApprovalRegistry:
    """Maps approval ids -> threads. In production persist this (Postgres table)."""

    def __init__(self) -> None:
        self._items: dict[str, PendingApproval] = {}
        self._lock = threading.Lock()

    def add(self, item: PendingApproval) -> None:
        with self._lock:
            self._items[item.approval_id] = item

    def get(self, approval_id: str) -> PendingApproval | None:
        with self._lock:
            return self._items.get(approval_id)

    def pop(self, approval_id: str) -> PendingApproval | None:
        with self._lock:
            return self._items.pop(approval_id, None)

    def for_tenant(self, tenant_id: str) -> list[PendingApproval]:
        with self._lock:
            return [i for i in self._items.values() if i.tenant_id == tenant_id]


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    graph=None,
    audit: AuditLog | None = None,
) -> FastAPI:
    settings = settings or Settings()
    services = services or Services.demo()
    audit = audit or AuditLog(settings.audit_log_path)
    stack = ExitStack()
    graph = graph or build_graph(settings, services, audit, make_checkpointer(settings, stack))
    limiter = RateLimiter(settings.rate_limit_per_minute)
    budget = TokenBudget(settings.daily_token_budget_per_user)
    approvals = ApprovalRegistry()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        configure_logging()
        yield
        stack.close()

    app = FastAPI(title="Enterprise LangGraph Agent", version="0.1.0", lifespan=lifespan,
                  docs_url=None if settings.env == "prod" else "/docs", redoc_url=None)  # fmt: skip
    app.state.audit, app.state.settings, app.state.approvals = audit, settings, approvals

    # -- plumbing -------------------------------------------------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unhandled(_: Request, __: Exception):
        # never leak stack traces / internals to clients; details are in server logs
        return JSONResponse({"detail": "internal error"}, status_code=500)

    def principal_dep(authorization: str | None = Header(default=None)) -> Principal:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            return verify_token(settings, authorization.split(" ", 1)[1])
        except AuthError:
            raise HTTPException(401, "invalid or expired token") from None

    def admit(principal: Principal, request_id: str) -> None:
        if kill_switch_active(settings):
            raise HTTPException(503, "service temporarily disabled")
        if not limiter.allow(f"{principal.tenant_id}:{principal.user_id}"):
            audit.record(
                "rate_limited", request_id=request_id, user_id=principal.user_id, tenant_id=principal.tenant_id
            )
            raise HTTPException(429, "rate limit exceeded")
        if not budget.has_budget(f"{principal.tenant_id}:{principal.user_id}"):
            audit.record(
                "budget_exhausted", request_id=request_id, user_id=principal.user_id, tenant_id=principal.tenant_id
            )
            raise HTTPException(429, "daily token budget exhausted")

    def finish(config: dict, principal: Principal, thread_key: str, request_id: str, thread_id: str) -> ChatOut:
        snap = graph.get_state(config)
        values = snap.values
        budget.charge(f"{principal.tenant_id}:{principal.user_id}", values.get("tokens_used", 0))
        interrupts = [i for t in snap.tasks for i in t.interrupts]
        if snap.next and interrupts:
            payload = interrupts[0].value
            approval_id = uuid.uuid4().hex
            approvals.add(
                PendingApproval(
                    approval_id, thread_key, principal.tenant_id, principal.user_id, time.time(), payload["actions"]
                )
            )
            return ChatOut(request_id=request_id, thread_id=thread_id, status="pending_approval",
                           approval_id=approval_id, actions=payload["actions"])  # fmt: skip
        answer = _text(values["messages"][-1].content)
        blocked = values.get("blocked")
        status = "blocked" if blocked and blocked != "output_retry" else "completed"
        return ChatOut(request_id=request_id, thread_id=thread_id, status=status, answer=answer)

    # -- chat UI (static files; the page itself holds no secrets and needs a bearer token to do anything) --
    def _ui(suffix: str) -> Response:
        name, ctype = _UI_FILES[suffix]
        return Response((_STATIC / name).read_bytes(), media_type=ctype, headers={"Content-Security-Policy": _UI_CSP})

    @app.get("/ui", include_in_schema=False)
    def ui_page() -> Response:
        return _ui("")

    @app.get("/ui/chat.js", include_in_schema=False)
    def ui_js() -> Response:
        return _ui("/chat.js")

    @app.get("/ui/chat.css", include_in_schema=False)
    def ui_css() -> Response:
        return _ui("/chat.css")

    # -- endpoints ------------------------------------------------------------------------
    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict:
        if kill_switch_active(settings):
            raise HTTPException(503, "kill switch active")
        return {"status": "ready"}

    @app.post("/v1/chat", response_model=ChatOut)
    def chat(body: ChatIn, request: Request, principal: Principal = Depends(principal_dep)) -> ChatOut:
        rid = request.state.request_id
        admit(principal, rid)
        thread_id = body.thread_id or uuid.uuid4().hex
        if not _THREAD_RE.match(thread_id):
            raise HTTPException(422, "invalid thread_id")
        # Namespace by tenant+user: a guessed thread id can never reach someone else's state.
        thread_key = f"{principal.tenant_id}:{principal.user_id}:{thread_id}"
        config = {"configurable": {"thread_id": thread_key}, "recursion_limit": 40}
        graph.invoke(
            {"messages": [("user", body.message)], "principal": principal.to_dict(), "request_id": rid},
            config,
        )
        return finish(config, principal, thread_key, rid, thread_id)

    @app.get("/v1/approvals")
    def list_approvals(principal: Principal = Depends(principal_dep)) -> list[dict]:
        if "approver" not in principal.roles:
            raise HTTPException(403, "approver role required")
        return [
            {
                "approval_id": a.approval_id,
                "requested_by": a.requester,
                "actions": a.actions,
                "age_s": int(time.time() - a.created_at),
            }
            for a in approvals.for_tenant(principal.tenant_id)
        ]

    @app.post("/v1/approvals/{approval_id}", response_model=ChatOut)
    def decide(
        approval_id: str, body: ApprovalIn, request: Request, principal: Principal = Depends(principal_dep)
    ) -> ChatOut:
        rid = request.state.request_id
        if "approver" not in principal.roles:
            raise HTTPException(403, "approver role required")
        item = approvals.get(approval_id)
        if item is None or item.tenant_id != principal.tenant_id:
            raise HTTPException(404, "approval not found")  # same answer for other tenants: no enumeration
        if item.requester == principal.user_id:
            audit.record(
                "approval_self_blocked", request_id=rid, user_id=principal.user_id, tenant_id=principal.tenant_id
            )
            raise HTTPException(403, "requester cannot approve their own action")
        approvals.pop(approval_id)
        expired = time.time() - item.created_at > settings.approval_timeout_seconds
        approve = body.approve and not expired
        reason = "expired" if expired else body.reason
        config = {"configurable": {"thread_id": item.thread_key}, "recursion_limit": 40}
        graph.invoke(Command(resume={"approved": approve, "approver": principal.user_id, "reason": reason}), config)
        requester = Principal(item.requester, item.tenant_id, frozenset())
        thread_id = item.thread_key.rsplit(":", 1)[1]
        return finish(config, requester, item.thread_key, rid, thread_id)

    return app
