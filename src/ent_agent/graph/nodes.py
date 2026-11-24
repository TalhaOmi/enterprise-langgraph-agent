"""Graph nodes. Each node is a small, single-purpose control point.

    input_guard -> agent <-> authorize_tools -> [approval_gate] -> run_tools -> agent
                     \\-> output_guard -> END          (blocked runs -> refuse -> END)

LangGraph rules this file follows (they are the usual production footguns):
  * A node that calls `interrupt()` is RE-EXECUTED FROM ITS FIRST LINE on resume. So the
    approval node does no side effects and writes no audit record before the interrupt;
    side effects live in `run_tools`, which is idempotent per tool_call id.
  * Every tool_call gets exactly one ToolMessage (denied / rejected / executed).
  * All loops are bounded (max steps, token cap, recursion_limit set by the caller).
  * Identity (`principal`) is read from state that the API fills from the verified token.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import interrupt
from pydantic import ValidationError

from ..audit import AuditLog, fingerprint
from ..auth import Principal
from ..config import Settings
from ..controls import kill_switch_active
from ..guardrails.input_guard import check_input
from ..guardrails.limits import CircuitBreaker, CircuitOpenError
from ..guardrails.output_guard import check_output
from ..guardrails.policy import authorize
from ..services import Services
from ..tools.registry import REGISTRY, ToolContext
from .prompts import REFUSAL, UNAVAILABLE, build_system_prompt

log = logging.getLogger("ent_agent.graph")
_MAX_HISTORY = 30


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def _is_real_human(m: BaseMessage) -> bool:
    return isinstance(m, HumanMessage) and not m.additional_kwargs.get("guardrail")


def _current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    start = max((i for i, m in enumerate(messages) if _is_real_human(m)), default=0)
    return messages[start:]


def _trim(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Bound context growth; never start mid tool-exchange (providers reject orphan tool msgs)."""
    window = messages[-_MAX_HISTORY:]
    while window and not isinstance(window[0], HumanMessage):
        window = window[1:]
    return window


class AgentNodes:
    def __init__(self, settings: Settings, services: Services, llm: BaseChatModel, audit: AuditLog, canary: str):
        self.s, self.services, self.llm, self.audit, self.canary = settings, services, llm, audit, canary
        self.breaker = CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_reset_seconds)
        self.system_prompt = build_system_prompt(canary)

    # -- helpers ---------------------------------------------------------------------------
    def _who(self, state: dict) -> dict:
        p = state.get("principal", {})
        return {"request_id": state.get("request_id"), "user_id": p.get("user_id"), "tenant_id": p.get("tenant_id")}

    # -- 1. input guard --------------------------------------------------------------------
    def input_guard(self, state: dict) -> dict:
        last = state["messages"][-1]
        verdict = check_input(_text(last.content), max_chars=self.s.max_input_chars)
        self.audit.record(
            "input_check", **self._who(state), allowed=verdict.allowed, reasons=verdict.reasons,
            pii_kinds=verdict.pii_kinds, content_fp=fingerprint(_text(last.content)),
        )  # fmt: skip
        reset = {"steps": 0, "tokens_used": 0, "output_retries": 0, "ready_calls": [], "pending_calls": []}
        if not verdict.allowed:
            return {**reset, "blocked": ",".join(verdict.reasons)}
        # same id => add_messages REPLACES the raw message with the redacted one
        return {**reset, "blocked": None, "messages": [HumanMessage(content=verdict.text, id=last.id)]}

    def refuse(self, state: dict) -> dict:
        # Generic refusal: never tell an attacker which rule fired (that is in the audit log).
        return {"messages": [AIMessage(content=REFUSAL)]}

    # -- 2. agent --------------------------------------------------------------------------
    def agent(self, state: dict) -> dict:
        if kill_switch_active(self.s):
            self.audit.record("kill_switch_block", **self._who(state))
            return {"blocked": "kill_switch", "messages": [AIMessage(content=UNAVAILABLE)]}
        steps = state.get("steps", 0) + 1
        if steps > self.s.max_agent_steps:
            self.audit.record("step_limit", **self._who(state))
            return {"blocked": "step_limit", "steps": steps, "messages": [AIMessage(content=UNAVAILABLE)]}
        if state.get("tokens_used", 0) > self.s.max_tokens_per_request:
            self.audit.record("token_cap", **self._who(state))
            return {"blocked": "token_cap", "messages": [AIMessage(content=UNAVAILABLE)]}
        prompt = [SystemMessage(content=self.system_prompt), *_trim(state["messages"])]
        try:
            ai: AIMessage = self.breaker.call(lambda: self.llm.invoke(prompt))
        except CircuitOpenError:
            self.audit.record("llm_circuit_open", **self._who(state))
            return {"blocked": "llm_unavailable", "messages": [AIMessage(content=UNAVAILABLE)]}
        except Exception:  # noqa: BLE001 - provider errors must not leak details to users
            log.exception("llm call failed")
            self.audit.record("llm_error", **self._who(state))
            return {"blocked": "llm_error", "messages": [AIMessage(content=UNAVAILABLE)]}
        usage = (ai.usage_metadata or {}).get("total_tokens") or len(_text(ai.content)) // 4 + 1
        self.audit.record(
            "llm_call", **self._who(state), step=steps, tokens=usage,
            tool_calls=[c["name"] for c in ai.tool_calls],
        )  # fmt: skip
        return {"messages": [ai], "steps": steps, "blocked": None, "tokens_used": state.get("tokens_used", 0) + usage}

    # -- 3. authorize tool calls -----------------------------------------------------------
    def authorize_tools(self, state: dict) -> dict:
        ai: AIMessage = state["messages"][-1]
        principal = Principal.from_dict(state["principal"])
        ready: list[dict] = []
        pending: list[dict] = []
        replies: list[ToolMessage] = []

        def deny(call: dict, reason: str) -> None:
            self.audit.record("tool_denied", **self._who(state), tool=call["name"], reason=reason)
            replies.append(
                ToolMessage(content=f"DENIED: {reason}", tool_call_id=call["id"], name=call["name"], status="error")
            )

        for i, call in enumerate(ai.tool_calls):
            if i >= self.s.max_tool_calls_per_step:
                deny(call, "too many tool calls in one step")
                continue
            spec = REGISTRY.get(call["name"])
            if spec is None:
                deny(call, "unknown tool")
                continue
            try:
                args = spec.args_model.model_validate(call["args"])
            except ValidationError as exc:
                bad = ", ".join(sorted({str(e["loc"][0]) if e["loc"] else "args" for e in exc.errors()}))
                deny(call, f"invalid arguments ({bad})")
                continue
            decision = authorize(spec, args, principal, self.s)
            if not decision.allowed:
                deny(call, decision.reason)
                continue
            record = {"id": call["id"], "name": spec.name, "args": args.model_dump(mode="json")}
            if decision.needs_approval:
                self.audit.record(
                    "approval_requested", **self._who(state), tool=spec.name, call_id=call["id"], args=record["args"]
                )
                pending.append(record)
            else:
                ready.append(record)
        return {"ready_calls": ready, "pending_calls": pending, "messages": replies}

    # -- 4. human approval gate ------------------------------------------------------------
    def approval_gate(self, state: dict) -> dict:
        pending = state["pending_calls"]
        requester = state["principal"]["user_id"]
        # ---- execution pauses here; on resume this function restarts from the top ----
        decision = interrupt({
            "type": "approval_required",
            "requested_by": requester,
            "tenant_id": state["principal"]["tenant_id"],
            "actions": [{"id": c["id"], "tool": c["name"], "args": c["args"]} for c in pending],
        })  # fmt: skip
        approver = decision.get("approver")
        approved = bool(decision.get("approved")) and approver is not None and approver != requester
        self.audit.record(
            "approval_decision", **self._who(state), approved=approved, approver=approver,
            reason=decision.get("reason", ""), call_ids=[c["id"] for c in pending],
        )  # fmt: skip
        if approved:
            return {"ready_calls": [*state.get("ready_calls", []), *pending], "pending_calls": []}
        replies = [
            ToolMessage(
                content="REJECTED by human approver; action not performed.",
                tool_call_id=c["id"],
                name=c["name"],
                status="error",
            )
            for c in pending
        ]
        return {"pending_calls": [], "messages": replies}

    # -- 5. run tools ----------------------------------------------------------------------
    def run_tools(self, state: dict) -> dict:
        principal = Principal.from_dict(state["principal"])
        replies: list[ToolMessage] = []
        for call in state.get("ready_calls", []):
            spec = REGISTRY[call["name"]]
            ctx = ToolContext(principal, call["id"], self.services, self.s)
            try:
                result = spec.handler(spec.args_model.model_validate(call["args"]), ctx)
                status = "success"
            except Exception:  # noqa: BLE001 - tool bugs must not crash or leak into the model
                log.exception("tool %s failed", spec.name)
                result, status = "Tool failed unexpectedly.", "error"
            result = result[: self.s.max_tool_result_chars]
            self.audit.record(
                "tool_executed", **self._who(state), tool=spec.name, call_id=call["id"], status=status,
                args_fp=fingerprint(str(call["args"])),
            )  # fmt: skip
            replies.append(ToolMessage(content=result, tool_call_id=call["id"], name=spec.name, status=status))
        return {"ready_calls": [], "messages": replies}

    # -- 6. output guard -------------------------------------------------------------------
    def output_guard(self, state: dict) -> dict:
        last: AIMessage = state["messages"][-1]
        turn = _current_turn(state["messages"])
        retrieved: set[str] = set()
        for m in turn:
            if isinstance(m, ToolMessage) and m.name == "search_knowledge_base":
                retrieved |= set(re.findall(r'<retrieved_document id="([^"]+)"', _text(m.content)))
        v = check_output(
            _text(last.content), canary=self.canary, allowed_doc_ids=retrieved,
            require_citation=False, max_chars=self.s.max_output_chars,
        )  # fmt: skip
        self.audit.record("output_check", **self._who(state), ok=v.ok, hard_fails=v.hard_fails, notes=v.notes)
        if v.ok:
            return {"messages": [AIMessage(content=v.text, id=last.id)], "blocked": None}
        retries = state.get("output_retries", 0)
        # Replace the bad answer in history so the leak/secret is not re-sent to the model.
        withheld = AIMessage(content="[response withheld by guardrail]", id=last.id)
        if retries < self.s.max_output_retries:
            fix = HumanMessage(
                content=f"[guardrail] Your previous answer was rejected ({', '.join(v.hard_fails)}). "
                "Answer again using only the user's request and cited documents.",
                additional_kwargs={"guardrail": True},
            )
            return {"messages": [withheld, fix], "output_retries": retries + 1, "blocked": "output_retry"}
        return {"messages": [AIMessage(content=UNAVAILABLE, id=last.id)], "blocked": "output_fallback"}
