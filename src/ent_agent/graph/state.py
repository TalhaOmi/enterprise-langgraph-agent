from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    principal: dict[str, Any]  # from the verified JWT, re-supplied on every request
    request_id: str
    blocked: str | None  # reason code when the run is stopped by a guardrail/control
    steps: int  # agent loop iterations this turn
    tokens_used: int  # tokens consumed this turn
    output_retries: int
    ready_calls: list[dict[str, Any]]  # authorised, will run
    pending_calls: list[dict[str, Any]]  # authorised, waiting for a human
