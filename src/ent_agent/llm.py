"""Model factory. Swap providers with one env var; keep the rest of the system unchanged."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings


def get_llm(settings: Settings, canary: str = "") -> BaseChatModel:
    if settings.llm_model == "fake":
        if settings.env == "prod":  # belt and braces; Settings already refuses this
            raise RuntimeError("fake model is not allowed in prod")
        from .testing import FakeAgentModel

        return FakeAgentModel(canary=canary)

    from langchain.chat_models import init_chat_model

    # temperature 0: deterministic-ish behaviour is easier to test, audit and reproduce.
    return init_chat_model(settings.llm_model, temperature=0, timeout=settings.llm_timeout_seconds, max_retries=2)
