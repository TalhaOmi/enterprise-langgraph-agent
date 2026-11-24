"""Typed, validated configuration. Every knob is an env var prefixed AGENT_.

Rule: production must refuse to start with insecure defaults (fail closed).
"""

from __future__ import annotations

from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_SECRET = "dev-only-secret-change-me-32-bytes-min!!"  # noqa: S105


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    # --- model -------------------------------------------------------------
    # "fake" = deterministic offline model. Otherwise "provider:model",
    # e.g. "anthropic:claude-sonnet-5-5" or "openai:gpt-4o".
    llm_model: str = "fake"
    llm_timeout_seconds: float = 30.0

    # --- authn / authz -----------------------------------------------------
    jwt_secret: SecretStr = SecretStr(DEV_SECRET)
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "ent-idp"
    jwt_audience: str = "ent-agent"
    jwt_ttl_seconds: int = 900  # short-lived tokens (15 min)

    # --- input / output guardrails ----------------------------------------
    max_input_chars: int = 8_000
    max_output_chars: int = 6_000
    max_tool_result_chars: int = 4_000
    max_output_retries: int = 1
    max_tool_calls_per_step: int = 5

    # --- runaway / cost controls (OWASP LLM10) ----------------------------
    max_agent_steps: int = 6
    max_tokens_per_request: int = 8_000
    daily_token_budget_per_user: int = 200_000
    rate_limit_per_minute: int = 30
    breaker_failure_threshold: int = 5
    breaker_reset_seconds: float = 30.0

    # --- human-in-the-loop -------------------------------------------------
    approval_timeout_seconds: int = 300
    refund_hard_limit: float = 500.0  # above this the policy denies outright

    # --- persistence / audit ----------------------------------------------
    checkpoint_url: str | None = None  # postgres URL; None => in-memory (dev/test only)
    audit_log_path: str = "audit.log.jsonl"
    kill_switch_file: str = ".kill_switch"
    kill_switch: bool = False

    # --- network egress ----------------------------------------------------
    allowed_fetch_domains: list[str] = []

    @model_validator(mode="after")
    def _prod_must_be_hardened(self) -> Settings:
        if self.env != "prod":
            return self
        problems = []
        secret = self.jwt_secret.get_secret_value()
        if secret == DEV_SECRET or len(secret) < 32:
            problems.append("AGENT_JWT_SECRET must be set to a random value of >= 32 chars")
        if self.llm_model == "fake":
            problems.append("AGENT_LLM_MODEL must be a real provider model in prod")
        if not self.checkpoint_url:
            problems.append("AGENT_CHECKPOINT_URL (durable checkpointer) is required in prod")
        if problems:
            raise ValueError("Refusing to start in prod: " + "; ".join(problems))
        return self
