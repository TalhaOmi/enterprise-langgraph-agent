# Enterprise launch checklist

The rules below come from articles I read while researching this project (linked at the bottom).
Each row says **where it lives in this repo** and whether it is done in code or needs *your*
infrastructure. Legend: ✅ implemented + tested here · 🔶 you must provide/operate it · ⬜ out of scope for the template.

## 1. Input guardrails (before the model)

| Rule | Status | Where |
|---|---|---|
| Normalise input (unicode, zero-width/control chars) and cap length per message | ✅ | `guardrails/input_guard.py`, `ChatIn` schema |
| Heuristic prompt-injection / jailbreak detection that returns a *risk label only* (never triggers tools) | ✅ | `input_guard.INJECTION_PATTERNS` |
| Never rely on validation alone (defence in depth) | ✅ | layers 2-6 below assume layer 1 fails |
| Redact PII/secrets **before** data reaches an external model | ✅ | `guardrails/pii.py`, `AgentNodes.input_guard` |
| Add an ML injection/PII detector (Lakera, LLM Guard, Presidio) behind the same interface | 🔶 | swap in `check_input` / `redact` |
| Treat retrieved content as untrusted: strip instruction-like text, tag provenance | ✅ | `rag/store.py` (`sanitize_untrusted`, `render_for_llm`) |

## 2. Tool / action safety (OWASP LLM06 excessive agency)

| Rule | Status | Where |
|---|---|---|
| Least privilege: role → tool allowlist (capability-based, default deny) | ✅ | `guardrails/policy.py` `ROLE_TOOLS` |
| Risk tiers (read / write / high) | ✅ | `tools/registry.py` `Risk` |
| Strict argument schemas: types, ranges, regex ids, reject unknown fields | ✅ | `tools/registry.py` (`extra="forbid"`) |
| Argument-level policy-as-code in one reviewable place | ✅ | `policy.ARG_POLICIES` (refund hard limit) |
| Identity comes from the verified token, never from the model or tool args | ✅ | `ToolContext.principal` |
| Human approval (two-step commit) for irreversible/financial actions | ✅ | `approval_gate` node + `/v1/approvals` |
| Separation of duties: approver ≠ requester, same tenant, `approver` role, expiry | ✅ | `api.decide`, `approval_gate` |
| Idempotent side effects (nodes re-run after interrupt/resume) | ✅ | `services.py` idempotency keys |
| SSRF protection for URL tools: https, allowlist, block private/link-local/metadata IPs, no redirects, timeouts | ✅ | `guardrails/net.py`, `fetch_url` |
| Short-lived scoped credentials for downstream systems | 🔶 | use your IdP / workload identity; JWT TTL is 15 min here |

## 3. Output guardrails (after the model)

| Rule | Status | Where |
|---|---|---|
| Don't trust the model to self-censor; independent checks on every output | ✅ | `guardrails/output_guard.py` |
| System-prompt leakage detection (canary token) | ✅ | `make_canary`, `output_guard` |
| Secret detection = block; PII = redact (choose per workflow risk) | ✅ | `output_guard.check_output` |
| Strip active content (html, markdown-image exfiltration) | ✅ | `output_guard` |
| Groundedness: citations must reference documents retrieved this turn | ✅ | `[doc:id]` check |
| Self-correction loop with bounded retries, then safe fallback | ✅ | `output_guard` node + routing in `builder.py` |
| Clients must render output as text (OWASP LLM05: encode before rendering, parameterise queries) | 🔶 | your front end / downstream code |
| LLM-judge groundedness / toxicity (budgeted — adds latency and cost) | 🔶 | add as an extra check where risk justifies it |

## 4. Reliability, cost and abuse controls (OWASP LLM10)

| Rule | Status | Where |
|---|---|---|
| Bounded agent loop (max steps) + LangGraph `recursion_limit` | ✅ | `max_agent_steps`, `api.py` |
| Per-request token cap, per-user daily budget | ✅ | `tokens_used`, `TokenBudget` |
| Per-user rate limit | ✅ | `RateLimiter` |
| Circuit breaker around the model provider, generic errors to users | ✅ | `CircuitBreaker`, `AgentNodes.agent` |
| Redis-backed limits for multi-replica deployments | 🔶 | same interfaces |
| Provider failover / semantic cache | 🔶 | wrap `get_llm` |
| Kill switch (file or env) that stops model + tool activity without redeploy | ✅ | `controls.py`, `docs/RUNBOOK.md` |

## 5. Identity, tenancy and data

| Rule | Status | Where |
|---|---|---|
| Verify JWT: pinned algorithm, `exp/iat/aud/iss` required, short TTL | ✅ | `auth.py` |
| Tenant isolation: ACL filter **before** retrieval ranking; thread ids namespaced by tenant+user | ✅ | `rag/store.py`, `api.chat` |
| Corpus integrity (hash at ingest, verify at retrieval) | ✅ | `KnowledgeStore` |
| No secrets in prompts; secrets via env/secret manager | ✅ | `config.py` (`SecretStr`), `build_system_prompt` |
| Prod refuses to start with default secret, fake model, or in-memory state | ✅ | `Settings._prod_must_be_hardened` |
| Encrypt checkpoints at rest / DB TLS / data residency decisions | 🔶 | Postgres encryption; LangGraph encrypted serializer |
| Real SSO, MFA for admin interfaces | 🔶 | your IdP |

## 6. Audit, observability, evaluation

| Rule | Status | Where |
|---|---|---|
| Tamper-evident audit log: who, what, when, why, rule fired; hashes not raw prompts | ✅ | `audit.py` (hash chain + `verify`) |
| Every guardrail decision emitted as an event (track fire-rates over time) | ✅ | `input_check`, `output_check`, `tool_denied`, … |
| Structured JSON logs + request id on every response | ✅ | `logging_setup.py`, `api.py` |
| Ship audit + traces to SIEM/WORM; retention ≥ 12 months (SOC 2 / GDPR / EU AI Act Art. 12) | 🔶 | log pipeline |
| Alerting on anomalies (guardrail spikes, token z-score, cost per user) | 🔶 | metrics stack |
| Red-team regression suite in CI; re-run on every model/prompt/tool change | ✅ | `evals/redteam_cases.jsonl`, `tests/test_redteam.py` |
| Quality targets from the articles: jailbreak block >95%, PII recall >98%, hallucination <5% on RAG evals | 🔶 | grow the eval set; these need *your* data |
| Incident playbook | ✅ | `docs/RUNBOOK.md` |

## 7. Governance & compliance (things code cannot do)

- 🔶 Classify the use case under the **EU AI Act** (unacceptable / high / limited / minimal). High-risk systems need risk management, human oversight, logging and technical documentation (Annex IV).
- 🔶 Map each guardrail to a control id / **NIST AI RMF** function (Govern, Map, Measure, Manage) in a control register. This repo's table above is a starting point.
- 🔶 **GDPR**: lawful basis, DPIA for high-risk processing, right-to-erasure path for prompt logs, transparency notice that users talk to an AI.
- 🔶 AI Bill of Materials: model + dataset + tool-server inventory, provenance, version pinning (OWASP LLM03).
- 🔶 Named owners (security, legal/privacy, ML, business) and an AI governance review before launch.
- 🔶 SOC 2 / ISO 27001 evidence collection; DPA with your model provider; data residency check.

## OWASP Top 10 for LLM Apps (2025) coverage

| ID | Risk | Coverage in this repo |
|---|---|---|
| LLM01 | Prompt injection | input heuristics, untrusted-content handling, least privilege, HITL |
| LLM02 | Sensitive info disclosure | PII redaction in + out, tenant isolation, secrets blocked |
| LLM03 | Supply chain | 🔶 pin deps, scan (`pip-audit` in CI), keep an AiBOM |
| LLM04 | Data/model poisoning | corpus hash integrity; 🔶 ingestion review |
| LLM05 | Improper output handling | output guard, strip active content, strict tool schemas |
| LLM06 | Excessive agency | RBAC tools, risk tiers, approvals, limits |
| LLM07 | System prompt leakage | no secrets in prompt, canary detection |
| LLM08 | Vector/embedding weaknesses | filter-first tenant/role ACL, provenance, integrity |
| LLM09 | Misinformation | citation grounding, human approval for high stakes |
| LLM10 | Unbounded consumption | step/token caps, rate limit, budget, breaker, kill switch |

## Sources read while researching

- Gravitee, *OWASP Top 10 for LLM Applications 2025: a practical guide* — https://www.gravitee.io/blog/owasp-top-10-for-llm-applications-2025-a-practical-guide
- Arthur, *Best practices for building agent guardrails* — https://www.arthur.ai/blog/best-practices-for-building-agents-guardrails
- StackAI, *How to design AI agent guardrails: input validation, output filtering, safety controls* — https://www.stackai.com/insights/how-to-design-ai-agent-guardrails-best-practices-for-input-validation-output-filtering-and-safety-controls
- DEV Community, *Enterprise AI security: 12 best practices for deploying LLMs in production* — https://dev.to/jaipalsingh/enterprise-ai-security-12-best-practices-for-deploying-llms-in-production-525j
- FutureAGI, *AI compliance guardrails for enterprise LLMs* (GDPR, EU AI Act, NIST AI RMF, ISO 42001, SOC 2) — https://futureagi.com/blog/ai-compliance-guardrails-enterprise-llms-2025/
- DEV Community, *Human-in-the-loop checkpoints in LangGraph flows* — https://dev.to/aicomag/implementing-human-in-the-loop-checkpoints-in-langgraph-flows-fmp (high-level only)

Several are vendor blogs, so numeric targets (e.g. "95% jailbreak block rate") are their suggestions, not standards. The LangGraph-specific rules (interrupt re-execution, checkpointer requirement, bounded loops) come from my knowledge of the LangGraph API and are verified by the tests here; the official LangGraph/LangChain docs pages were not reachable from my environment, so re-check them against your pinned versions.
