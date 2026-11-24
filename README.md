# Enterprise LangGraph Agent

A production-shaped **LangChain + LangGraph** agent with layered guardrails, human approval for
risky actions, tenant isolation, a tamper-evident audit log, an authenticated API, tests, Docker
and CI. It runs **fully offline** (deterministic fake model) so you can study and test every
guardrail with no API key, then switch to a real model with one env var.

The demo domain is a support copilot (policy Q&A with citations, order lookup, tickets, refunds).
Swap the tools/services for your own; the safety architecture stays.

```
Client ─JWT─▶ API (authN · kill switch · rate limit · budget)
              └▶ LangGraph: input_guard ▶ agent ⇄ authorize_tools ▶ [approval_gate] ▶ run_tools
                                              └▶ output_guard ▶ END
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make test                    # 97 tests incl. red-team suite, no network needed (+1 opt-in Postgres test)
python scripts/demo.py       # guided walkthrough: grounded answer, injection refused, approval flow
make run                     # API on http://localhost:8000  (docs at /docs in dev)
```

Real model: `pip install -e ".[anthropic]"` (or `.[openai]`), then set
`AGENT_LLM_MODEL=anthropic:<model-name>` and the provider's API key env var.

## What is in the box

| Concern | Implementation |
|---|---|
| Input guardrails | normalisation, injection heuristics, length cap, PII/secret redaction |
| Output guardrails | canary leak detection, secret block, PII redaction, markup stripping, citation grounding, bounded retry → safe fallback |
| Agency control | role→tool allowlists, risk tiers, strict arg schemas, policy-as-code, **human approval** with separation of duties |
| RAG safety | tenant+role filter before ranking, provenance, integrity hashes, untrusted-content sanitising |
| Cost/abuse | step cap, token cap, rate limit, daily budget, circuit breaker, kill switch |
| Identity | short-lived JWT (pinned alg, aud/iss/exp required); identity never comes from the model |
| Audit | hash-chained JSONL (`AuditLog.verify`), fingerprints instead of raw prompts |
| State | LangGraph checkpointer (in-memory dev, Postgres prod) so approvals survive restarts |
| Quality | pytest suite + `evals/redteam_cases.jsonl` regression set, ruff, CI with `pip-audit` |
| Governance | `docs/ENTERPRISE_CHECKLIST.md` (rules from the research, mapped to code), runbook |

Read **`docs/ENTERPRISE_CHECKLIST.md`** first: it lists every rule, where it lives in the code, and
what *you* still have to provide (SSO, SIEM, Redis, DPIA, EU AI Act classification…).

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` `{message, thread_id?}` | run the agent. Returns `completed`, `blocked` or `pending_approval` (+ `approval_id`, `actions`) |
| `GET /v1/approvals` | approver: list pending approvals in own tenant |
| `POST /v1/approvals/{id}` `{approve, reason}` | approver decides; resumes the paused graph |
| `GET /healthz`, `GET /readyz` | liveness / readiness (fails when kill switch is on) |

Dev tokens: `python -c "from ent_agent.auth import issue_token; from ent_agent.config import Settings; print(issue_token(Settings(), 'alice', 'acme', ['support_agent']))"`

## Layout

```
src/ent_agent/
  api.py            FastAPI perimeter + approval endpoints
  graph/            state, nodes (one control point each), builder, prompts
  guardrails/       pii, input_guard, output_guard, policy, limits, net (SSRF)
  tools/registry.py tool specs: risk tier + strict schema + handler
  rag/store.py      tenant-isolated, integrity-checked knowledge store
  services.py       order/ticket back-ends (in-memory stand-ins, idempotent)
  audit.py auth.py config.py controls.py llm.py testing.py
tests/              unit + graph + API + red-team   evals/   attack dataset
docs/               checklist · architecture · runbook
```

## Deploy

`docker compose up --build` starts the API + Postgres. For production set (see `.env.example`):
`AGENT_ENV=prod`, a random `AGENT_JWT_SECRET`, a real `AGENT_LLM_MODEL`, and `AGENT_CHECKPOINT_URL`.
The app **refuses to boot in prod** with any insecure default. Install `.[postgres]` for the durable
checkpointer. It is covered by an opt-in durability test (`make test-postgres`) that proves a pending approval
survives a process restart and the refund still runs exactly once.

## Honest limits

* Regex injection/PII detection is a baseline. Add an ML detector and an LLM-judge for groundedness
  where risk justifies the latency and cost.
* Rate-limit, budget and approval registries are in-memory; use Redis/Postgres for >1 replica.
* The tests prove the *architecture* behaves with a scripted model; run the red-team set against your
  real model and prompts before launch and keep growing it.
* This is engineering scaffolding, not legal advice: compliance items in section 7 of the checklist need your counsel.

## Learning notes (concepts worth knowing)

* **StateGraph**: nodes read/write a shared typed state; conditional edges are the control flow.
* **Checkpointer + `thread_id`**: saves state after every node; required for `interrupt()`.
* **`interrupt()` / `Command(resume=…)`**: pause for a human, resume later; the node re-runs from its top.
* **Reducers** (`add_messages`): merge updates; same message id = replace (used to swap in redacted text).
* **Guardrail placement**: deterministic and cheap *before* the model, independent checks *after*, policy *between* model and tools.

## Systems engineering artefacts

`docs/se/` holds the requirements (`requirements.json`, EARS style, with IDs), the generated traceability matrix, an FMEA, an interface control document and change-control rules. `python scripts/trace_check.py` lints requirement wording, confirms every linked test exists, and (with `--junit`) reports each requirement as verified or not. CI runs it.

## Course

`course/` holds the tested step-by-step solution code for the interactive LangGraph course (modules 4 to 22: RAG, advanced RAG, observability, multi-agent and memory, evaluation).
