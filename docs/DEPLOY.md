# Deploying to a real host

Status: the **smoke test** and the **chat UI** were tested against a live local server. The **Dockerfile, compose file and the two host configs in `deploy/` were NOT run or validated** by the author (no Docker daemon, no access to Fly or Render). Treat them as drafts and validate each step.

## Known limits to fix before real customers (read this first)
| Limit | Why it matters | What to do |
|-------|----------------|------------|
| Approval registry is in memory (`ApprovalRegistry` in `api.py`) | A restart forgets pending approval ids; a second instance cannot see them | Run exactly **one** instance for now. Later: store approvals in a Postgres table |
| Rate limiter and token budget are in memory | Each instance counts separately; a restart resets them | Same: one instance, or move counters to Redis/Postgres |
| Audit log is a local file | A host with an ephemeral disk loses it | Mount a volume (`/data`) and ship the log to your log platform |
| The first request after boot may pay model start-up costs | Cold starts | Keep one instance always running |

## Steps (host-neutral)
1. **Pick a host.** Fly.io or Render are simplest for one container plus Postgres. AWS (App Runner, ECS, or EC2 you already know) also works.
2. **Postgres.** Create a managed Postgres. Put its URL in `AGENT_CHECKPOINT_URL`. Prod refuses to start without it.
3. **Secrets** (never in git): `AGENT_JWT_SECRET` (`openssl rand -hex 32`), your model key (`ANTHROPIC_API_KEY`), `AGENT_CHECKPOINT_URL`.
4. **Config:** `AGENT_ENV=prod`, `AGENT_LLM_MODEL=anthropic:<model>`. A fake model in prod is refused on purpose.
5. **Deploy** using `deploy/fly.toml` or `deploy/render.yaml` as a starting point (validate them first).
6. **Health checks:** point the host at `/healthz`. Use `/readyz` for load-balancer readiness (it returns 503 when the kill switch is on).
7. **Smoke test:** `set SMOKE_USER_TOKEN=...` and `set SMOKE_APPROVER_TOKEN=...` then `python scripts/smoke_test.py https://your-app`. All 9 checks must pass.
8. **Tokens:** in real use they come from your identity provider. For a **demo deployment** only, create a separate deployment with its own secret and mint tokens with `scripts/make_token.py` (it refuses prod config, so run it with a dev config but the demo secret).
9. **Rollback:** keep the previous image tag. To stop the service fast without a deploy: create the kill-switch file or set `AGENT_KILL_SWITCH=true` and restart.
10. **Cost safety:** set a spending limit with your model provider, and keep `AGENT_DAILY_TOKEN_BUDGET_PER_USER` and the rate limit.

## Before telling anyone "it is deployed"
- All 9 smoke checks pass against the public URL.
- You tried the chat UI in a browser with two different users.
- You looked at the logs and the audit file once.
- You know how to switch it off.
