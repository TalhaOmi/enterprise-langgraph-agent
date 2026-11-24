# Operations runbook

## Kill switch (stop everything now)
* `touch .kill_switch` in the service working directory (or set `AGENT_KILL_SWITCH=true` and restart).
* Effect: `/v1/chat` returns 503, `/readyz` fails, any in-flight graph run stops at its next agent step.
  `/healthz` stays green so the orchestrator does not restart-loop the pod.
* Resume: remove the file. Pending approvals stay valid until they expire.

## Incident: suspected prompt injection / data leak
1. Trip the kill switch if data may still be flowing.
2. Verify the audit log is intact: `python -c "from ent_agent.audit import AuditLog; print(AuditLog.verify('audit.log.jsonl'))"`.
3. Pull events by `request_id` / `user_id` / `tenant_id`: `input_check`, `llm_call`, `tool_denied`, `tool_executed`, `approval_*`, `output_check`.
4. Contain: revoke the user's sessions at the IdP; rotate `AGENT_JWT_SECRET` if tokens may be compromised (invalidates all tokens).
5. Add the attack as a line in `evals/redteam_cases.jsonl`, fix, and ship only when CI is green.
6. Notify per your breach/AI-incident policy (GDPR 72h clock if personal data was exposed).

## Routine
* **Daily:** watch guardrail fire-rates (`input_check` blocked %, `output_check` hard-fails, `tool_denied`). A sudden rise in PII findings or output failures is a signal to investigate before users report it.
* **Weekly:** review a sample of blocked and approved actions; check false-positive rate.
* **Per change** (model, prompt, tool, dependency): run `make test`, extend red-team cases, review the diff of `policy.py`.
* **Quarterly:** rotate secrets, run `pip-audit`, review roles/tools matrix with security, re-assess the EU AI Act risk class.

## Approvals
* Approvers call `GET /v1/approvals` then `POST /v1/approvals/{id}` with `{"approve": true|false, "reason": "..."}`.
* Expired approvals (`AGENT_APPROVAL_TIMEOUT_SECONDS`) are treated as rejected.
* An approver can never approve their own request (enforced in the API *and* inside the graph).

## Go-live gate (all must be true)
- [ ] `AGENT_ENV=prod` starts cleanly (secret, real model, Postgres checkpointer set)
- [ ] `make test` green, red-team suite extended with your domain's attacks
- [ ] Audit log shipped to SIEM/WORM with retention agreed with compliance
- [ ] Real IdP issuing JWTs (`iss`, `aud`, roles) and approver group staffed
- [ ] Rate-limit/budget state backed by Redis if more than one replica
- [ ] DPIA / EU AI Act classification / provider DPA signed off
- [ ] Kill switch tested in staging; on-call knows this page
