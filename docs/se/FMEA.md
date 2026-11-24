# FMEA: failure modes of the support agent

Scores are **illustrative first-pass judgements by the author, not measured data**. Scale 1 (best) to 10 (worst); for Detection, 10 means hard to catch. RPN = Severity x Occurrence x Detection. Revisit after real traffic.

| # | Failure mode | Effect | Cause | S | O | D | RPN | Mitigation | Requirement |
|---|---|---|---|---|---|---|-----|-----------|-------------|
| F1 | Cross-tenant data leak | One customer sees another's data | Missing tenant filter | 10 | 2 | 5 | 100 | ACL filter before ranking; tenant-namespaced thread ids | REQ-010 |
| F2 | Prompt injection steers the agent | Unauthorised tool use or leak | Hostile user or document text | 8 | 6 | 4 | 192 | Input guard, untrusted-text sanitising, red-team set in CI | REQ-001 |
| F3 | Double refund on resume | Money paid twice | Node re-runs from first line after interrupt | 9 | 3 | 3 | 81 | Idempotency key per tool call; side effect after interrupt | REQ-004 |
| F4 | Model provider outage | Requests fail | Upstream incident | 5 | 5 | 2 | 50 | Circuit breaker, fallback message | REQ-020 |
| F5 | Runaway tool loop | Cost spike, no answer | Model keeps calling tools | 6 | 4 | 2 | 48 | Step cap and recursion limit | REQ-014 |
| F6 | Stale approval used | Action authorised long after the fact | Approval left pending | 7 | 3 | 4 | 84 | 300 s timeout treated as rejection | REQ-006 |
| F7 | Audit log edited after the fact | Lost accountability | Insider access | 8 | 2 | 6 | 96 | Hash chain verification | REQ-015 |
| F8 | Insecure config in prod | Forged tokens, lost state | Dev defaults deployed | 9 | 3 | 3 | 81 | Refuse to start in prod | REQ-019 |

Highest RPN first: F2, F1, F7, F6, F3/F8. Open gap: the Postgres restart test (REQ-021) is opt-in, so F3's durable path is only verified when someone runs it.
