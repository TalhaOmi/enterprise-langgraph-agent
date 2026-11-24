# Interface control document

Each interface is written down, owned, and changed only through `CHANGE_CONTROL.md`. Owner is a role, not a person.

| ID | Interface | Provider -> Consumer | Contract | Owner role |
|----|-----------|----------------------|----------|-----------|
| IF-01 | HTTP API | Client -> `api.py` | `POST /v1/chat`, `POST /v1/approvals/{id}`; JWT bearer; message max 8000 chars; responses carry security headers | API owner |
| IF-02 | Graph state | `api.py` -> LangGraph | `messages`, `principal`, `request_id`; thread id `tenant:user:thread` | Agent owner |
| IF-03 | Tool registry | Graph -> `tools/registry.py` | Each tool has a pydantic args model (`extra=forbid`) and a risk tier READ/WRITE/HIGH | Policy owner |
| IF-04 | Knowledge store | Graph -> `rag/store.py` | Query returns only docs matching tenant and role, each with a hash and id for `[doc:id]` citations | Data owner |
| IF-05 | Model provider | `llm.py` -> provider | `get_llm()` returns a chat model; timeout 30 s; behind circuit breaker | Platform owner |
| IF-06 | Audit sink | Graph and API -> `audit.py` | Append-only JSONL, hash-chained, fingerprints not raw prompts | Security owner |
| IF-07 | Checkpointer | Graph -> Postgres | `PostgresSaver` in prod, in-memory only in dev/test | Platform owner |
