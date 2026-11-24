"""Print dev tokens for trying the chat UI. Refuses to run against a prod config. Usage: python scripts/make_token.py"""

from ent_agent.auth import issue_token
from ent_agent.config import Settings

s = Settings()
if s.env == "prod":
    raise SystemExit("refusing to mint tokens in prod: tokens come from your identity provider")
for user, roles in [("alice", ["support_agent"]), ("bob", ["approver"])]:
    print(f"\n{user} ({', '.join(roles)}):\n{issue_token(s, user, 'acme', roles)}")
