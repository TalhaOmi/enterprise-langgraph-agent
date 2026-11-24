"""Identity. The agent NEVER takes identity from the model or from request bodies.

Identity comes only from a verified, short-lived JWT issued by your IdP (Okta, Entra ID,
Auth0...). `issue_token` exists for dev/tests so the repo is runnable offline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jwt

from .config import Settings


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    roles: frozenset[str]

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "tenant_id": self.tenant_id, "roles": sorted(self.roles)}

    @classmethod
    def from_dict(cls, d: dict) -> Principal:
        return cls(d["user_id"], d["tenant_id"], frozenset(d["roles"]))


def issue_token(
    settings: Settings,
    user_id: str,
    tenant_id: str,
    roles: list[str],
    ttl: int | None = None,
) -> str:
    now = int(time.time())
    claims = {
        "sub": user_id,
        "tenant": tenant_id,
        "roles": roles,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + (ttl if ttl is not None else settings.jwt_ttl_seconds),
    }
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm)


def verify_token(settings: Settings, token: str) -> Principal:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],  # pin the algorithm: never trust the header
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
        user, tenant, roles = claims["sub"], claims["tenant"], claims["roles"]
        if not (isinstance(user, str) and isinstance(tenant, str) and isinstance(roles, list)):
            raise AuthError("malformed claims")
        return Principal(user, tenant, frozenset(str(r) for r in roles))
    except (jwt.PyJWTError, KeyError) as exc:
        raise AuthError("invalid token") from exc
