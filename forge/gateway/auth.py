"""API-key authentication and role-based access control.

Roles are ordered: viewer < operator < admin. Inference needs `viewer`; mutating
platform state (deploy, rollback, approve) needs `operator`.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException

from forge.config import settings

ROLE_RANK = {"viewer": 0, "operator": 1, "admin": 2}


@dataclass(frozen=True)
class Principal:
    tenant: str
    role: str
    key_id: str

    def can(self, role: str) -> bool:
        return ROLE_RANK[self.role] >= ROLE_RANK[role]


def authenticate(authorization: str | None) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    key = authorization.split(" ", 1)[1].strip()
    found = settings().api_keys.get(key)
    if not found:
        raise HTTPException(401, "invalid API key", headers={"WWW-Authenticate": "Bearer"})
    tenant, role = found
    return Principal(tenant=tenant, role=role, key_id=key[:6] + "…")


def require(role: str):
    def dep(authorization: str | None = Header(default=None)) -> Principal:
        p = authenticate(authorization)
        if not p.can(role):
            raise HTTPException(403, f"role '{p.role}' cannot perform an action requiring '{role}'")
        return p

    return dep
