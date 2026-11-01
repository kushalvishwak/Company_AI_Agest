"""
Layer 4: RBAC -- role definitions.

Role-based access control (RBAC) is the organizational half of the fix
for "the agent sees more than the user who's asking": each role gets
specific permissions, defined the same way they would be for a human
user of the same system.

Four roles, ranked least to most privileged. A tool or a retrieval
filter declares the minimum role it needs; `role_satisfies` checks a
caller's role against it. This is deliberately a simple ranked model
(not a full permission matrix) -- enough to demonstrate the pattern
without overthinking what a real deployment's role design should be.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR = "hr"
    ADMIN = "admin"


ROLE_RANK: dict[str, int] = {
    Role.EMPLOYEE: 0,
    Role.MANAGER: 1,
    Role.HR: 2,
    Role.ADMIN: 3,
}


def role_satisfies(caller_role: str, required_role: str | None) -> bool:
    """True if `caller_role` meets or exceeds `required_role`. A tool
    with no `required_role` is open to any authenticated caller."""
    if required_role is None:
        return True
    if caller_role not in ROLE_RANK or required_role not in ROLE_RANK:
        return False
    return ROLE_RANK[caller_role] >= ROLE_RANK[required_role]
