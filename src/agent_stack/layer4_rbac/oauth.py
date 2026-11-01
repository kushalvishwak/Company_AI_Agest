"""
Layer 4: RBAC -- the OAuth half.

The technical half of the RBAC fix is passing the actual requesting
user's identity on every tool call rather than relying on one blanket
service-account credential. For MCP-connected tools this is formalized
through OAuth: a protected MCP server is an OAuth 2.1 resource server
that accepts access tokens scoped to the calling user, and checks the
caller's own token and role -- not the agent's general-purpose
credentials -- before it does anything.

`TokenIssuer` here is a deliberately simplified, self-contained stand-in
for that pattern: short-lived, HMAC-signed, scoped tokens, issued per
user and verified on every call. It demonstrates the shape (identity +
role + scope + expiry, carried on every tool call) without pulling in a
full OAuth 2.1 authorization-server implementation, which would be out
of scope for this reference project. Swap in a real identity provider
(Okta, Auth0, Azure AD, etc) behind the same `AuthContext` shape for
production use.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass


class PermissionDenied(Exception):
    """Raised when a token is invalid, expired, or out of scope --
    mirrors the 403 a real OAuth-protected MCP server would return."""


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    role: str
    scopes: frozenset[str]
    issued_at: float
    expires_at: float

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at


class TokenIssuer:
    """Issues and verifies short-lived, scoped, per-user tokens.

    This is NOT a spec-complete OAuth 2.1 implementation -- it is the
    minimum needed to demonstrate "pass the user's identity on every
    tool call, short-lived and scoped, verified server-side" without
    guessing at a production identity-provider integration this project
    has no basis for in the source material it implements.
    """

    def __init__(self, signing_key: bytes | None = None) -> None:
        self._key = signing_key or os.environ.get(
            "RBAC_SIGNING_KEY", "dev-only-signing-key-do-not-use-in-prod"
        ).encode("utf-8")

    def issue_token(
        self,
        user_id: str,
        role: str,
        scopes: list[str] | frozenset[str],
        ttl_seconds: int = 300,
    ) -> str:
        """Short-lived on purpose: the default 5-minute TTL means a
        leaked token is only useful for a few minutes, and a caller has
        to keep re-authenticating rather than holding one long-lived
        credential."""
        now = time.time()
        payload = {
            "user_id": user_id,
            "role": role,
            "scopes": sorted(set(scopes)),
            "issued_at": now,
            "expires_at": now + ttl_seconds,
        }
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = hmac.new(self._key, body, hashlib.sha256).digest()
        token = base64.urlsafe_b64encode(body).decode() + "." + \
            base64.urlsafe_b64encode(signature).decode()
        return token

    def verify_token(self, token: str) -> AuthContext:
        try:
            body_b64, sig_b64 = token.split(".")
            body = base64.urlsafe_b64decode(body_b64.encode())
            signature = base64.urlsafe_b64decode(sig_b64.encode())
        except (ValueError, Exception) as exc:  # malformed token
            raise PermissionDenied(f"Malformed token: {exc}") from exc

        expected_signature = hmac.new(self._key, body, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected_signature):
            raise PermissionDenied("Token signature does not match -- rejecting.")

        payload = json.loads(body)
        ctx = AuthContext(
            user_id=payload["user_id"],
            role=payload["role"],
            scopes=frozenset(payload["scopes"]),
            issued_at=payload["issued_at"],
            expires_at=payload["expires_at"],
        )
        if ctx.is_expired():
            raise PermissionDenied(
                f"Token for {ctx.user_id} expired at {ctx.expires_at}."
            )
        return ctx
