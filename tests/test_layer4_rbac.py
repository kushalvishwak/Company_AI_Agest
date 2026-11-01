import time

import pytest

from agent_stack.layer4_rbac import PermissionDenied, TokenIssuer, role_satisfies


def test_role_hierarchy():
    assert role_satisfies("admin", "employee")
    assert role_satisfies("hr", "employee")
    assert not role_satisfies("employee", "hr")
    assert role_satisfies("employee", None)  # no requirement = open


def test_token_round_trip():
    issuer = TokenIssuer(signing_key=b"test-key")
    token = issuer.issue_token("u-1", "employee", scopes=["chat"])
    ctx = issuer.verify_token(token)
    assert ctx.user_id == "u-1"
    assert ctx.role == "employee"
    assert ctx.has_scope("chat")


def test_expired_token_is_denied():
    issuer = TokenIssuer(signing_key=b"test-key")
    token = issuer.issue_token("u-1", "employee", scopes=["chat"], ttl_seconds=0)
    time.sleep(0.01)
    with pytest.raises(PermissionDenied):
        issuer.verify_token(token)


def test_tampered_token_is_denied():
    issuer = TokenIssuer(signing_key=b"test-key")
    token = issuer.issue_token("u-1", "employee", scopes=["chat"])
    body_b64, sig_b64 = token.split(".")
    tampered = body_b64 + "A." + sig_b64  # corrupt the payload half
    with pytest.raises(PermissionDenied):
        issuer.verify_token(tampered)


def test_token_signed_with_different_key_is_rejected():
    issuer_a = TokenIssuer(signing_key=b"key-a")
    issuer_b = TokenIssuer(signing_key=b"key-b")
    token = issuer_a.issue_token("u-1", "admin", scopes=["chat"])
    with pytest.raises(PermissionDenied):
        issuer_b.verify_token(token)
