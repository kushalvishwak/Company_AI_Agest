from agent_stack.layer3_tools import MCPServer
from agent_stack.layer3_tools.tools import build_hr_tool_registry
from agent_stack.layer4_rbac import TokenIssuer


def _server():
    issuer = TokenIssuer(signing_key=b"test-key")
    registry = build_hr_tool_registry()
    return MCPServer(registry, issuer), issuer


def test_apply_leave_requires_confirmation_then_succeeds():
    server, issuer = _server()
    token = issuer.issue_token("u-1", "employee", scopes=["chat"])

    first = server.call("apply_leave", {"from": "2 Nov", "to": "4 Nov"}, token=token)
    assert first.status == "needs_confirmation"

    second = server.call(
        "apply_leave", {"from": "2 Nov", "to": "4 Nov"}, token=token, confirmed=True
    )
    assert second.status == "ok"
    assert second.data["status"] == "applied"
    assert second.data["id"].startswith("LV-")


def test_employee_cannot_read_salary():
    server, issuer = _server()
    employee_token = issuer.issue_token("u-1", "employee", scopes=["chat"])

    result = server.call(
        "get_salary", {"employee_id": "emp-003"}, token=employee_token
    )
    assert result.status == "denied"
    assert "403" in result.message


def test_hr_can_read_salary():
    server, issuer = _server()
    hr_token = issuer.issue_token("u-2", "hr", scopes=["chat"])

    result = server.call("get_salary", {"employee_id": "emp-003"}, token=hr_token)
    assert result.status == "ok"
    assert result.data["salary_inr"] == 9_000_000


def test_invalid_token_is_denied_before_role_check():
    server, _issuer = _server()
    result = server.call(
        "get_salary", {"employee_id": "emp-003"}, token="not-a-real-token"
    )
    assert result.status == "denied"


def test_unknown_tool_is_an_error():
    server, issuer = _server()
    token = issuer.issue_token("u-1", "employee", scopes=["chat"])
    result = server.call("delete_everything", {}, token=token)
    assert result.status == "error"
