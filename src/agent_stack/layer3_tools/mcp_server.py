"""
Layer 3: Tools + MCP -- the server half.

The Model Context Protocol (MCP) standardizes the wiring between an
agent and the backend systems it touches: rather than a bespoke
integration per system, each backend (HR, Jira, Slack, ...) sits behind
an MCP server, and the agent is the client speaking one common protocol
to all of them.

`MCPServer` here is a small, in-process stand-in for that pattern --
enough to demonstrate tool dispatch, RBAC enforcement (Layer 4) on every
call, and a confirmation gate on write actions, without pulling in the
full MCP SDK and a real transport, which this reference project has no
need for. Swap this class for a real `mcp` server/client pair behind the
same `call()` signature for production use.

Two mistakes this module guards against directly:
  - Write actions going live with no confirmation step.
  - Tools running under one powerful shared credential instead of the
    calling user's own permissions (enforced here via `TokenIssuer` from
    layer4_rbac on every single call, not once at connection time).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..layer4_rbac.oauth import AuthContext, PermissionDenied, TokenIssuer
from ..layer4_rbac.roles import role_satisfies
from .tool_registry import ToolRegistry


@dataclass
class ToolResult:
    status: str  # "ok" | "denied" | "needs_confirmation" | "error"
    data: dict | None = None
    message: str | None = None


class MCPServer:
    def __init__(self, registry: ToolRegistry, token_issuer: TokenIssuer) -> None:
        self.registry = registry
        self.token_issuer = token_issuer

    def call(
        self,
        tool_name: str,
        arguments: dict,
        *,
        token: str,
        confirmed: bool = False,
    ) -> ToolResult:
        # 1. Verify the CALLER's own token on every single call -- never
        #    a blanket agent credential. This is the check that stops an
        #    intern's agent session from reading the CEO's salary just
        #    because the agent process itself has broad access.
        try:
            auth: AuthContext = self.token_issuer.verify_token(token)
        except PermissionDenied as exc:
            return ToolResult(status="denied", message=str(exc))

        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(status="error", message=f"Unknown tool: {tool_name}")

        # 2. Role check, mirroring a real OAuth-protected MCP server's
        #    403 for a caller whose role doesn't meet the tool's bar.
        if not role_satisfies(auth.role, tool.required_role):
            return ToolResult(
                status="denied",
                message=(
                    f"403: not allowed for this role "
                    f"(caller role={auth.role!r}, needs={tool.required_role!r})"
                ),
            )

        # 3. Confirmation gate on write actions: pay / delete / send
        #    tools never execute on the first call.
        if tool.requires_confirmation and not confirmed:
            return ToolResult(
                status="needs_confirmation",
                message=(
                    f"'{tool_name}' changes state and needs explicit "
                    f"confirmation (confirmed=True) before it runs."
                ),
            )

        try:
            result = tool.handler(arguments)
        except Exception as exc:  # the handler is untrusted-ish third-party code
            return ToolResult(status="error", message=str(exc))

        return ToolResult(status="ok", data=result)
