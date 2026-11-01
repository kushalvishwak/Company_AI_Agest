"""
Example HR tools behind the mock MCP server -- the two scenarios the
architecture write-up this project implements uses throughout:

  - `apply_leave`: "2-4 Nov leave laga do" -> apply_leave(from, to) ->
    {"status": "applied", "id": "LV-..."}. A write action, so it is
    registered with `requires_confirmation=True`.

  - `get_salary`: the RBAC cautionary tale -- an employee asking for
    someone else's salary must get a 403, not an answer, because the
    tool is registered with `required_role="hr"` and the MCP server
    checks the CALLER's role on every call (see mcp_server.py).

All data here is an in-memory mock. No real HR system is involved.
"""

from __future__ import annotations

import itertools

from .. import Tool, ToolRegistry

_leave_id_counter = itertools.count(1)

# In-memory "database" of applied leave requests, newest last.
LEAVE_REQUESTS: list[dict] = []

# In-memory mock employee directory. Illustrative only, as the source
# material this project implements explicitly says its own examples are.
EMPLOYEE_DIRECTORY: dict[str, dict] = {
    "emp-001": {"name": "Asha Verma", "role": "employee", "salary_inr": 900_000},
    "emp-002": {"name": "Rahul Nair", "role": "manager", "salary_inr": 1_800_000},
    "emp-003": {"name": "CEO (mock)", "role": "admin", "salary_inr": 9_000_000},
}


def _apply_leave(arguments: dict) -> dict:
    date_from = arguments.get("from")
    date_to = arguments.get("to")
    if not date_from or not date_to:
        raise ValueError("apply_leave requires both 'from' and 'to'.")
    leave_id = f"LV-{next(_leave_id_counter):03d}"
    record = {"id": leave_id, "from": date_from, "to": date_to, "status": "applied"}
    LEAVE_REQUESTS.append(record)
    return {"status": "applied", "id": leave_id}


def _get_salary(arguments: dict) -> dict:
    employee_id = arguments.get("employee_id")
    employee = EMPLOYEE_DIRECTORY.get(employee_id)
    if employee is None:
        raise ValueError(f"No such employee: {employee_id!r}")
    return {"employee_id": employee_id, "salary_inr": employee["salary_inr"]}


def build_hr_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="apply_leave",
            description="File a leave request for the given date range.",
            parameters={
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "Start date"},
                    "to": {"type": "string", "description": "End date"},
                },
                "required": ["from", "to"],
            },
            handler=_apply_leave,
            required_role="employee",
            requires_confirmation=True,
        )
    )

    registry.register(
        Tool(
            name="get_salary",
            description="Look up an employee's salary. HR-only.",
            parameters={
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                },
                "required": ["employee_id"],
            },
            handler=_get_salary,
            required_role="hr",
            requires_confirmation=False,
        )
    )

    return registry
