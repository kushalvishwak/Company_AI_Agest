"""
Layer 3: Tools + MCP -- the registry half.

Employees don't just want answers; they want things done. Tool calling
closes that gap: the application describes a set of tools to the model,
the model decides when to call one and with what arguments, and the
surrounding code (here, `MCPServer`, in the same package) actually
executes it.

Common mistake this module warns about: handing a single assistant too
many tools at once makes the model more likely to pick the wrong one.
`ToolRegistry` logs a warning past a configurable threshold instead of
silently letting a toolset sprawl.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

# Past this many tools on one registry, a single assistant is more likely
# to pick the wrong tool or call it with the wrong arguments. Not a hard
# limit -- a warning that the toolset is growing past "focused".
TOO_MANY_TOOLS_THRESHOLD = 8


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON-schema-shaped parameter spec
    handler: Callable[[dict], dict]
    required_role: str | None = None
    requires_confirmation: bool = False


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        if len(self.tools) > TOO_MANY_TOOLS_THRESHOLD:
            logger.warning(
                "This registry now exposes %d tools to one assistant "
                "(threshold: %d). A focused toolset per assistant beats a "
                "kitchen-sink approach -- consider splitting this into "
                "multiple scoped assistants or MCP servers.",
                len(self.tools),
                TOO_MANY_TOOLS_THRESHOLD,
            )

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def as_llm_tools(self) -> list[dict]:
        """The shape passed to `LLMClient.complete(tools=...)`."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self.tools.values()
        ]
