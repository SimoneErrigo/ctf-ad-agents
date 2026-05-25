from __future__ import annotations

from langchain_core.tools import BaseTool

from src.tools.hitl import apply_hitl
from src.tools.mcp_client import MCPToolRegistry, build_registry


async def get_traffic_tools(registry: MCPToolRegistry | None = None) -> list[BaseTool]:
    """Return the LangChain tools the traffic / defender agent is allowed to use.

    The agent consumes the `defender` MCP view, which is a superset of the
    `traffic` view: read-only packet tools (list_packets, get_packet, get_flow,
    validate_filter, get_filter_dsl, list_services, get_capture_status) PLUS
    rule CRUD (list_rules, create_rule, update_rule, delete_rule, list_alerts).

    `create_rule` / `update_rule` are wrapped with HITL so that any
    action that affects live traffic (action="drop" / "both") pauses for
    operator approval before reaching Janus.

    Pass a pre-built `registry` when the application already loaded it at
    startup (recommended, avoids re-fetching tools for each agent). When
    called with no argument it builds a fresh one on the spot, useful for
    quick scripts and unit tests.
    """
    reg = registry or await build_registry()
    raw = reg.for_agent("defender")
    # Make MCP / Janus failures recoverable: turn exceptions into a tool_result
    # string so the agent loop can adapt instead of crashing the run. Without
    # this, an MCP schema-validation failure or a Janus 4xx leaves the thread
    # state with an orphan tool_use (no matching tool_result) and any future
    # turn on the same THREAD_ID fails with a Bedrock contract error.
    for t in raw:
        t.handle_tool_error = True
    return apply_hitl(raw)
