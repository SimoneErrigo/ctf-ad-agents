from __future__ import annotations

from langchain_core.tools import BaseTool

from src.tools.mcp_client import (
    MCPToolRegistry,
    build_patcher_read_registry,
    build_registry,
)


async def get_traffic_tools(
    registry: MCPToolRegistry | None = None,
    patcher_read_registry: MCPToolRegistry | None = None,
) -> list[BaseTool]:
    """Return the LangChain tools the traffic / defender agent is allowed to use.

    The agent consumes the `defender` MCP view, which is a superset of the
    `traffic` view: read-only packet tools (list_packets, get_packet, get_flow,
    validate_filter, get_filter_dsl, list_services, get_capture_status) PLUS
    rule CRUD (list_rules, create_rule, update_rule, delete_rule, list_alerts).

    It ALSO gets the patcher's read-only `list_vm_services` so service-inventory
    questions can report what actually runs on the VM (including services not
    behind Janus), not just the Janus-proxied set. Only the patcher has VM SSH,
    so that one tool is borrowed from the patcher read view; everything else
    stays Janus-scoped.

    HITL for `create_rule` / `update_rule` is applied by the agent's
    HumanInTheLoopMiddleware (`traffic_hitl`), not here, so the tools are
    returned raw.

    Pass a pre-built `registry` when the application already loaded it at
    startup (recommended, avoids re-fetching tools for each agent). When
    called with no argument it builds a fresh one on the spot, useful for
    quick scripts and unit tests.
    """
    reg = registry or await build_registry()
    raw = reg.for_agent("defender")
    # Borrow only the VM-inventory tool from the patcher read view (it owns VM
    # SSH); the rest of the traffic toolset stays Janus-scoped.
    preg = patcher_read_registry or await build_patcher_read_registry()
    raw = [*raw, *(t for t in preg.for_agent("patcher_read") if t.name == "list_vm_services")]
    # Make MCP / Janus failures recoverable: turn exceptions into a tool_result
    # string so the agent loop can adapt instead of crashing the run. Without
    # this, an MCP schema-validation failure or a Janus 4xx leaves the thread
    # state with an orphan tool_use (no matching tool_result) and any future
    # turn on the same THREAD_ID fails with a Bedrock contract error.
    for t in raw:
        t.handle_tool_error = True
    return raw
