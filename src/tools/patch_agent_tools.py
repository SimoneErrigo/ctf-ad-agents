from __future__ import annotations

from langchain_core.tools import BaseTool

from src.tools.hitl import apply_hitl
from src.tools.mcp_client import MCPToolRegistry, build_patcher_registry


async def get_patch_tools(registry: MCPToolRegistry | None = None) -> list[BaseTool]:
    """Return the LangChain tools the patch agent is allowed to use.

    The agent consumes the `patcher` MCP view, served by the dedicated patcher
    MCP server (src/patcher/) rather than janus-mcp. That server has a single
    endpoint and no per-agent views, so its tools live under one `patcher` key
    built by `build_patcher_registry()`.

    `deploy` / `rollback` are wrapped with HITL so any code push to the
    competition VM pauses for operator approval before it runs.

    Pass a pre-built `registry` when the application already loaded it at
    startup (recommended, avoids re-fetching tools for each agent). When
    called with no argument it builds a fresh one on the spot, useful for
    quick scripts and unit tests.
    """
    reg = registry or await build_patcher_registry()
    raw = reg.for_agent("patcher")
    # Make MCP / patcher failures recoverable: turn exceptions into a tool_result
    # string so the agent loop can adapt instead of crashing the run. Without
    # this, an MCP schema-validation failure or a patcher error leaves the thread
    # state with an orphan tool_use (no matching tool_result) and any future
    # turn on the same THREAD_ID fails with a Bedrock contract error.
    for t in raw:
        t.handle_tool_error = True
    return apply_hitl(raw)
