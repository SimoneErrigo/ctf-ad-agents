from __future__ import annotations

from langchain_core.tools import BaseTool
from src.tools.mcp_client import MCPToolRegistry, build_registry


async def get_traffic_tools(registry: MCPToolRegistry | None = None,) -> list[BaseTool]:
    """Return the LangChain tools the traffic agent is allowed to use.

    Pass a pre-built `registry` when the application already loaded it at
    startup (recommended, avoids re-fetching tools for each agent). When
    called with no argument it builds a fresh one on the spot, useful for
    quick scripts and unit tests.
    """
    reg = registry or await build_registry()
    return reg.for_agent("traffic")
