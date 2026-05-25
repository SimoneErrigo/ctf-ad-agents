"""Async MCP client: one dedicated endpoint per agent.
To add a new agent:
    1. Tag the relevant tools on the MCP server (e.g. `tags={"defender"}`).
    2. Add a row in the server's AGENT_VIEWS so a /defender/mcp endpoint is
        mounted.
    3. Add an entry to AGENT_MCP_URLS below.
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)

# Each server is configured by a single <SERVICE>_MCP_BASE_URL (host:port, no
# path); the per-endpoint path is appended below.
JANUS_MCP_BASE_URL: str = os.getenv("JANUS_MCP_BASE_URL", "http://localhost:8765").rstrip("/")
PATCHER_MCP_BASE_URL: str = os.getenv("PATCHER_MCP_BASE_URL", "http://localhost:8766").rstrip("/")

# agent name -> Streamable-HTTP URL of its dedicated Janus MCP endpoint.
# The traffic sub-agent needs BOTH read-only packet tools AND the rule CRUD
# surface, so it consumes the `defender` view, a superset of `traffic` via
# dual-tagging in janus-mcp/tools/traffic.py. The bare `traffic` view is left
# registered for future read-only consumers.
AGENT_MCP_URLS: dict[str, str] = {
    "traffic": f"{JANUS_MCP_BASE_URL}/traffic/mcp",
    "defender": f"{JANUS_MCP_BASE_URL}/defender/mcp",
}

# Patcher MCP is a separate FastMCP server (src/patcher/) with a single endpoint
# (no per-agent views); only the patch agent consumes it.
PATCHER_MCP_URL: str = f"{PATCHER_MCP_BASE_URL}/mcp"


class MCPToolRegistry:
    """Stores the LangChain tools loaded per agent."""

    def __init__(self, by_agent: dict[str, list[BaseTool]]) -> None:
        self._by_agent = by_agent

    def all(self) -> list[BaseTool]:
        seen: set[str] = set()
        out: list[BaseTool] = []
        for tools in self._by_agent.values():
            for t in tools:
                if t.name not in seen:
                    seen.add(t.name)
                    out.append(t)
        return out

    def for_agent(self, agent_name: str) -> list[BaseTool]:
        if agent_name not in self._by_agent:
            raise KeyError(
                f"Unknown agent '{agent_name}'. Register it in AGENT_MCP_URLS first."
            )
        tools = self._by_agent[agent_name]
        if not tools:
            log.warning(
                "Agent '%s' got 0 tools from its MCP endpoint, check that the "
                "server actually mounts /%s/mcp and that some tools carry the "
                "matching tag.",
                agent_name,
                agent_name,
            )
        return tools


async def build_registry() -> MCPToolRegistry:
    """Connect to every agent's MCP endpoint and load its tools."""
    by_agent: dict[str, list[BaseTool]] = {}
    for agent, url in AGENT_MCP_URLS.items():
        client = MultiServerMCPClient({
            agent: {"url": url, "transport": "streamable_http"},
        })
        tools = await client.get_tools()
        by_agent[agent] = tools
        log.info(
            "MCP[%s] loaded %d tools from %s: %s",
            agent,
            len(tools),
            url,
            [t.name for t in tools],
        )
    return MCPToolRegistry(by_agent)


async def build_patcher_registry() -> MCPToolRegistry:
    """Connect to the patcher MCP endpoint and load its tools.

    The patcher is a separate FastMCP server (src/patcher/) with a single
    endpoint and no per-agent views, so it gets its own one-entry registry
    instead of sharing AGENT_MCP_URLS with the janus-mcp views. Reusing
    MCPToolRegistry keeps the patch agent's loader structure identical to the
    janus agents (see traffic_agent_tools / patch_agent_tools).
    """
    client = MultiServerMCPClient({
        "patcher": {"url": PATCHER_MCP_URL, "transport": "streamable_http"},
    })
    tools = await client.get_tools()
    log.info(
        "MCP[patcher] loaded %d tools from %s: %s",
        len(tools),
        PATCHER_MCP_URL,
        [t.name for t in tools],
    )
    return MCPToolRegistry({"patcher": tools})
