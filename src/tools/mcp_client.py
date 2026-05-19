"""Async MCP client: one dedicated endpoint per agent.
To add a new agent:
    1. Tag the relevant tools on the MCP server (e.g. `tags={"defender"}`).
    2. Add a row in the server's AGENT_VIEWS so a /defender/mcp endpoint is
        mounted.
    3. Add an entry to AGENT_MCP_URLS below (env var or hard-coded default).
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)


def _default_url(agent: str) -> str:
    base = os.getenv("JANUS_MCP_BASE_URL", "http://localhost:8765").rstrip("/")
    return f"{base}/{agent}/mcp"


# agent name -> Streamable-HTTP URL of its dedicated MCP endpoint.
# Per-agent overrides via env vars: JANUS_TRAFFIC_MCP_URL, JANUS_DEFENDER_MCP_URL, ...
AGENT_MCP_URLS: dict[str, str] = {
    "traffic": os.getenv("JANUS_TRAFFIC_MCP_URL", _default_url("traffic")),
    # "defender": os.getenv("JANUS_DEFENDER_MCP_URL", _default_url("defender")),
}


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
