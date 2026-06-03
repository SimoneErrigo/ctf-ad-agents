"""Async MCP client: one dedicated endpoint per agent.
To add a new agent:
    1. Tag the relevant tools on the MCP server (e.g. `tags={"defender"}`).
    2. Add a row in the server's AGENT_VIEWS so a /defender/mcp endpoint is
        mounted.
    3. Add an entry to AGENT_MCP_URLS below.
"""

from __future__ import annotations

import functools
import logging
import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)


def _collapse_text_blocks(content: object) -> object:
    """Join a list of MCP text content blocks into a single string.

    `langchain_mcp_adapters` returns ToolMessage content as a list of content
    blocks (e.g. ``[{"type": "text", "text": ...}]``). The Agent Chat UI renders
    non-string content as ``[object Object]``, so for the common text-only case we
    flatten it to a plain string. Any non-text block (image/file) means we keep
    the original list so those still render through the UI's block handling.
    """
    if not isinstance(content, list):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            return content
    return "\n".join(parts)


def _stringify_tool_content(tool: BaseTool) -> BaseTool:
    """Wrap a content_and_artifact MCP tool so its ToolMessage content is a plain
    string for the UI, leaving the structured `artifact` untouched.

    Only the async ``coroutine`` is wrapped (MCP tools are async-only). We use
    ``functools.wraps`` so ``inspect.signature`` still resolves to the original;
    StructuredTool introspects it for ``callbacks``/config params at call time.
    """
    original = tool.coroutine
    if original is None:
        return tool

    @functools.wraps(original)
    async def wrapped(*args: object, **kwargs: object) -> object:
        result = await original(*args, **kwargs)
        # content_and_artifact tools return (content, artifact); a handoff tool
        # may return a Command/ToolMessage directly, leave those untouched.
        if isinstance(result, tuple) and len(result) == 2:
            content, artifact = result
            return _collapse_text_blocks(content), artifact
        return result

    tool.coroutine = wrapped
    return tool


def _normalize_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Apply the UI-friendly content normalization to every loaded MCP tool."""
    return [_stringify_tool_content(t) for t in tools]

# Each server is configured by a single <SERVICE>_MCP_BASE_URL (host:port, no
# path); the per-endpoint path is appended below.
JANUS_MCP_BASE_URL: str = os.getenv("JANUS_MCP_BASE_URL", "http://localhost:8765").rstrip("/")
PATCHER_MCP_BASE_URL: str = os.getenv("PATCHER_MCP_BASE_URL", "http://localhost:8766").rstrip("/")
EXPLOITER_MCP_BASE_URL: str = os.getenv("EXPLOITER_MCP_BASE_URL", "http://localhost:8767").rstrip("/")

# agent name -> Streamable-HTTP URL of its dedicated Janus MCP endpoint.
# The traffic sub-agent needs BOTH read-only packet tools AND the rule CRUD
# surface, so it consumes the `defender` view, a superset of `traffic` via
# dual-tagging in janus-mcp/tools/traffic.py. The `exploit` view is read-only
# packet tools (no rule CRUD), consumed by the exploit agent.
AGENT_MCP_URLS: dict[str, str] = {
    "traffic": f"{JANUS_MCP_BASE_URL}/traffic/mcp",
    "defender": f"{JANUS_MCP_BASE_URL}/defender/mcp",
    "exploit": f"{JANUS_MCP_BASE_URL}/exploit/mcp",
}

# Patcher MCP is a separate FastMCP server (src/patcher/). The full `patch` view
# is at /mcp (patch agent); a read-only `read` view is at /read/mcp (exploit
# agent, to read service source without write/deploy).
PATCHER_MCP_URL: str = f"{PATCHER_MCP_BASE_URL}/mcp"
PATCHER_MCP_READ_URL: str = f"{PATCHER_MCP_BASE_URL}/read/mcp"

# Exploiter MCP is a separate FastMCP server (src/exploiter/) with a single
# endpoint; only the exploit agent consumes it.
EXPLOITER_MCP_URL: str = f"{EXPLOITER_MCP_BASE_URL}/mcp"


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
        tools = _normalize_tools(await client.get_tools())
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
    tools = _normalize_tools(await client.get_tools())
    log.info(
        "MCP[patcher] loaded %d tools from %s: %s",
        len(tools),
        PATCHER_MCP_URL,
        [t.name for t in tools],
    )
    return MCPToolRegistry({"patcher": tools})


async def _load_single(agent: str, url: str) -> MCPToolRegistry:
    """Load one MCP endpoint into a one-entry registry keyed by `agent`."""
    client = MultiServerMCPClient({agent: {"url": url, "transport": "streamable_http"}})
    tools = _normalize_tools(await client.get_tools())
    log.info("MCP[%s] loaded %d tools from %s: %s", agent, len(tools), url, [t.name for t in tools])
    return MCPToolRegistry({agent: tools})


async def build_exploiter_registry() -> MCPToolRegistry:
    """Load the exploiter MCP endpoint (xfarm authoring + farm recon)."""
    return await _load_single("exploiter", EXPLOITER_MCP_URL)


async def build_patcher_read_registry() -> MCPToolRegistry:
    """Load the patcher read-only view (source reading, no write/deploy)."""
    return await _load_single("patcher_read", PATCHER_MCP_READ_URL)
