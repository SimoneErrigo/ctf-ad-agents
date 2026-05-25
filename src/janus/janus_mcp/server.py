"""Janus MCP server with per-agent filtered views.

We expose one HTTP endpoint per agent (e.g. /traffic/mcp) by mounting an
independent FastMCP instance restricted to a specific tag set via
`view.enable(tags=..., only=True)`. Each instance registers the same tool
set; FastMCP then exposes only the tools whose tags intersect the allowed
set, and disables all the others.

That way each agent receives, over the wire, only the tools it is allowed
to use, with zero client-side filtering and zero dependence on `_meta`
propagation through the MCP/LangChain adapter stack.

Adding a new agent:
  1. tag the relevant tools server-side, e.g. `@mcp.tool(tags={"defender"})`,
  2. add `"defender": {"defender"}` to AGENT_VIEWS,
  3. point the defender sub-agent at `http://janus:8080/defender/mcp`.
"""

from __future__ import annotations

import contextlib
import logging
from typing import AsyncIterator

import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from janus_mcp.client import JanusClient
from janus_mcp.config import get_settings
from janus_mcp.tools import register_defender_tools, register_traffic_tools


# Agent name -> tag set the view exposes. Add a row per new agent.
#
# Read-only traffic tools are tagged with both "traffic" and "defender" inside
# tools/traffic.py — the defender agent needs to look at packets to write rules
# (validate_filter, list_packets, get_flow…), it does NOT need a private copy.
AGENT_VIEWS: dict[str, set[str]] = {
    "traffic": {"traffic"},
    "defender": {"defender"},
    # "exploit":  {"exploit",  "shared"},
}


INSTRUCTIONS_TRAFFIC = (
    "Tools to extract traffic captured by Janus (the team's reverse proxy / sniffer). "
    "Use `list_services` first to discover service IDs. Use `list_packets` with the "
    "`q` filter DSL for queries (see Janus FILTERS.md); call `get_packet` for full "
    "detail and `get_flow` to reconstruct a full request/response chain."
)

INSTRUCTIONS_DEFENDER = (
    "Tools to inspect Janus traffic AND to manage drop/alert rules. Read-only "
    "packet tools (list_services, list_packets, get_packet, get_flow, "
    "validate_filter, get_filter_dsl, get_capture_status) are available so you "
    "can confirm a candidate rule expression matches the right traffic with "
    "`list_packets(q=...)` before calling create_rule. Use action=\"alert\" for "
    "anything you are not 100% sure about; promotion to \"drop\" should go "
    "through the operator (human-in-the-loop)."
)

_AGENT_INSTRUCTIONS = {
    "traffic": INSTRUCTIONS_TRAFFIC,
    "defender": INSTRUCTIONS_DEFENDER,
}


def _build_view(agent: str, include_tags: set[str], client: JanusClient) -> FastMCP:
    """Build one FastMCP instance restricted to `include_tags`.

    FastMCP removed the `include_tags=` constructor kwarg; the supported way
    is to register every tool and then call `view.enable(tags=..., only=True)`,
    which keeps only the matching ones and disables all the rest.
    """
    view = FastMCP(
        name=f"janus-{agent}",
        instructions=_AGENT_INSTRUCTIONS.get(agent, INSTRUCTIONS_TRAFFIC),
    )
    register_traffic_tools(view, client)
    register_defender_tools(view, client)
    view.enable(tags=include_tags, only=True)
    return view


def build_app() -> tuple[Starlette, JanusClient]:
    """Build a Starlette app that mounts one MCP endpoint per agent.

    Result: each agent's endpoint lives at /<agent>/mcp and exposes only
    the tools tagged with one of the tags in AGENT_VIEWS[<agent>].
    """
    settings = get_settings()
    client = JanusClient(settings)

    routes: list = []
    sub_apps: list[Starlette] = []
    for agent, tags in AGENT_VIEWS.items():
        view = _build_view(agent, tags, client)
        # http_app(path="/mcp") returns a Starlette app rooted at "/mcp".
        # Mounting it under "/<agent>" exposes the endpoint as "/<agent>/mcp".
        sub_app = view.http_app(path="/mcp")
        sub_apps.append(sub_app)
        routes.append(Mount(f"/{agent}", app=sub_app))

    async def index(_request):
        return JSONResponse({
            "service": "janus-mcp",
            "agents": {
                name: {"endpoint": f"/{name}/mcp", "tags": sorted(tags)}
                for name, tags in AGENT_VIEWS.items()
            },
        })

    routes.append(Route("/", endpoint=index))

    # Each FastMCP http_app needs its own lifespan to be entered, otherwise
    # the streamable-http transport never initializes and tool calls fail.
    # We chain them all under the parent Starlette via AsyncExitStack.
    @contextlib.asynccontextmanager
    async def combined_lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for sub in sub_apps:
                await stack.enter_async_context(
                    sub.router.lifespan_context(sub)
                )
            yield

    app = Starlette(routes=routes, lifespan=combined_lifespan)
    return app, client


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    app, _client = build_app()
    logging.info(
        "Janus MCP listening on http://%s:%s — endpoints: %s",
        settings.janus_mcp_host,
        settings.janus_mcp_port,
        [f"/{a}/mcp" for a in AGENT_VIEWS],
    )
    uvicorn.run(app, host=settings.janus_mcp_host, port=settings.janus_mcp_port)


if __name__ == "__main__":
    main()
