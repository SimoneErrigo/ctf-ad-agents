from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from janus_mcp.client import JanusClient
from janus_mcp.config import get_settings

log = logging.getLogger(__name__)

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
_FILTERS_MD_PATH = _RESOURCES_DIR / "FILTERS.md"
_CHEATSHEET_MD_PATH = _RESOURCES_DIR / "CHEATSHEET.md"


def _read_resource(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("Resource %s not bundled; using inline fallback.", path)
        return fallback


# Short fallback only used if CHEATSHEET.md is missing from the wheel, the
# canonical content lives in resources/CHEATSHEET.md.
_CHEATSHEET_FALLBACK = (
    "Janus filter DSL — short cheatsheet (full reference: resource "
    "`docs://janus/filters`). See resources/CHEATSHEET.md."
)
_FILTERS_FALLBACK = _CHEATSHEET_FALLBACK

FILTERS_FULL = _read_resource(_FILTERS_MD_PATH, _FILTERS_FALLBACK)
FILTER_CHEATSHEET = _read_resource(_CHEATSHEET_MD_PATH, _CHEATSHEET_FALLBACK)


# Janus packet schema (backend/internal/sniffer/models.go):
#   body []byte -> base64 string in JSON (raw bytes)
#   body_string string -> UTF-8 decoded body (if printable)
#   headers map[string]string -> HTTP headers
# There is NO `raw` / `raw_string` / `request_headers` / `response_headers`.

# Noise headers stripped before the packet reaches the LLM. We deliberately
# keep Authorization, Cookie, User-Agent, X-Forwarded-For, Host, Referer:
# the traffic agent uses those to distinguish attacker sessions.
_NOISE_HEADERS = frozenset({
    "accept",
    "accept-encoding",
    "accept-language",
    "connection",
    "content-type",
    "content-length",
    "etag",
    "x-powered-by",
    "date",
    "access-control-allow-origin",
    "cache-control",
    "pragma",
    "expires",
    "server",
    "vary",
    "transfer-encoding",
})


def _slim_headers(packet: dict[str, Any]) -> None:
    """In-place: drop noisy HTTP headers from packet['headers']."""
    headers = packet.get("headers")
    if isinstance(headers, dict):
        packet["headers"] = {
            k: v for k, v in headers.items() if k.lower() not in _NOISE_HEADERS
        }


def _trim_body(packet: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """Strip raw bytes (`body`) and trim `body_string` to max_bytes.

    `body` is base64 of the raw bytes, heavy and rarely useful to the LLM,
    we drop it entirely. `body_string` is the readable form, keep it but
    truncate it for huge payloads so the agent context stays small.
    """
    packet.pop("body", None)
    if max_bytes <= 0:
        return packet
    value = packet.get("body_string")
    if isinstance(value, str) and len(value) > max_bytes:
        packet["body_string"] = value[:max_bytes]
        packet["body_string_truncated"] = True
        packet["body_string_original_length"] = len(value)
    return packet


# Fields stripped in summary mode. Janus' own `summary=true` already returns
# Lite packets (no body/headers/matched_flagids) so this is a defensive net
# in case the backend changes; keep ids, timing, routing, status, method, url.
_HEAVY_FIELDS = frozenset({
    "body",
    "body_string",
    "headers",
    "body_string_truncated",
    "body_string_original_length",
    "matched_flagids",
    # matched_rules per row can be huge; summaries use get_packet for detail.
    "matched_rules",
})


def _summarize_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in packet.items() if k not in _HEAVY_FIELDS}


def register_traffic_tools(mcp: FastMCP, client: JanusClient) -> None:
    s = get_settings()

    @mcp.resource(
        uri="docs://janus/filters",
        name="Janus filter DSL (full reference)",
        description="Full FILTERS.md, fields, operators, examples, drop/alert rule snippets.",
        mime_type="text/markdown",
    )
    def filter_dsl_resource() -> str:
        return FILTERS_FULL

    @mcp.tool(
        tags={"traffic"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_services() -> list[dict[str, Any]]:
        """List proxied services known to Janus (id, name, ports, protocol)."""
        return await client.list_services()

    @mcp.tool(
        tags={"traffic"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    )
    async def get_filter_dsl() -> str:
        """Short Janus filter-DSL cheatsheet for building `q` expressions.

        Call ONCE per task if you are unsure of syntax. For the full reference
        read the MCP resource `docs://janus/filters`.
        """
        return FILTER_CHEATSHEET

    @mcp.tool(
        tags={"traffic"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def validate_filter(expression: str) -> dict[str, Any]:
        """Validate a Janus filter DSL expression before using it.

        Returns `{"ok": true}` or `{"ok": false, "error": "...", "position": N}`.
        Always run this on non-trivial `q` strings before calling list_packets.
        """
        return await client.validate_filter(expression)

    @mcp.tool(
        tags={"traffic"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_packets(
        q: Annotated[
            str | None,
            Field(description="Janus filter DSL expression. Empty matches all. Validate first."),
        ] = None,
        limit: Annotated[
            int,
            Field(ge=1, le=2000, description="Max packets to return (default 200; capped by server)."),
        ] = 200,
        offset: Annotated[int, Field(ge=0, description="Pagination offset.")] = 0,
        sort: Literal["asc", "desc"] = "desc",
        summary: Annotated[
            bool,
            Field(description="If true (default) bodies/headers are stripped; use get_packet for full detail."),
        ] = True,
    ) -> dict[str, Any]:
        """Query captured packets via the Janus filter DSL.

        Returns `{packets, total, limit, offset}`. Compose service / method /
        direction / time filters inside `q`. For 'recent' use sort=desc + larger
        limit — never invent time_from/time_to.
        """
        capped = min(limit, s.janus_max_limit)
        if summary:
            capped = min(capped, s.janus_summary_max_limit)
        params: dict[str, Any] = {
            "q": q,
            "limit": capped,
            "offset": offset,
            "sort": sort,
        }
        if summary:
            params["summary"] = "true"
        result = await client.list_packets(params)
        if summary and isinstance(result, dict):
            packets = result.get("packets")
            if isinstance(packets, list):
                result["packets"] = [
                    _summarize_packet(packet) if isinstance(packet, dict) else packet
                    for packet in packets
                ]
                if limit > capped:
                    result["mcp_truncated"] = True
                    result["mcp_requested_limit"] = limit
                    result["mcp_returned_limit"] = capped
                    result["mcp_next_offset"] = offset + len(result["packets"])
                    result["mcp_hint"] = (
                        "Summary results are capped by janus-mcp to keep agent context "
                        "bounded. Narrow q, increase offset, or fetch candidate details "
                        "with get_packet/get_flow."
                    )
        return result

    @mcp.tool(
        tags={"traffic"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_packet(
        packet_id: Annotated[int, Field(ge=1, description="Packet ID returned by list_packets.")],
    ) -> dict[str, Any]:
        """Fetch a single packet with full headers + body (body trimmed if huge,
        noisy HTTP headers stripped, raw `body` bytes dropped)."""
        pkt = await client.get_packet(packet_id)
        _slim_headers(pkt)
        return _trim_body(pkt, s.janus_body_max_bytes)

    @mcp.tool(
        tags={"traffic"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_flow(
        packet_id: Annotated[int, Field(ge=1, description="Any packet ID belonging to the flow.")],
    ) -> dict[str, Any]:
        """Reconstruct the full request/response flow for a packet (bodies trimmed,
        noisy HTTP headers stripped, raw `body` bytes dropped)."""
        flow = await client.get_flow(packet_id)

        packets = flow.get("packets", [])
        if len(packets) > s.janus_max_flow_packets:
            target_idx = next(
                (idx for idx, packet in enumerate(packets) if packet.get("id") == packet_id),
                0,
            )
            half_window = s.janus_max_flow_packets // 2
            start = max(0, target_idx - half_window)
            end = start + s.janus_max_flow_packets
            if end > len(packets):
                end = len(packets)
                start = max(0, end - s.janus_max_flow_packets)

            flow["packets"] = packets[start:end]
            flow["flow_truncated"] = True
            flow["flow_original_packet_count"] = len(packets)
            flow["flow_window_start"] = start
            flow["flow_window_end"] = end
            flow["flow_truncated_reason"] = (
                f"max {s.janus_max_flow_packets} packets around packet_id {packet_id}"
            )
            packets = flow.get("packets") or []
        slimmed: list[dict[str, Any]] = []
        for p in packets:
            if isinstance(p, dict):
                _slim_headers(p)
                slimmed.append(_trim_body(p, s.janus_body_max_bytes))
            else:
                slimmed.append(p)
        flow["packets"] = slimmed
        return flow

    @mcp.tool(
        tags={"traffic"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_capture_status() -> dict[str, Any]:
        """Capture state: mode (live/static), capturing flag, current window."""
        return await client.get_capture_status()
