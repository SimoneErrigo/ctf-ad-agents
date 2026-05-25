"""the traffic agent must stay read-only on captured packets. 
HITL is NOT enforced here"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from janus_mcp.client import JanusClient

log = logging.getLogger(__name__)


_RULE_ACTIONS = ("drop", "alert", "both")


def register_defender_tools(mcp: FastMCP, client: JanusClient) -> None:
    """Register the defender CRUD surface on `mcp`, tagged `defender`."""

    @mcp.tool(
        tags={"defender"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_rules(
        service_id: Annotated[
            str | None,
            Field(description="Optional service ID; if omitted, returns rules for all services."),
        ] = None,
    ) -> list[dict[str, Any]]:
        """List Janus drop/alert rules.

        Useful before creating a new rule to avoid duplicates: Janus rejects
        a POST whose (service_id, expression, action) already exists.
        """
        return await client.list_rules(service_id)

    @mcp.tool(
        tags={"defender"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_rule(
        rule_id: Annotated[str, Field(description="Rule ID returned by list_rules.")],
    ) -> dict[str, Any]:
        """Fetch a single rule by ID."""
        return await client.get_rule(rule_id)

    @mcp.tool(
        tags={"defender"},
        annotations={"destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
    async def create_rule(
        service_id: Annotated[str, Field(description="ID of the service the rule applies to (see list_services).")],
        name: Annotated[str, Field(min_length=1, description="Short human-readable rule name.")],
        expression: Annotated[
            str,
            Field(
                min_length=1,
                description=(
                    "Janus filter DSL expression (same language as list_packets `q`). "
                    "Validate it with validate_filter before submitting."
                ),
            ),
        ],
        action: Annotated[
            Literal["drop", "alert", "both"],
            Field(description="`alert` only logs a match; `drop` blocks the request; `both` does both."),
        ] = "alert",
        priority: Annotated[
            int,
            Field(ge=0, description="Higher priority rules are evaluated first. Default 0."),
        ] = 0,
        enabled: Annotated[bool, Field(description="Create the rule in enabled state.")] = True,
    ) -> dict[str, Any]:
        """Create a Janus rule. Default action is `alert`, the caller must
        explicitly pass `action=\"drop\"` or `\"both\"` for blocking, and the
        LangChain wrapper is expected to gate that on operator approval."""
        rule = {
            "service_id": service_id,
            "name": name,
            "expression": expression,
            "action": action,
            "priority": priority,
            "enabled": enabled,
            "type": "",
            "scope": "",
            "pattern": "",
        }
        return await client.create_rule(rule)

    @mcp.tool(
        tags={"defender"},
        annotations={"destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
    async def update_rule(
        rule_id: Annotated[str, Field(description="ID of the rule to update.")],
        service_id: Annotated[str, Field(description="Service ID the rule applies to.")],
        name: Annotated[str, Field(min_length=1, description="Rule name.")],
        expression: Annotated[str, Field(min_length=1, description="Janus filter DSL expression.")],
        action: Annotated[Literal["drop", "alert", "both"], Field(description="Match action.")],
        priority: Annotated[int, Field(ge=0, description="Evaluation priority.")] = 0,
        enabled: Annotated[bool, Field(description="Enabled state.")] = True,
    ) -> dict[str, Any]:
        """Update an existing rule. Janus requires a full body, pass every field."""
        rule = {
            "id": rule_id,
            "service_id": service_id,
            "name": name,
            "expression": expression,
            "action": action,
            "priority": priority,
            "enabled": enabled,
            "type": "",
            "scope": "",
            "pattern": "",
        }
        return await client.update_rule(rule_id, rule)

    @mcp.tool(
        tags={"defender"},
        annotations={"destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def delete_rule(
        rule_id: Annotated[str, Field(description="ID of the rule to delete.")],
    ) -> dict[str, Any]:
        """Delete a Janus rule. Returns `{deleted: rule_id}` on success."""
        await client.delete_rule(rule_id)
        return {"deleted": rule_id}

    @mcp.tool(
        tags={"defender"},
        annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_alerts(
        service_id: Annotated[str | None, Field(description="Filter by service ID.")] = None,
        rule_id: Annotated[str | None, Field(description="Filter by rule ID.")] = None,
        src_ip: Annotated[str | None, Field(description="Filter by source IP.")] = None,
        q: Annotated[
            str | None,
            Field(description="Optional filter-DSL expression evaluated against the linked packet."),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=500, description="Page size.")] = 50,
        offset: Annotated[int, Field(ge=0, description="Pagination offset.")] = 0,
    ) -> dict[str, Any]:
        """List alerts (rule matches). Use this after creating an alert rule to
        confirm it is firing on real traffic before promoting it to `drop`."""
        params: dict[str, Any] = {
            "service_id": service_id,
            "rule_id": rule_id,
            "src_ip": src_ip,
            "q": q,
            "limit": limit,
            "offset": offset,
        }
        return await client.list_alerts(params)
