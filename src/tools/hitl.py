from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt

log = logging.getLogger(__name__)

def is_approval(decision: Any) -> bool:
    """Return True iff `decision` represents an explicit operator approval."""
    if isinstance(decision, dict):
        return bool(decision.get("approved"))
    if isinstance(decision, str):
        return decision.strip().lower() in {"approve", "approved", "yes", "y", "ok"}
    return False


def rejection_reason(decision: Any) -> str | None:
    """Extraction of a rejection reason from the resume payload."""
    if isinstance(decision, dict):
        reason = decision.get("reason")
        return reason if isinstance(reason, str) and reason else None
    if isinstance(decision, str):
        s = decision.strip()
        if s.lower().startswith("reject"):
            # accept "reject" and "reject: <reason>"
            rest = s.split(":", 1)
            return rest[1].strip() if len(rest) == 2 else None
        return None
    return None


def _format_rejection(payload: dict[str, Any], decision: Any) -> dict[str, Any]:
    return {
        "status": "rejected",
        "reason": rejection_reason(decision) or "operator rejected",
        "payload": payload,
    }


def _wrap_with_hitl(
    mcp_tool: BaseTool,
    *,
    type: str,
    description_suffix: str,
    should_control: callable,
    summary: callable,
) -> BaseTool:
    """Generic HITL wrapper.

    Reuses the MCP tool's args_schema so the agent sees the same signature.
    """

    async def _controlled(**kwargs: Any) -> Any:
        if should_control(kwargs):
            decision = interrupt({
                "type": type,
                "tool": mcp_tool.name,
                "summary": summary(kwargs),
                "payload": kwargs,
                "prompt": (
                    "Operator approval required for a critical action. "
                    "Reply with {'approved': true} to apply, or "
                    "{'approved': false, 'reason': '...'} to reject."
                ),
            })
            if not is_approval(decision):
                log.info("HITL rejected %s: %s", mcp_tool.name, kwargs)
                return _format_rejection(kwargs, decision)
            log.info("HITL approved %s", mcp_tool.name)
        return await mcp_tool.ainvoke(kwargs)

    return StructuredTool.from_function(
        coroutine=_controlled,
        name=mcp_tool.name,
        description=f"{mcp_tool.description}\n\n{description_suffix}",
        args_schema=mcp_tool.args_schema,
    )


def wrap_rule_create(mcp_tool: BaseTool) -> BaseTool:
    """Require operator approval for create_rule when action is drop or both."""

    def _should_control(kw: dict[str, Any]) -> bool:
        return kw.get("action", "alert") in ("drop", "both")

    def _summary(kw: dict[str, Any]) -> str:
        return (
            f"Create rule '{kw.get('name')}' on service '{kw.get('service_id')}' "
            f"with action='{kw.get('action')}' — expression: {kw.get('expression')!r}"
        )

    return _wrap_with_hitl(
        mcp_tool,
        type="approve_rule_create",
        description_suffix=(
            "HITL: when action='drop' or 'both', the call is paused and the "
            "operator must approve before the rule is created in Janus. "
            "action='alert' is applied immediately without approval."
        ),
        should_control=_should_control,
        summary=_summary,
    )


def wrap_rule_update(mcp_tool: BaseTool) -> BaseTool:
    """Require operator approval for update_rule when the new action is drop or both.

    We always require approval for updates that result in a drop, even if the
    rule was previously a drop, the operator should re-confirm any change to
    live blocking behavior.
    """

    def _should_control(kw: dict[str, Any]) -> bool:
        return kw.get("action") in ("drop", "both")

    def _summary(kw: dict[str, Any]) -> str:
        return (
            f"Update rule id='{kw.get('rule_id')}' on service "
            f"'{kw.get('service_id')}' to action='{kw.get('action')}' — "
            f"expression: {kw.get('expression')!r}"
        )

    return _wrap_with_hitl(
        mcp_tool,
        type="approve_rule_update",
        description_suffix=(
            "HITL: updates that set action='drop' or 'both' are paused until "
            "the operator approves. Updates that keep / set action='alert' "
            "are applied immediately."
        ),
        should_control=_should_control,
        summary=_summary,
    )


def wrap_patch_deploy(mcp_tool: BaseTool) -> BaseTool:
    """Require operator approval for any patch deploy. Always requires approval."""

    def _summary(kw: dict[str, Any]) -> str:
        return (
            f"Deploy patch for service '{kw.get('service')}', branch "
            f"{kw.get('branch', 'main')!r}, message: {kw.get('message')!r}"
        )

    return _wrap_with_hitl(
        mcp_tool,
        type="approve_patch_deploy",
        description_suffix=(
            "HITL: deploying a patch pushes code to the competition VM and "
            "triggers a service rebuild, always paused for operator approval."
        ),
        should_control=lambda _kw: True,
        summary=_summary,
    )


def wrap_patch_rollback(mcp_tool: BaseTool) -> BaseTool:
    """Require operator approval for rollback, same reasoning as deploy."""

    def _summary(kw: dict[str, Any]) -> str:
        return (
            f"Rollback service '{kw.get('service')}' to commit "
            f"{kw.get('commit_sha')!r}"
        )

    return _wrap_with_hitl(
        mcp_tool,
        type="approve_patch_rollback",
        description_suffix=(
            "HITL: a rollback rewinds the deployed code on the VM, always "
            "paused for operator approval."
        ),
        should_control=lambda _kw: True,
        summary=_summary,
    )


def wrap_exploit_start(mcp_tool: BaseTool) -> BaseTool:
    """Require operator approval before launching an exploit against all teams."""

    def _summary(kw: dict[str, Any]) -> str:
        return f"Launch exploit '{kw.get('name')}' against ALL teams (real attack)"

    return _wrap_with_hitl(
        mcp_tool,
        type="approve_exploit_start",
        description_suffix=(
            "HITL: starting an exploit attacks every team's service and submits "
            "flags, always paused for operator approval. Use test_exploit "
            "(NOP team) first; it is NOT gated."
        ),
        should_control=lambda _kw: True,
        summary=_summary,
    )


def wrap_exploit_push(mcp_tool: BaseTool) -> BaseTool:
    """Require operator approval before uploading exploit source to the farm."""

    def _summary(kw: dict[str, Any]) -> str:
        return f"Push exploit '{kw.get('name')}' source to the farm, message: {kw.get('message')!r}"

    return _wrap_with_hitl(
        mcp_tool,
        type="approve_exploit_push",
        description_suffix=(
            "HITL: pushing uploads the exploit source to the shared farm where "
            "workers pick it up, paused for operator approval."
        ),
        should_control=lambda _kw: True,
        summary=_summary,
    )


_DEFAULT_WRAPPERS: dict[str, callable] = {
    "create_rule": wrap_rule_create,
    "update_rule": wrap_rule_update,
    "deploy": wrap_patch_deploy,
    "rollback": wrap_patch_rollback,
    "start_exploit": wrap_exploit_start,
    "push_exploit": wrap_exploit_push,
}


def apply_hitl(tools: list[BaseTool], wrappers: dict[str, callable] | None = None) -> list[BaseTool]:
    """Return a new list of tools with HITL wrappers applied by name."""
    table = wrappers if wrappers is not None else _DEFAULT_WRAPPERS
    out: list[BaseTool] = []
    for t in tools:
        wrapper = table.get(t.name)
        out.append(wrapper(t) if wrapper else t)
    return out
