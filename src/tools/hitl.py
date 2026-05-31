from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt

log = logging.getLogger(__name__)

# Agent Inbox interrupt contract. We raise a HumanInterrupt-shaped value and read
# back a HumanResponse, which Agent Chat UI renders as an accept/respond/ignore
# card. We use plain dicts rather than importing the TypedDicts.
#  Schema: https://github.com/langchain-ai/agent-inbox
#
# What the operator may do with a controlled action. allow_edit stays off: these are
# destructive / outward-facing calls, approve-or-reject is enough. Flip
# allow_edit to True to let the operator tweak args before the call runs,
# _interpret_response already applies the edited args, so it is a safe toggle.
_HITL_CONFIG = {
    "allow_accept": True,
    "allow_respond": True,
    "allow_edit": False,
    "allow_ignore": True,
}


def _interpret_response(response: Any) -> tuple[bool, str | None, dict | None]:
    """Map an Agent Inbox HumanResponse to ``(approved, reason, edited_args)``.

    Resume arrives as a list of HumanResponse (one per interrupt raised); we raise
    exactly one, so we read the first element. Anything we do not recognise is
    treated as a rejection, we never silently approve.
    """
    if isinstance(response, list):
        response = response[0] if response else {}
    if not isinstance(response, dict):
        return False, "operator rejected", None

    rtype = response.get("type")
    if rtype == "accept":
        return True, None, None
    if rtype == "edit":
        # Approved with edits: HumanResponse.args is an ActionRequest
        # {"action": ..., "args": {...}}; run the tool with the new args.
        request = response.get("args")
        edited = request.get("args") if isinstance(request, dict) else None
        return True, None, edited if isinstance(edited, dict) else None
    if rtype == "response":
        reason = response.get("args")
        return (
            False,
            reason if isinstance(reason, str) and reason else "operator rejected",
            None,
        )
    # "ignore" or anything unexpected
    return False, "operator rejected", None


def _wrap_with_hitl(
    mcp_tool: BaseTool,
    *,
    description_suffix: str,
    should_control: callable,
    summary: callable,
) -> BaseTool:
    """Generic HITL wrapper.

    Reuses the MCP tool's args_schema so the agent sees the same signature. When
    ``should_control(args)`` is true the call pauses via an Agent Inbox interrupt
    until the operator accepts (optionally with edited args) or rejects.
    """

    async def _controlled(**kwargs: Any) -> Any:
        if should_control(kwargs):
            response = interrupt([
                {
                    "action_request": {"action": mcp_tool.name, "args": kwargs},
                    "config": _HITL_CONFIG,
                    "description": summary(kwargs),
                }
            ])
            approved, reason, edited = _interpret_response(response)
            if not approved:
                log.info("HITL rejected %s: %s", mcp_tool.name, reason)
                return {"status": "rejected", "reason": reason, "payload": kwargs}
            if edited:
                log.info("HITL approved %s with edits", mcp_tool.name)
                kwargs = edited
            else:
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
        description_suffix=(
            "HITL: starting an exploit attacks every team's service and submits "
            "flags, always paused for operator approval. Use test_exploit "
            "(default target host) first; it is NOT gated."
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


def apply_hitl(tools: list[BaseTool]) -> list[BaseTool]:
    """Return a new list of tools with HITL wrappers applied by name."""
    out: list[BaseTool] = []
    for t in tools:
        wrapper = _DEFAULT_WRAPPERS.get(t.name)
        out.append(wrapper(t) if wrapper else t)
    return out
