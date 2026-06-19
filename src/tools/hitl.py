from __future__ import annotations

from typing import Any

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage

# Human-in-the-loop via LangChain's HumanInTheLoopMiddleware, attached per
# sub-agent: it gates ONLY the critical tools registered below; every other tool
# runs autonomously. Several hitl tool calls in one turn (e.g. "start all
# exploits") are batched into a SINGLE interrupt -> one approval card.

# Operators may approve or reject.
_DECISIONS = ["approve", "reject"]


def _describe(summary, detail=None):
    def _factory(tool_call: dict[str, Any], state: Any, runtime: Any) -> str:
        line = summary(tool_call.get("args", {}))
        return line + (detail(state) if detail else "")

    return _factory


def _control(summary, detail=None) -> dict[str, Any]:
    return {"allowed_decisions": _DECISIONS, "description": _describe(summary, detail)}


def _clip(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


def _last_tool_msg(state: Any, name: str):
    for m in reversed(state.get("messages") or []):
        if isinstance(m, ToolMessage) and m.name == name:
            return m
    return None


# approval-card summaries, one per critical tool

def _create_rule(a: dict[str, Any]) -> str:
    return (
        f"Create rule '{a.get('name')}' on service '{a.get('service_id')}' "
        f"with action='{a.get('action')}', expression: {a.get('expression')!r}"
    )


def _delete_rule(a: dict[str, Any]) -> str:
    return f"Delete rule id='{a.get('rule_id')}' (removes a live defense)"


def _update_rule(a: dict[str, Any]) -> str:
    return (
        f"Update rule id='{a.get('rule_id')}' on service "
        f"'{a.get('service_id')}' to action='{a.get('action')}', "
        f"expression: {a.get('expression')!r}"
    )


def _deploy(a: dict[str, Any]) -> str:
    return (
        f"Deploy patch for service '{a.get('service')}', branch "
        f"{a.get('branch', 'main')!r}, message: {a.get('message')!r}"
    )


def _rollback(a: dict[str, Any]) -> str:
    return f"Rollback service '{a.get('service')}' to commit {a.get('commit_sha')!r}"


def _rollback_to(a: dict[str, Any]) -> str:
    target = a.get("target_commit")
    where = f"commit {target!r}" if target else "the first (seed) commit, dropping all patches"
    return f"Rollback service '{a.get('service')}' to {where}"


def _start_exploit(a: dict[str, Any]) -> str:
    return f"Launch exploit '{a.get('name')}' against ALL teams (real attack)"


def _push_exploit(a: dict[str, Any]) -> str:
    return f"Push exploit '{a.get('name')}' source to the farm, message: {a.get('message')!r}"


def _deploy_detail(state: Any) -> str:
    m = _last_tool_msg(state, "get_diff")
    if m is None:
        return "\n\n(no diff in transcript, review the stream)"
    art = getattr(m, "artifact", None) or {}
    diff = (art.get("structured_content") or {}).get("diff") if isinstance(art, dict) else None
    diff = diff or (m.content if isinstance(m.content, str) else str(m.content))
    return f"\n\n```diff\n{_clip(diff)}\n```"


def _exploit_detail(state: Any) -> str:
    """Preview the MOST RECENT exploit edit: full source for write_exploit_file,
    the old->new snippet for replace_text (the extend workflow edits in place,
    so showing only the last full write would be stale or missing)."""
    for m in reversed(state.get("messages") or []):
        for tc in reversed(getattr(m, "tool_calls", None) or []):
            args = tc.get("args") or {}
            if tc.get("name") == "write_exploit_file":
                return f"\n\n```python\n{_clip(args.get('content') or '')}\n```"
            if tc.get("name") == "replace_text":
                return (
                    f"\n\nLatest edit to {args.get('path')!r}:\n```\n"
                    f"--- old\n{_clip(args.get('old') or '', 800)}\n"
                    f"+++ new\n{_clip(args.get('new') or '', 800)}\n```"
                )
    return "\n\n(no source in transcript, review the stream)"


# per-agent middleware factories

def traffic_hitl() -> HumanInTheLoopMiddleware:
    """HITL Janus rule writes.

    NOTE: HITL create_rule/update_rule for ANY action (alert included), not just
    drop/both, the middleware keys on tool name, not args, and any live rule
    write is worth an operator click. delete_rule is gated too: removing a live
    drop rule takes a defense down, which is at least as dangerous as adding one.
    """
    return HumanInTheLoopMiddleware(
        interrupt_on={
            "create_rule": _control(_create_rule),
            "update_rule": _control(_update_rule),
            "delete_rule": _control(_delete_rule),
        }
    )


def patch_hitl() -> HumanInTheLoopMiddleware:
    """Gate code deploy / rollback to the competition VM."""
    return HumanInTheLoopMiddleware(
        interrupt_on={
            "deploy": _control(_deploy, _deploy_detail),
            "rollback": _control(_rollback),
            "rollback_to": _control(_rollback_to),
        }
    )


def exploit_hitl() -> HumanInTheLoopMiddleware:
    """Gate the offensive / outward-facing exploit actions."""
    return HumanInTheLoopMiddleware(
        interrupt_on={
            "start_exploit": _control(_start_exploit),
            "push_exploit": _control(_push_exploit, _exploit_detail),
        }
    )
