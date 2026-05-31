from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware import HumanInTheLoopMiddleware

# Human-in-the-loop via LangChain's HumanInTheLoopMiddleware, attached per
# sub-agent. It controls ONLY the critical tools registered below; every other tool
# the agent has runs autonomously. When the model emits several hitl tool calls
# in one turn (e.g. "start all exploits"), the middleware batches them into a
# SINGLE interrupt carrying every action_request, which Agent Chat UI renders as
# one approval card and resumes with one decision per action.
#
# The interrupt() the middleware raises bubbles up (GraphBubbleUp) through the
# `security_analysis` tool wrapper to the conversational checkpointer just like a
# hand-rolled interrupt did, so HITL still pauses/resumes the whole run. See
# src/graph.py (the GraphBubbleUp re-raise) and [[hitl-interrupt-propagation]].

# Operators may approve or reject.
_DECISIONS = ["approve", "reject"]


def _describe(summary: Callable[[dict[str, Any]], str]):
    """Adapt a ``summary(args) -> str`` into the middleware's description
    callable signature ``(tool_call, state, runtime) -> str``. The returned
    string is the human-readable line shown on the approval card."""

    def _factory(tool_call: dict[str, Any], state: Any, runtime: Any) -> str:
        return summary(tool_call.get("args", {}))

    return _factory


def _control(summary: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    """InterruptOnConfig (plain dict) for one critical tool."""
    return {"allowed_decisions": _DECISIONS, "description": _describe(summary)}


# approval-card summaries, one per critical tool

def _create_rule(a: dict[str, Any]) -> str:
    return (
        f"Create rule '{a.get('name')}' on service '{a.get('service_id')}' "
        f"with action='{a.get('action')}', expression: {a.get('expression')!r}"
    )


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


def _start_exploit(a: dict[str, Any]) -> str:
    return f"Launch exploit '{a.get('name')}' against ALL teams (real attack)"


def _push_exploit(a: dict[str, Any]) -> str:
    return f"Push exploit '{a.get('name')}' source to the farm, message: {a.get('message')!r}"


# per-agent middleware factories

def traffic_hitl() -> HumanInTheLoopMiddleware:
    """HITL Janus rule writes.

    NOTE: HITL create_rule/update_rule for ANY action (alert included), not just
    drop/both, the middleware keys on tool name, not args, and any live rule
    write is worth an operator click.
    """
    return HumanInTheLoopMiddleware(
        interrupt_on={
            "create_rule": _control(_create_rule),
            "update_rule": _control(_update_rule),
        }
    )


def patch_hitl() -> HumanInTheLoopMiddleware:
    """Gate code deploy / rollback to the competition VM."""
    return HumanInTheLoopMiddleware(
        interrupt_on={
            "deploy": _control(_deploy),
            "rollback": _control(_rollback),
        }
    )


def exploit_hitl() -> HumanInTheLoopMiddleware:
    """Gate the offensive / outward-facing exploit actions."""
    return HumanInTheLoopMiddleware(
        interrupt_on={
            "start_exploit": _control(_start_exploit),
            "push_exploit": _control(_push_exploit),
        }
    )
