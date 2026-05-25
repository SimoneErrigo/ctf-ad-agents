from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langgraph.types import Command

from src.graph import build_chat_app

DEFAULT_QUERY = "Are there any attacks visible in the captured traffic so far?"


def _prompt_decision() -> dict[str, Any]:
    """Ask the operator to approve/reject a HITL interrupt (drop rule, deploy, rollback)."""
    while True:
        try:
            choice = input("Approve? [a]pprove / [r]eject [reason]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return {"approved": False, "reason": "operator cancelled"}
        parts = choice.split(maxsplit=1)
        head = parts[0].lower() if parts else ""
        if head in {"a", "approve", "y", "yes"}:
            return {"approved": True}
        if head in {"r", "reject", "n", "no"}:
            return {"approved": False, "reason": parts[1] if len(parts) > 1 else "rejected"}
        print("(answer 'a' or 'r')")


async def _drive_turn(agent, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Run one user turn, pausing for operator approval whenever the graph interrupts."""
    result = await agent.ainvoke(payload, config)
    while result.get("__interrupt__"):
        value = getattr(result["__interrupt__"][0], "value", None)
        print("\n=== APPROVAL REQUESTED ===")
        if isinstance(value, dict):
            print(f"tool: {value.get('tool', '?')}")
            print(f"action: {value.get('summary', '')}")
        else:
            print(f"  {value!r}")
        result = await agent.ainvoke(Command(resume=_prompt_decision()), config)
    return result


def _print_answer(result: dict[str, Any], prefix: str = "") -> None:
    messages = result.get("messages") or []
    if not messages:
        print(f"{prefix}(no message)")
        return
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content")
    print(f"{prefix}{content}")


async def _run_chat(agent, config: dict[str, Any]) -> None:
    print(f"[chat] thread_id={config['configurable']['thread_id']}. Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            line = input("[you] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.lower() in {"exit", "quit"}:
            return
        if not line:
            continue
        result = await _drive_turn(agent, {"messages": [{"role": "user", "content": line}]}, config)
        _print_answer(result, prefix="[assistant] ")
        print()


async def amain(query: str | None, chat: bool) -> None:
    config = {"configurable": {"thread_id": os.environ.get("THREAD_ID", "cli-session")}}
    async with build_chat_app() as agent:
        if chat:
            await _run_chat(agent, config)
        else:
            result = await _drive_turn(
                agent, {"messages": [{"role": "user", "content": query or DEFAULT_QUERY}]}, config,
            )
            _print_answer(result)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:]
    chat = bool(args) and args[0] in ("--chat", "-c")
    if chat:
        args = args[1:]
    query = " ".join(args).strip() or None
    asyncio.run(amain(query, chat))


if __name__ == "__main__":
    main()
