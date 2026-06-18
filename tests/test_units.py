"""Unit tests for the pure graph logic (no network, no MCP, no Bedrock)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Send

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.conversational import _strip_tool_noise
from src.agents.orchestrator import dispatch_next, make_final_node
from src.tools.hitl import _exploit_detail


class FakeLLM:
    """Captures the prompt and returns a canned reply."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="synthesized")


def _dispatch_marker() -> ToolMessage:
    return ToolMessage(
        "Routing to specialists.", tool_call_id="t1", name="dispatch_security_task"
    )


# dispatch_next (sequential specialist dispatch)

def test_dispatch_sends_first_pending_specialist():
    state = {"pending": [
        {"source": "traffic", "query": "q1"},
        {"source": "exploit", "query": "q2"},
    ]}
    cmd = dispatch_next(state)
    assert isinstance(cmd.goto, Send)
    assert cmd.goto.node == "traffic"
    assert cmd.goto.arg["messages"][0].content == "q1"
    # the dispatched specialist is dropped from the queue; the rest carry over so the
    # next loop back into `dispatch` runs them one at a time.
    assert cmd.update["pending"] == [{"source": "exploit", "query": "q2"}]


def test_dispatch_empty_queue_goes_to_final():
    assert dispatch_next({"pending": []}).goto == "final"


# final node

def test_final_only_synthesizes_reports_after_current_dispatch():
    llm = FakeLLM()
    node = make_final_node(llm)
    state = {
        "query": "current request",
        "classifications": [{"source": "traffic", "query": "q"}],
        "messages": [
            AIMessage(content="STALE old-turn report", name="traffic"),
            _dispatch_marker(),
            AIMessage(content="FRESH report", name="traffic"),
        ],
    }
    asyncio.run(node(state))
    human = llm.calls[0][1].content
    assert "FRESH report" in human
    assert "STALE old-turn report" not in human


def test_final_with_no_reports_replies_without_calling_llm():
    llm = FakeLLM()
    node = make_final_node(llm)
    state = {"query": "q", "classifications": [], "messages": [_dispatch_marker()]}
    result = asyncio.run(node(state))
    assert llm.calls == []
    msg = result["messages"][0]
    assert msg.name == "supervisor" and msg.content


# supervisor context middleware

def test_strip_tool_noise_drops_tool_traffic_keeps_text():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content=[
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "x", "name": "t", "input": {}},
            ],
            name="traffic",
            id="a1",
        ),
        ToolMessage("result", tool_call_id="x"),
        AIMessage(content=[{"type": "tool_use", "id": "y", "name": "t", "input": {}}]),
    ]
    out = _strip_tool_noise(messages)
    assert [type(m) for m in out] == [HumanMessage, AIMessage]
    assert out[1].content == [{"type": "text", "text": "thinking"}]


# HITL approval-card preview (push_exploit)

def test_exploit_detail_previews_latest_replace_text_edit():
    state = {"messages": [
        AIMessage(content="", tool_calls=[{
            "name": "write_exploit_file", "id": "1",
            "args": {"content": "OLD FULL SOURCE"},
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "replace_text", "id": "2",
            "args": {"path": "main.py", "old": "a()", "new": "b()"},
        }]),
    ]}
    detail = _exploit_detail(state)
    assert "main.py" in detail and "b()" in detail
    assert "OLD FULL SOURCE" not in detail


def test_exploit_detail_previews_full_write():
    state = {"messages": [AIMessage(content="", tool_calls=[{
        "name": "write_exploit_file", "id": "1", "args": {"content": "print(1)"},
    }])]}
    assert "print(1)" in _exploit_detail(state)
