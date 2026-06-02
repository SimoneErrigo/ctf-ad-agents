from __future__ import annotations

import os
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

SUPERVISOR_SYSTEM_PROMPT = (
    "You coordinate an Attack & Defense CTF assistant for a human operator.\n\n"
    "You have ONE tool: dispatch_security_task(query). Use it for ANY operator "
    "goal about: recent traffic analysis, Janus alert/drop rules, source "
    "patch/deploy/rollback, or writing/testing/pushing/starting/stopping an "
    "exploit. Call dispatch_security_task EXACTLY ONCE per operator message, even "
    "when the request has multiple parts (e.g. analyze traffic AND audit the "
    "source): pass the ENTIRE request verbatim in that one query and the pipeline "
    "splits it across specialists. NEVER emit more than one dispatch call in a "
    "turn. Preserve the operator's exact scope: service name, named bug/class, "
    "source-vs-traffic wording, and requested actions (test, push, start, patch, "
    "rule). If the operator names a bug/class and does not explicitly say "
    "from/live/observed traffic, mark the task SOURCE-ONLY.\n\n"
    "For small talk, clarification, or anything you can answer from the "
    "conversation, just reply, do NOT dispatch.\n\n"
    "After you dispatch, the specialists run and the operator is given their "
    "synthesized result, so that turn is done. Earlier specialist results may "
    "already be in the conversation; dispatch only for the operator's CURRENT "
    "request."
)


@tool
def dispatch_security_task(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Hand the operator's security request to the specialist pipeline (traffic
    analysis, Janus rules, source patch, or exploit). Pass the full task verbatim,
    preserving service name, named bug/class, source-vs-traffic wording, and the
    requested actions."""
    # Hand off to the parent graph's classify node, which fans out to the
    # specialists. Because we leave the agent subgraph via Command(graph=PARENT),
    # the subgraph's local state (including the AIMessage that carries THIS tool
    # call) is never propagated to the parent, only this Command's update reaches
    # it. So carry that AIMessage up alongside the ToolMessage: otherwise the
    # parent transcript holds an orphaned tool_result and Bedrock rejects the next
    # supervisor turn with "toolResult blocks exceed toolUse blocks". add_messages
    # dedupes by id, so re-emitting the AIMessage here is safe.
    tool_call_message = state["messages"][-1]
    return Command(
        goto="classify",
        graph=Command.PARENT,
        update={
            "query": query,
            "messages": [
                tool_call_message,
                ToolMessage("Routing to specialists.", tool_call_id=tool_call_id),
            ],
        },
    )


def _text_only(content: object) -> object | None:
    """Keep text, drop tool_use blocks. Returns None if nothing remains."""
    if isinstance(content, str):
        return content or None
    kept = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_use")]
    return kept or None


def _strip_tool_noise(messages: list) -> list:
    """The supervisor's read-only view of the shared transcript: drop ToolMessages
    and tool_use blocks. The channel stays full for the UI; this only trims what
    the synthesis model ingests (cost) and keeps specialists' foreign tool_use
    out of the supervisor's single-tool context."""
    out: list = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage):
            content = _text_only(msg.content)
            if content is None:
                continue
            out.append(AIMessage(content=content, name=msg.name, id=msg.id))
        else:
            out.append(msg)
    return out


@wrap_model_call
async def _supervisor_context_save(request, handler):
    # The supervisor only chats/routes (synthesis is the `final` node), but on
    # multi-turn threads it re-enters and would re-ingest the specialists' raw tool
    # traffic; strip it so the routing turn stays cheap and free of foreign tool_use.
    return await handler(request.override(messages=_strip_tool_noise(request.messages)))


def build_conversational_llm() -> ChatBedrockConverse:
    """LLM powering the supervisor agent."""
    return ChatBedrockConverse(
        name="conversational-agent",
        model=os.environ.get("CHAT_AGENT_MODEL", os.environ["ROUTER_AGENT_MODEL"]),
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        temperature=0.1,
    )


def build_supervisor_agent():
    """Supervisor node: the operator-facing front door. Chats/clarifies, or hands a
    security goal off to the specialist pipeline (the `final` node writes the
    synthesized reply, not the supervisor). No checkpointer, the top-level graph
    (Agent Server) owns persistence."""
    return create_agent(
        model=build_conversational_llm(),
        tools=[dispatch_security_task],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=[_supervisor_context_save],
        name="supervisor",
    )
