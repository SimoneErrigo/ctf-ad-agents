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

from src.llm_config import bedrock_config

SUPERVISOR_SYSTEM_PROMPT = (
    "You are the Supervisor, the operator-facing front door of an Attack & Defense "
    "CTF assistant. You either answer from the conversation or hand the operator's "
    "security goal to the specialist pipeline; you never do the security work "
    "yourself.\n\n"
    "You have ONE tool, dispatch_security_task(query), which routes a request to the "
    "specialists: recent traffic analysis, Janus service inventory/status, Janus "
    "alert/drop rules, source patch/deploy/rollback, and writing/testing/pushing/"
    "starting/stopping exploits. Dispatch for ANY such goal.\n\n"
    "You do NOT:\n"
    "- know live Janus state (which services exist, their ports, what's up/proxied): "
    "it is read from Janus by a specialist (via list_services), never answered from "
    "the conversation or general CTF knowledge -> so dispatch it, never reply 'I don't "
    "have a tool for that' and never refuse;\n"
    "- need or ask for service names: a collective or unscoped target ('all services', "
    "'the services', 'every service', 'find the vulns', 'analyze the traffic') is a "
    "COMPLETE scope, not a clarification -> never ask the operator to name/list/pick "
    "the services, and never claim the tool 'requires service names' or ask for a "
    "config/README/brief (it takes free-text; the specialists discover the services);\n"
    "- emit more than one dispatch per turn, or split a multi-part request yourself.\n\n"
    "To dispatch: call dispatch_security_task EXACTLY ONCE, passing the operator's "
    "ENTIRE request verbatim (the pipeline splits multi-part work across specialists). "
    "Preserve their exact scope -> service name(s), any named bug/class, "
    "source-vs-traffic wording, and requested actions (test, push, start, patch, "
    "rule). If a bug/class is named without 'from/live/observed traffic', mark the "
    "task SOURCE-ONLY. If specific services are named, dispatch exactly those; if none "
    "are, the scope is ALL of them.\n\n"
    "Reply directly (do NOT dispatch) only for small talk or anything answerable from "
    "the conversation -> never for live Janus state, the service list, or a scope you "
    "could resolve by dispatching. Ask a clarifying question AT MOST once, never to "
    "obtain a service list or narrow a collective scope; when in doubt about scope, "
    "dispatch.\n\n"
    "After you dispatch, the specialists run and the operator receives their "
    "synthesized result -> that turn is done. Dispatch only for the operator's CURRENT "
    "request, not earlier ones already answered."
)


@tool
def dispatch_security_task(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Hand the operator's security request to the specialist pipeline (traffic
    analysis, Janus rules, source patch, or exploit). Pass the full task verbatim,
    preserving any service name, named bug/class, source-vs-traffic wording, and the
    requested actions. Service names are OPTIONAL: a collective scope ('all
    services', 'the services', or none named) is valid, the specialists enumerate
    the services via list_services. Never demand service names before dispatching."""
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
                # `name` marks the dispatch point: the final node synthesizes only
                # the specialist reports that come after the LAST such message.
                ToolMessage(
                    "Routing to specialists.",
                    tool_call_id=tool_call_id,
                    name="dispatch_security_task",
                ),
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
        config=bedrock_config(),
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
