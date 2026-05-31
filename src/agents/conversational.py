from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse

CONVERSATIONAL_SYSTEM_PROMPT = (
    "You assist an Attack & Defense CTF operator. You have one tool: "
    "security_analysis(query). Use it once for any goal about traffic, Janus "
    "rules, patch/deploy/rollback, or exploit write/test/start. Do not split a "
    "single operator goal into multiple tool calls.\n\n"
    "When calling the tool, preserve the operator's exact scope: service name, "
    "vulnerability class, source-vs-traffic wording, and requested actions "
    "(test, push, start, patch, rule). Do not broaden the task. If the operator "
    "names a bug/class and does not explicitly say from/live/observed traffic, "
    "mark exploit/patch goals as SOURCE-ONLY in the tool query.\n\n"
    "HITL actions pause for approval: drop/both rules, deploy/rollback, "
    "push_exploit, and start_exploit. If approval is rejected, report it and stop.\n\n"
    "After the tool returns, relay the result honestly. Do not retry when the "
    "result starts with `[security_analysis failed` or `[security_analysis error`. "
    "Never upgrade failure or pending status into success. For exploits, claim "
    "LIVE/ACTIVE/pushed/started/flags only if the tool result explicitly shows "
    "a successful test with flags_found>=1 plus successful push/start."
)


def build_conversational_llm() -> ChatBedrockConverse:
    """LLM powering the conversational agent."""
    return ChatBedrockConverse(
        name="conversational-agent",
        model=os.environ.get("CHAT_AGENT_MODEL", os.environ["ROUTER_AGENT_MODEL"]),
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        temperature=0.1,
    )


def build_conversational_agent(tools, checkpointer):
    """Build the stateful conversational agent with LangGraph 
       checkpointer providing thread persistence."""
    return create_agent(
        model=build_conversational_llm(),
        tools=tools,
        system_prompt=CONVERSATIONAL_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
