from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse

CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are a blue-team assistant for an Attack & Defense CTF competition. "
    "You help the operator understand captured traffic and ongoing attacks. "
    "Use the `security_analysis` tool to investigate: it runs a full traffic "
    "analysis for a service/question and returns a report. Call it whenever "
    "the user asks about attacks, exploits, suspicious activity, or the state "
    "of a specific service. The conversation is persistent: rely on earlier "
    "turns for context and only re-run the tool when new analysis is needed."
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
