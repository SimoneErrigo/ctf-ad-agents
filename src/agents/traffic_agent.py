from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_aws.middleware import BedrockPromptCachingMiddleware

from src.tools.mcp_client import MCPToolRegistry
from src.tools.traffic_agent_tools import get_traffic_tools

_REQUIRED_ENV = (
    "REGION_NAME",
    "TRAFFIC_AGENT_MODEL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

SYSTEM_PROMPT = (
    "You are a cybersecurity blue-team expert in an Attack & Defense CTF competition. "
    "Analyze traffic collected by Janus, a reverse proxy that captures and logs HTTP traffic between clients and services. "
    "Identify attack patterns, exploited or potentially exploited vulnerabilities, indicators of compromise (IoCs), and suspicious behaviors. "
    "Provide a concise report including the affected requests/flows, the suspected attack technique, confidence level, and supporting evidence. "
    "Do not attribute attacks based on IP addresses, as IPs are unreliable in CTF environments. "
    "Distinguish malicious activity from legitimate checker traffic and explicitly flag uncertain cases.\n\n"
    "INVESTIGATION METHOD: be fast and decisive, you are time-constrained:\n"
    "- Stop early: as soon as you can identify the attack(s) on the requested service "
    "with supporting evidence, STOP calling tools and write the report. Do NOT pursue "
    "exhaustiveness, do NOT scan unrelated services, do NOT page through old history.\n"
    "- Recent traffic only: an A&D game runs in rounds and only recent traffic matters. "
    "Query with `list_packets(sort=\"desc\", limit<=200)` and call `get_capture_status` "
    "once to learn the current window. There is NO round/time field in the filter DSL, "
    "so 'recent' means sort=desc + a bounded limit, never invent time_from/time_to and "
    "never try to filter by round. You may read `flagid_round` in the returned rows to "
    "judge how recent a packet is and ignore clearly old ones.\n"
    "- Filter server-side, do not scan client-side: always narrow with `q`. Start from "
    "attack indicators that exist as fields, e.g. `(flagged OR contains_flagid) AND "
    "direction == \"request\"`, then payload substrings (`body icontains \"select\"`, "
    "`body icontains \"union\"`, `url icontains \"../\"`, `body contains \"<script\"`, "
    "`body icontains \"/etc/passwd\"`, `body icontains \"sleep(\"`). One well-built `q` "
    "beats many broad listings. Validate non-trivial `q` with `validate_filter`; consult "
    "`get_filter_dsl` at most once.\n"
    "- Few rounds, batched: plan your queries up front and issue independent "
    "`list_packets`/`get_packet` calls in parallel within a single step. Use "
    "`get_packet`/`get_flow` only for the few packets you will actually cite as evidence. "
    "Budget yourself to roughly 2-3 rounds of tool calls."
)


def _assert_env() -> None:
    missing = [k for k in _REQUIRED_ENV if k not in os.environ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables for traffic agent: "
            + ", ".join(missing)
        )


def _build_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=os.environ["TRAFFIC_AGENT_MODEL"],
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        temperature=0.1,
    )


async def build_traffic_agent(registry: MCPToolRegistry | None = None):
    """Build the traffic agent with tools loaded from the Janus MCP registry."""
    _assert_env()
    tools = await get_traffic_tools(registry)
    return create_agent(
        model=_build_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            BedrockPromptCachingMiddleware(
                ttl="5m",
                min_messages_to_cache=0,
                unsupported_model_behavior="raise",
            )
        ],
    )
