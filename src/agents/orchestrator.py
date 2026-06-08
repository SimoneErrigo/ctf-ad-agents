from __future__ import annotations

import json
import logging
import os

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.llm_config import bedrock_config
from src.state.state import Classification, RouterState

log = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""

    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )


def build_orchestrator_llm() -> ChatBedrockConverse:
    """LLM powering the orchestrator's classification (routing) step."""
    return ChatBedrockConverse(
        name="orchestrator-llm",
        model=os.environ["ROUTER_AGENT_MODEL"],
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        temperature=0.1,
        config=bedrock_config(),
    )


def build_synthesis_llm() -> ChatBedrockConverse:
    """LLM for the final synthesis node."""
    return ChatBedrockConverse(
        name="synthesis-llm",
        model=os.environ.get("SYNTHESIS_MODEL") or os.environ.get("CHAT_AGENT_MODEL") or os.environ["ROUTER_AGENT_MODEL"],
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        temperature=0.1,
        config=bedrock_config(),
    )


CLASSIFY_PROMPT = """Route the operator request to the smallest set of specialists:
one classification per specialist needed, each a scoped sub-question.

Preserve the operator's scope in every sub-question: service name, any named
bug/class (exploit/patch then target ONLY that, no fallback bugs), source-vs-traffic
wording, and requested actions. SOURCE vs TRAFFIC: if a bug/class is named and the
operator did not explicitly say from/live/observed traffic, the sub-question is
SOURCE-ONLY (do not inspect Janus traffic); generic words like "find" are not traffic.

exploit -> write/test/push/start/stop a Python exploit (xfarm), or report/lifecycle
running ones. Route here for write/create/test/push/start/run/replicate/steal (do NOT
add traffic just to feed it), for stop/kill or "which are running", and for analysis.
- build: "For service X, build ONE Python exploit for ONLY <named bug/class if any>.
  SOURCE-ONLY unless the operator said from/live/observed traffic. Test once, then
  push/start through HITL only after flags_found>=1." ("find the bug then exploit it"
  is a build task.)
- report-only -> ANALYSIS-ONLY: the operator only asks to find/analyze/report/audit/
  describe/list a vuln or attack with NO action verb (write/create/test/push/start/
  run/replicate/steal, or fix/patch/deploy/rollback) and no "then exploit/steal/fix":
  "For service X, inspect the SOURCE and report the vulnerability/attack surface for
  ONLY <named bug/class if any>. SOURCE-ONLY unless they said from/live/observed
  traffic. Do NOT build, test, push, start, or patch."
- "another exploit" / "more flags" / no service named: tell exploit to PREFER a flag
  store not already covered (it sees coverage in farm_services); cover all stores
  before doubling up (extra per store are backups).
- lifecycle (no rewrite, no test): "Stop the running exploit <name>" / "List the
  exploits currently running". A BULK request (start/stop ALL, or several) is ONE
  classification for the whole batch (the agent enumerates), never one per exploit.
  If the operator names an existing exploit to start/run, use that exact name and do
  not rewrite unless asked to fix.

patch -> minimal source fix, deploy, rollback. Route here ONLY for fix/patch/deploy/
rollback.
- "Patch ONLY <named bug/class/path if any> in service X with a minimal source fix;
  deploy through HITL if requested."

traffic -> recent Janus traffic reports, Janus alert/drop rules, and Janus service
inventory/status. Route here ONLY for a standalone traffic report, a rule/block/alert,
or an inventory question.
- report/rule: "Inspect only recent traffic for service X; report endpoint/parameter/
  payload evidence, or create the requested Janus rule via HITL."
- inventory: "List the services Janus currently proxies (name, id, ports, enabled) via
  list_services and report which are up." No packet reads. If the request is ONLY
  inventory/status (list/how many/names/ports/which are up), route it to traffic and
  NOTHING else -> never to patch (patch knows only the source it has, not the live
  proxy inventory), and add no packet-analysis scope the operator did not ask for.

ALL-SERVICES FAN-OUT: if an exploit or patch request targets every/the services
collectively and names no specific service ("analyze the services", "find the vulns",
"patch all of them"), emit ONE classification with fan_out_all=true and a
service-agnostic `query` (do NOT name a service); the router replicates it per live
service. A standalone traffic report or inventory question is NOT fanned out (one task
already spans every service) -> leave fan_out_all unset.
"""


def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to sub-agents based on the classifications produced by classify.

    Each sub-agent is a subgraph node on the shared `messages` channel, so we
    hand it ONLY its scoped sub-question as the input message; its own reasoning
    and tool calls merge back into the shared transcript (visible in the UI)."""
    return [
        Send(c["source"], {"messages": [HumanMessage(content=c["query"])]})
        for c in state["classifications"]
    ]


async def _service_names(registry) -> list[str]:
    """Live service names from Janus (the traffic view's list_services). Returns []
    on any failure so a fan-out task degrades to one un-expanded task, never crashes."""
    try:
        tool = next(t for t in registry.for_agent("traffic") if t.name == "list_services")
        # Invoke as a tool-call so we get the structured `artifact` (list[dict]);
        # fall back to parsing the string content as JSON.
        msg = await tool.ainvoke({"name": "list_services", "args": {}, "id": "ls", "type": "tool_call"})
        rows = getattr(msg, "artifact", None)
        if rows is None:
            rows = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
    except Exception:
        log.warning("classify: list_services failed; not fanning out", exc_info=True)
        return []
    return [str(r.get("name") or r.get("id")) for r in rows or []
            if isinstance(r, dict) and (r.get("name") or r.get("id"))]


def make_classify_node(structured_llm, registry=None):
    """Build the ``classify`` node: route the query into sub-questions, expanding any
    fan_out_all task into one per live Janus service."""

    async def classify_query(state: RouterState) -> dict:
        result = await structured_llm.ainvoke([
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": state["query"]},
        ])
        out: list[Classification] = []
        names: list[str] | None = None  
        for c in result.classifications:
            if c.get("fan_out_all"):
                if names is None:
                    names = await _service_names(registry) if registry else []
                if names:
                    out.extend(
                        {"source": c["source"], "query": f"For service '{n}': {c['query']}"}
                        for n in names
                    )
                    continue
            out.append({"source": c["source"], "query": c["query"]})
        return {"classifications": out}

    return classify_query


SYNTHESIS_PROMPT = (
    "You write the operator's FINAL answer in an Attack & Defense CTF, from the "
    "specialist reports below.\n"
    "- Open with a one-line combined verdict, then synthesize ACROSS specialists and "
    "cross-reference their findings (e.g. which observed attack maps to which source "
    "vulnerability).\n"
    "- Be concise; do not repeat the reports verbatim (the operator can already see "
    "them).\n"
    "- Relay status honestly: HITL actions (drop/both rules, deploy/rollback, push/"
    "start) pause for approval, so if one was rejected, report it and stop. Never "
    "upgrade pending or failed status into success; claim an exploit LIVE/started or "
    "flags captured only if a report explicitly shows a successful test "
    "(flags_found>=1) plus a successful push/start."
)

_SPECIALISTS = ("traffic", "patch", "exploit")


def _message_text(msg: AIMessage) -> str:
    """Flatten an AIMessage's content (string or content blocks) to plain text."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = [
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p)


def make_final_node(synthesis_llm):
    """Build the ``final`` node: combine the specialists' reports into one
    operator-facing answer. Specialists fan in HERE (not the supervisor), so the
    node builds its own System+Human prompt and the model replies to a user turn,
    writing a fresh combined answer instead of prefill-continuing the last report.
    The reply is appended to the shared ``messages`` so the UI renders it."""

    async def synthesize(state: RouterState) -> dict:
        wanted = {c["source"] for c in state.get("classifications") or []} or set(
            _SPECIALISTS
        )

        # A fanned-out task yields several reports from the same specialist name, so
        # collect each run's FINAL message (no tool calls = its report), not just the
        # last one on the channel; this keeps one report per service.
        reports = [
            (msg.name, _message_text(msg))
            for msg in state["messages"]
            if isinstance(msg, AIMessage) and msg.name in wanted and not msg.tool_calls
        ]
        request = state.get("query") or "the operator's request"
        blocks = "\n\n".join(
            f"## {name} specialist report\n{text}" for name, text in reports if text
        )
        human = (
            f"Operator request:\n{request}\n\n{blocks}\n\n"
            "Write the final answer for the operator now."
        )
        result = await synthesis_llm.ainvoke([
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content=human),
        ])
        return {"messages": [AIMessage(content=result.content, name="supervisor")]}

    return synthesize
