from __future__ import annotations

import os

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.state.state import Classification, RouterState


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
    )


CLASSIFY_PROMPT = """Route the operator request to the smallest set of agents.

Agents:
- traffic: recent Janus traffic reports, Janus service inventory/status (which
  services are up/proxied and their ports), and Janus alert/drop rules.
- patch: source fix, deploy, rollback.
- exploit: write/test/push/start/STOP a Python exploit via xfarm, and report
  which exploits are currently running.

Hard routing rules:
- If the user asks to write/create/test/push/start/run/replicate/steal an
  exploit, include exploit. Do NOT also include traffic just to feed exploit.
- If the user asks for "another exploit" / "more flags" / does not name a
  service, tell exploit to PREFER a FLAG STORE (farm service) not already
  covered (it sees per-service coverage in farm_services), cover all stores
  before doubling up. Extra exploits per store are fine only as backups.
- If the user asks to STOP/kill a running exploit, or asks which exploits are
  running/active, include exploit (lifecycle only, do NOT rewrite/test).
- For a BULK lifecycle request (start/run/stop ALL exploits, or several at once),
  emit ONE exploit classification for the whole batch, never one per exploit; the
  exploit agent enumerates them and acts on each.
- If the user names an existing exploit and asks to start/run it, tell exploit
  to use that exact existing name and not rewrite/replace files unless asked to fix.
- Include traffic only for a standalone traffic report, a Janus rule/block/alert,
  or a Janus service inventory/status question (which services are up/proxied,
  list the services, their ports). This is read from Janus via list_services.
- Include patch only for fix/patch/deploy/rollback.
- Preserve scope in every sub-question: service name, any named bug/class phrase,
  source vs traffic wording, and requested actions.
- If a bug/class is named, whatever it is, the exploit/patch sub-question must
  say ONLY that named scope; no fallback bugs.
- If a bug/class is named and the user did not explicitly say from/live/observed
  traffic, the exploit/patch sub-question must say SOURCE-ONLY and DO NOT inspect
  Janus traffic. Generic words like "find" do not mean traffic.
- "find the bug/attack then exploit it" is exploit only unless the user also asks
  for a separate report, rule, or patch.
- ANALYSIS-ONLY: if the user only asks to find/analyze/report/audit/describe/list
  a vulnerability or attack and uses NO action verb (write/create/test/push/start/
  run/replicate/steal an exploit, or fix/patch/deploy/rollback), route to exploit
  in REPORT-ONLY mode: inspect the source and report the vulnerability, but DO NOT
  build/test/push/start any exploit and DO NOT patch. "find the bug" WITHOUT an
  explicit "then exploit/steal/fix it" is report-only, never a build task.

Sub-question shape:
- exploit (build): "For service X, build ONE Python exploit for ONLY <named bug/class if any>.
  SOURCE-ONLY and do not inspect Janus traffic unless the user said from/live/
  observed traffic. Test once, then push/start through HITL only after flags_found>=1."
- exploit (report-only): "For service X, inspect the SOURCE and report the
  vulnerability / attack surface for ONLY <named bug/class if any>. SOURCE-ONLY,
  do not inspect Janus traffic unless the user said from/live/observed traffic.
  DO NOT build, test, push, or start any exploit; report findings only."
- exploit (lifecycle): "Stop the running exploit <name>" or "List the exploits
  currently running", no rewriting, no testing, just the lifecycle action.
- exploit (bulk lifecycle): "Start ALL available registered exploits" / "Stop ALL
  running exploits" as ONE task; the agent lists them and acts on each.
- traffic: "Inspect only recent traffic for service X; report endpoint/parameter/
  payload evidence, or create the requested Janus rule via HITL."
- traffic (inventory): "List the services Janus currently proxies (name, id, ports,
  enabled) via list_services and report which are up." No traffic packet reads.
- patch: "Patch ONLY <named bug/class/path if any> in service X with a minimal source fix;
  deploy through HITL if requested."
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


def make_classify_node(structured_llm):
    """Build the ``classify`` node: route the user query into sub-questions."""

    async def classify_query(state: RouterState) -> dict:
        result = await structured_llm.ainvoke([
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": state["query"]},
        ])
        return {"classifications": result.classifications}

    return classify_query


SYNTHESIS_PROMPT = (
    "You write the FINAL answer to the operator of an Attack & Defense CTF, from "
    "the specialist reports below. Open with a one-line combined verdict, then "
    "synthesize ACROSS all specialists and cross-reference their findings (e.g. "
    "which observed attack maps to which source vulnerability). Be concise and do "
    "NOT repeat the reports verbatim, the operator can already see them. Relay "
    "results honestly: HITL actions (drop/both rules, deploy/rollback, "
    "push_exploit, start_exploit) pause for approval, if one was rejected report "
    "it and stop; never upgrade pending or failed status into success; for "
    "exploits claim LIVE/started/flags only if a report explicitly shows a "
    "successful test (flags_found>=1) plus successful push/start."
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
        
        reports: dict[str, str] = {}
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.name in wanted:
                text = _message_text(msg)
                if text:
                    reports[msg.name] = text
        request = state.get("query") or "the operator's request"
        blocks = "\n\n".join(
            f"## {name} specialist report\n{text}" for name, text in reports.items()
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
