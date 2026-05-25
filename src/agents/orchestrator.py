from __future__ import annotations

import os

from langchain_aws import ChatBedrockConverse
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.state.state import Classification, RouterState


class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""

    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )


def build_orchestrator_llm() -> ChatBedrockConverse:
    """LLM powering the orchestrator (classification + multi-source synthesis)."""
    return ChatBedrockConverse(
        name="orchestrator-llm",
        model=os.environ["ROUTER_AGENT_MODEL"],
        region_name=os.environ["REGION_NAME"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        temperature=0.1,
    )


CLASSIFY_PROMPT = """Analyze this query and determine which agent(s) to consult.
For each relevant agent, generate a targeted sub-question optimized for that agent.

Available agents:
- traffic: Analyzes network traffic captured by Janus and, when explicitly
  asked, can write Janus alert/drop rules. Use for: "are we under attack?",
  "what is hitting service X?", "show me the exploit", "create an alert rule
  for that pattern", "block this request". Drop rules require operator
  approval (HITL).
- patch: Reads service source code and produces a minimal source-level patch,
  then deploys it to the competition VM via git. Use for: "patch this bug",
  "fix the vulnerability in service X", "deploy a fix that blocks this
  attack at the code level", "rollback the last patch". Every deploy /
  rollback requires operator approval (HITL).

Routing rules:
- Pick the SMALLEST set of agents that can answer. Most queries are single-agent.
- Send to BOTH when the user explicitly asks for traffic analysis AND a patch
  (e.g. "find the attack and patch it") OR asks for a live Janus rule AND a
  source patch in the same request (e.g. "find SQLi, make a drop rule, then
  patch it"). In that case the traffic sub-question must BOTH produce concrete
  evidence (service/endpoint/parameter/payload) and, when the user asked for a
  rule, create or update the requested Janus rule through the HITL tool.
- Do NOT route a pure "explain / summarize" question to the patch agent.
- Do NOT satisfy "make/create/update a drop rule" by merely reporting a rule
  from conversation memory. The traffic agent must verify current Janus rules
  in this run, and must call create_rule/update_rule if the requested live rule
  is absent or stale.

Return ONLY the sources that are relevant. Each source should have a targeted
sub-question optimized for that specific domain.

The traffic agent is time-constrained, so its sub-question MUST scope the work:
name the specific service if given, ask only about RECENT traffic, steer toward
flagged / flag-id-bearing requests, and tell it to stop as soon as the attack(s)
are identified with evidence. Keep it to one or two sentences.

The patch agent's sub-question MUST name the target service and, when
available, the attack technique or vulnerable code path the operator (or the
traffic agent) already identified. Tell it to propose a minimal patch and to
go through the HITL gate before deploying.

Examples:
- User: "Are there any attacks in the RCEaas service?"
  - traffic: "Which attacks target the RCEaas service? Inspect only recent
    RCEaas traffic, prioritising flagged or flag-id-bearing requests; identify
    the attack pattern with evidence and stop as soon as it is confirmed."
- User: "Patch the path-traversal bug in ccforms."
  - patch: "Patch the path-traversal vulnerability in service 'ccforms'.
    Locate the unsafe filename handling, propose a minimal fix that rejects
    '..' and absolute paths, then propose deploy via HITL."
- User: "Find the SQLi on web1 and fix it."
  - traffic: "Identify the SQLi pattern hitting service 'web1' from recent
    traffic; report the offending endpoint, parameter, and a sample payload."
  - patch: "Patch the SQLi in service 'web1'. Use the endpoint/parameter the
    traffic agent will report; prefer query parameterization; propose deploy
    via HITL."
- User: "Find SQLi, make a drop rule, and patch it."
  - traffic: "Identify the recent SQLi pattern, verify current Janus rules, and
    create a tight drop rule for the payload through HITL if one is not already
    active. Report service id, expression, and rule id."
  - patch: "Patch the SQLi in the affected service using the endpoint/parameter
    reported by traffic; propose deploy via HITL."
"""


def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to sub-agents based on the classifications produced by classify."""
    return [
        Send(c["source"], {"query": c["query"]})
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


def make_synthesize_node(llm):
    """Build the ``synthesize`` node: merge sub-agent results into a final answer."""

    async def synthesize_results(state: RouterState) -> dict:
        results = state["results"]
        if not results:
            return {"final_answer": "No results found from any knowledge source."}

        # Single source: the sub-agent already produced a full report.
        # Skip the extra synthesis LLM call (about 5s + a model round-trip).
        if len(results) == 1:
            return {"final_answer": results[0]["result"]}

        formatted = [
            f"**From {r['source'].title()}:**\n{r['result']}"
            for r in results
        ]
        synthesis = await llm.ainvoke([
            {
                "role": "system",
                "content": (
                    f"Synthesize these search results to answer the original "
                    f"question: \"{state['query']}\"\n\n"
                    "- Combine information from multiple sources without redundancy\n"
                    "- Highlight the most relevant and actionable information\n"
                    "- Note any discrepancies between sources\n"
                    "- Keep the response concise and well-organized"
                ),
            },
            {"role": "user", "content": "\n\n".join(formatted)},
        ])
        return {"final_answer": synthesis.content}

    return synthesize_results
