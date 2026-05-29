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
- exploit: Writes a Python exploit and runs it on the team's exploitfarm via
  xfarm. It is self-sufficient: it reads the attack from captured traffic and
  the service source on its own. Use for: "write/create an exploit for service
  X", "replicate the attack they are running on us", "steal that exploit from
  the traffic", "run/launch the exploit". It tests against the NOP team first;
  pushing source and launching against all teams require operator approval (HITL).

Routing rules:
- Pick the SMALLEST set of agents that can answer. Most queries are single-agent.
- Send to BOTH when the user explicitly asks for traffic analysis AND a patch
  (e.g. "find the attack and patch it") OR asks for a live Janus rule AND a
  source patch in the same request (e.g. "find SQLi, make a drop rule, then
  patch it"). In that case the traffic sub-question must BOTH produce concrete
  evidence (service/endpoint/parameter/payload) and, when the user asked for a
  rule, create or update the requested Janus rule through the HITL tool.
- Route to `exploit` whenever the user asks to write, create, replicate, steal,
  or run an exploit. The exploit agent reads traffic itself, so you do NOT need
  to also route to `traffic` just to feed it, only add `traffic` if the user
  separately wants a traffic report or a Janus rule. You MAY combine `exploit`
  with `patch` (e.g. "patch CC-Forms and also build an exploit for the same
  bug"); both run in parallel and each inspects what it needs on its own.
- INSTRUMENTAL vs DELIVERABLE traffic: when the user says "find / identify /
  look at the attack (in the traffic) AND (then) build / write / replicate an
  exploit", the "find" is INSTRUMENTAL to the exploit, route to `exploit` ONLY.
  The exploit agent reads the traffic itself to recover the attack. Add
  `traffic` as a SECOND source only when the user asks for a STANDALONE traffic
  deliverable in its own right (a written traffic report for the operator, or a
  Janus alert/drop rule), not merely as a step toward the exploit.
- INSTRUMENTAL vs DELIVERABLE source: same rule for the code. "find / identify a
  vulnerability in the source AND (then) write / test / start an exploit" → the
  "find" is INSTRUMENTAL, route to `exploit` ONLY (it reads the source itself to
  locate the bug). Add `patch` as a SECOND source only when the user separately
  wants the bug FIXED / deployed, not merely identified for the exploit.
- Do NOT route a pure "explain / summarize" question to the patch or exploit agent.
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

The exploit agent's sub-question MUST name the target service and PRESERVE the
operator's scope: if the user said "in/from the source code", tell it to find
the bug by reading the source and NOT to inspect traffic; only when the user
referenced an ongoing/observed attack (or said "from traffic") tell it to
recover that attack from captured traffic and replicate it. Tell it to write
ONE exploit for ONE vulnerability, test it against the NOP team first, and go
through the HITL gate before pushing / starting against all teams.

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
- User: "Are they attacking CC-Forms? If so, patch it and build an exploit that
  replicates their attack."
  - patch: "Patch the vulnerability being exploited in service 'CC-Forms'.
    Identify the bug from the source, propose a minimal fix, deploy via HITL."
  - exploit: "Write an exploit for service 'CC-Forms' that replicates the attack
    currently hitting it: find the malicious request in recent captured traffic,
    reproduce it, test against the NOP team, then push/start via HITL."
- User: "Write an exploit for the RCEaas service."
  - exploit: "Write a Python exploit for service 'RCEaas': read its source to
    find an exploitable bug (or recover an attack from traffic), implement it,
    test against the NOP team, and go through HITL before pushing/starting."
- User: "Find the exploit they are launching on service X, then analyze the code
  and build an exploit for X." (the "find" is instrumental, exploit only)
  - exploit: "Write an exploit for service 'X' replicating the live attack:
    locate the malicious request in recent captured traffic and recover its
    exact bytes, read the service source to understand the bug (e.g. a crypto
    flaw that lets attackers forge tokens, which is not visible from traffic
    alone), implement it, test against the NOP team, then push/start via HITL."
- User: "Give me a report of what's hitting service X AND build an exploit that
  replicates it." (standalone traffic deliverable + exploit -> both sources)
  - traffic: "Report the attack(s) hitting service 'X' from recent traffic:
    endpoint, parameter, sample payload, and evidence; stop once confirmed."
  - exploit: "Write an exploit for service 'X' replicating the live attack:
    recover it from captured traffic, read the source as needed, test against
    the NOP team, then push/start via HITL."
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
