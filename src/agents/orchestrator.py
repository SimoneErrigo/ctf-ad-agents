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


CLASSIFY_PROMPT = """Analyze this query and determine which agent to consult.
For each relevant agent, generate a targeted sub-question optimized for that agent.

Available agents:
- traffic: Analyzes network traffic and identifies potential threats based on patterns in the data.

Return ONLY the sources that are relevant to the query. Each source should have
a targeted sub-question optimized for that specific knowledge domain.

The traffic agent is time-constrained, so the sub-question MUST scope the work:
name the specific service if given, ask only about RECENT traffic, steer toward
flagged / flag-id-bearing requests, and tell it to stop as soon as the attack(s)
are identified with evidence. Keep it to one or two sentences.

Example for "Are there any attacks in the RCEaas service?":
- traffic: "Which attacks target the RCEaas service? Inspect only recent RCEaas
  traffic, prioritising flagged or flag-id-bearing requests; identify the attack
  pattern with evidence and stop as soon as it is confirmed."
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
