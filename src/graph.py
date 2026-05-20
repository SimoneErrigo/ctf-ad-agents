from __future__ import annotations

from contextlib import asynccontextmanager

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from src.agents.conversational import build_conversational_agent
from src.agents.orchestrator import (
    ClassificationResult,
    build_orchestrator_llm,
    make_classify_node,
    make_synthesize_node,
    route_to_agents,
)
from src.agents.traffic_agent import build_traffic_agent
from src.persistence import postgres_checkpointer
from src.state.state import AgentInput, RouterState
from src.tools.mcp_client import build_registry


def make_traffic_node(traffic_agent):
    """Build the ``traffic`` node: adapt the traffic sub-agent to graph state."""

    async def query_traffic_agent(state: AgentInput) -> dict:
        result = await traffic_agent.ainvoke({
            "messages": [{"role": "user", "content": state["query"]}]
        })
        return {
            "results": [
                {"source": "traffic", "result": result["messages"][-1].content}
            ]
        }

    return query_traffic_agent


async def build_app():
    """Build the compiled LangGraph app. Call once at startup."""
    orchestrator_llm = build_orchestrator_llm()
    structured_llm = orchestrator_llm.with_structured_output(ClassificationResult)

    registry = await build_registry()
    traffic_agent = await build_traffic_agent(registry)

    workflow = StateGraph(RouterState)
    workflow.add_node("classify", make_classify_node(structured_llm))
    workflow.add_node("traffic", make_traffic_node(traffic_agent))
    workflow.add_node("synthesize", make_synthesize_node(orchestrator_llm))

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges("classify", route_to_agents, ["traffic"])
    workflow.add_edge("traffic", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def make_security_analysis_tool(inner_app):
    """Wrap the stateless orchestrator workflow as a single agent tool."""

    @tool
    async def security_analysis(query: str) -> str:
        """Investigate captured CTF traffic for attacks against a service.

        Use this for any question about attacks, exploits, suspicious
        requests, or the security state of a named service. Runs a full
        traffic analysis and returns a written report.
        """
        result = await inner_app.ainvoke({"query": query})
        return result["final_answer"]

    return security_analysis


@asynccontextmanager
async def build_chat_app():
    """Build the stateful conversational agent.

    Wraps the stateless orchestrator graph as a tool and exposes it through
    a conversational agent persisted in Postgres. Use as an async context
    manager so the checkpointer's connection pool is opened/closed correctly:

        async with build_chat_app() as agent:
            cfg = {"configurable": {"thread_id": "..."}}
            await agent.ainvoke({"messages": [...]}, cfg)
    """
    inner_app = await build_app()
    security_tool = make_security_analysis_tool(inner_app)
    async with postgres_checkpointer() as checkpointer:
        yield build_conversational_agent([security_tool], checkpointer)
