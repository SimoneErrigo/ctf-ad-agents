from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.orchestrator import (
    ClassificationResult,
    build_orchestrator_llm,
    make_classify_node,
    make_synthesize_node,
    route_to_agents,
)
from src.agents.traffic_agent import build_traffic_agent
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
