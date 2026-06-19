from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.conversational import build_supervisor_agent
from src.agents.orchestrator import (
    ClassificationResult,
    build_orchestrator_llm,
    build_synthesis_llm,
    make_classify_node,
    make_final_node,
    route_to_agents,
)
from src.agents.exploit_agent import build_exploit_agent
from src.agents.patch_agent import build_patch_agent
from src.agents.traffic_agent import build_traffic_agent
from src.state.state import RouterState
from src.tools.mcp_client import (
    build_exploiter_registry,
    build_patcher_read_registry,
    build_patcher_registry,
    build_registry,
)


async def make_graph():
    """Factory entrypoint for the Agent Server (``langgraph.json``).

    One graph on a shared ``messages`` channel: the supervisor chats with the
    operator and, on a security goal, hands off (Command(goto="classify")) to the
    classify -> fan-out -> specialists pipeline; the specialists run as subgraph
    nodes whose reasoning/tool calls stream into the shared transcript (so the UI
    renders them), then fan in to the ``final`` node which synthesizes their
    reports into the operator-facing reply. ``checkpointer=None``: the Agent Server
    injects persistence, so HITL interrupts raised by the specialists' middleware
    bubble natively to the server checkpointer that pauses/resumes the run.
    """
    structured_llm = build_orchestrator_llm().with_structured_output(ClassificationResult)
    synthesis_llm = build_synthesis_llm()

    registry = await build_registry()
    patcher_registry = await build_patcher_registry()
    exploiter_registry = await build_exploiter_registry()
    patcher_read_registry = await build_patcher_read_registry()

    traffic_agent = await build_traffic_agent(registry, patcher_read_registry)
    patch_agent = await build_patch_agent(patcher_registry)
    exploit_agent = await build_exploit_agent(
        registry, exploiter_registry, patcher_read_registry
    )

    workflow = StateGraph(RouterState)
    # `destinations` is a render-only hint: the supervisor->classify handoff is a
    # runtime Command(goto="classify", graph=PARENT) from the dispatch tool, invisible
    # to the static graph, so without this Studio draws classify/specialists/final as
    # orphan nodes. It does NOT add control edges or change routing.
    workflow.add_node(
        "supervisor", build_supervisor_agent(), destinations=("classify", END)
    )
    workflow.add_node("classify", make_classify_node(structured_llm, registry))
    workflow.add_node("traffic", traffic_agent)
    workflow.add_node("patch", patch_agent)
    # The exploit agent tends to need a longer ReAct budget for its test->edit loop.
    workflow.add_node("exploit", exploit_agent.with_config(recursion_limit=60))
    workflow.add_node("final", make_final_node(synthesis_llm))

    workflow.add_edge(START, "supervisor")
    # Default: the supervisor replied to the operator and we are done. The
    # dispatch tool overrides this with Command(goto="classify") when handing off.
    workflow.add_edge("supervisor", END)
    # classify fans out to ALL classified specialists IN PARALLEL (one Send each,
    # same superstep), or straight to `final` when there are none. The specialists
    # run concurrently -- their reasoning/tool calls stream interleaved into the
    # shared transcript, and each may open its own HITL gate (resumed by interrupt
    # id, independently or together) without blocking the others.
    workflow.add_conditional_edges(
        "classify", route_to_agents, ["traffic", "patch", "exploit", "final"]
    )
    # Specialists fan in to `final`, which waits for every branch (incl. all HITL
    # gates resolved) and synthesizes ONE operator-facing reply.
    workflow.add_edge("traffic", "final")
    workflow.add_edge("patch", "final")
    workflow.add_edge("exploit", "final")
    workflow.add_edge("final", END)

    return workflow.compile()
