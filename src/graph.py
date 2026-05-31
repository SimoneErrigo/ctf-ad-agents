from __future__ import annotations

from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp, GraphRecursionError
from langgraph.graph import END, START, StateGraph

from src.agents.conversational import build_conversational_agent
from src.agents.orchestrator import (
    ClassificationResult,
    build_orchestrator_llm,
    make_classify_node,
    make_synthesize_node,
    route_to_agents,
)
from src.agents.exploit_agent import build_exploit_agent
from src.agents.patch_agent import build_patch_agent
from src.agents.traffic_agent import build_traffic_agent
from src.state.state import AgentInput, RouterState
from src.tools.mcp_client import (
    build_exploiter_registry,
    build_patcher_read_registry,
    build_patcher_registry,
    build_registry,
)


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


def make_patch_node(patch_agent):
    """Build the ``patch`` node: adapt the patch sub-agent to graph state.

    HITL (interrupt) raised by the patch agent's deploy/rollback tools
    propagates up through this node to the conversational agent's
    checkpointer, which is what actually pauses the run. The subgraph itself
    is stateless on purpose — the conversational layer owns persistence.
    """

    async def query_patch_agent(state: AgentInput) -> dict:
        result = await patch_agent.ainvoke({
            "messages": [{"role": "user", "content": state["query"]}]
        })
        return {
            "results": [
                {"source": "patch", "result": result["messages"][-1].content}
            ]
        }

    return query_patch_agent


def make_exploit_node(exploit_agent):
    """Build the ``exploit`` node: adapt the exploit sub-agent to graph state.

    Like patch, this subgraph is stateless; HITL (push/start approvals) raised
    by the exploit agent's tools propagates up to the conversational layer's
    checkpointer, which is what pauses the run.
    """

    async def query_exploit_agent(state: AgentInput) -> dict:
        result = await exploit_agent.ainvoke(
            {"messages": [{"role": "user", "content": state["query"]}]},
            {"recursion_limit": 60},
        )
        return {
            "results": [
                {"source": "exploit", "result": result["messages"][-1].content}
            ]
        }

    return query_exploit_agent


async def build_app():
    """Build the compiled LangGraph app. Call once at startup."""
    orchestrator_llm = build_orchestrator_llm()
    structured_llm = orchestrator_llm.with_structured_output(ClassificationResult)

    registry = await build_registry()
    patcher_registry = await build_patcher_registry()
    exploiter_registry = await build_exploiter_registry()
    patcher_read_registry = await build_patcher_read_registry()
    traffic_agent = await build_traffic_agent(registry)
    patch_agent = await build_patch_agent(patcher_registry)
    exploit_agent = await build_exploit_agent(
        registry, exploiter_registry, patcher_read_registry
    )

    workflow = StateGraph(RouterState)
    workflow.add_node("classify", make_classify_node(structured_llm))
    workflow.add_node("traffic", make_traffic_node(traffic_agent))
    workflow.add_node("patch", make_patch_node(patch_agent))
    workflow.add_node("exploit", make_exploit_node(exploit_agent))
    workflow.add_node("synthesize", make_synthesize_node(orchestrator_llm))

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify", route_to_agents, ["traffic", "patch", "exploit"]
    )
    workflow.add_edge("traffic", "synthesize")
    workflow.add_edge("patch", "synthesize")
    workflow.add_edge("exploit", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def make_security_analysis_tool(inner_app):
    """Wrap the stateless orchestrator workflow as a single agent tool.

    The conversational layer calls this tool for every domain question
    (traffic analysis, alert/drop rule creation, source-level patching). The
    orchestrator inside picks the right sub-agent(s) and may run several in
    parallel.
    """

    @tool
    async def security_analysis(query: str) -> str:
        """Investigate captured CTF traffic, create Janus alert/drop rules, patch
        a vulnerable service, or build/run/stop an exploit.

        Use this tool for ANY question about: ongoing attacks against a named
        service, suspicious requests / exploits, writing Janus alert or drop
        rules, fixing a vulnerability in the source code, rolling back a
        previous patch, writing/replicating/running an exploit on the
        exploitfarm, STOPPING a running exploit, or checking which exploits are
        currently running. The tool internally routes to the traffic agent, the
        patch agent, the exploit agent, or several in parallel. Critical actions
        (drop rules, code deploy, rollback, exploit push/start) will pause for
        operator approval (human-in-the-loop), when that happens you will see a
        HITL prompt surfaced to the user. Stopping or listing exploits is NOT
        gated and runs immediately.

        IMPORTANT: make ONE call, not several, for a single goal. The exploit
        agent finds the vulnerability in the source ITSELF, so for a request like
        "find a vulnerability and write/test/start an exploit" issue a SINGLE
        call describing the whole exploit goal, do NOT make a separate "find
        vulnerability" call first (that re-reads the source twice and is slow).
        If the user names a bug/class and does not explicitly say from/live/
        observed traffic, put SOURCE-ONLY / do not inspect traffic in the query.
        """
        try:
            result = await inner_app.ainvoke({"query": query})
            return result["final_answer"]
        except GraphBubbleUp:
            # HITL interrupts (and other LangGraph control-flow signals) MUST
            # propagate: the ToolNode re-raises GraphBubbleUp so the interrupt
            # reaches the conversational agent's checkpointer, which pauses the
            # run. On Command(resume=...) the inner graph (sharing the inherited
            # checkpointer + nested checkpoint_ns) resumes inside the pending
            # interrupt instead of restarting. Swallowing it here breaks HITL.
            raise
        except GraphRecursionError:
            # A sub-agent ran out of steps without reaching a stop condition
            # (e.g. the exploit agent churning test->edit->test). This is a
            # terminal failure for THIS goal, instruct the caller NOT to retry,
            # otherwise the conversational agent loops the whole analysis again
            # and multiplies cost. Surfaced as a normal tool result so the
            # checkpoint has no dangling tool_use.
            return (
                "[security_analysis failed: the specialist agent exhausted its "
                "step budget without producing a working result (likely an "
                "exploit that could not be reproduced). Do NOT retry this query "
                "automatically. Report the failure to the operator and ask how to "
                "proceed (e.g. a different vulnerability, or more recon).]"
            )
        except Exception as e:
            # Return real errors as a string so LangGraph always stores a
            # tool_result and the checkpoint never has a dangling tool_use.
            return f"[security_analysis error: {type(e).__name__}: {e}]"

    return security_analysis


async def make_graph():
    """Factory entrypoint for the Agent Server (``langgraph.json``).

    Wraps the stateless orchestrator graph as the ``security_analysis`` tool and
    exposes it through the conversational agent. We pass ``checkpointer=None``:
    the Agent Server injects persistence itself (Postgres in a deployment,
    in-memory under ``langgraph dev``), and a custom checkpointer here would be
    ignored, so the platform owns threads/checkpoints. Point the server's
    ``DATABASE_URI`` at Postgres to persist them.

    HITL interrupts raised inside the orchestrator subgraph (drop-rule, patch
    deploy/rollback, exploit push/start) propagate up to the server-managed
    checkpointer, which pauses the run and lets it resume via
    ``Command(resume=<HumanResponse>)``.
    """
    inner_app = await build_app()
    security_tool = make_security_analysis_tool(inner_app)
    return build_conversational_agent([security_tool], checkpointer=None)
