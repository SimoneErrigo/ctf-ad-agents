from typing import Literal, NotRequired, TypedDict

from langgraph.graph import MessagesState


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""
    source: Literal["traffic", "patch", "exploit"]
    query: str
    # Set when the task targets ALL services: classify expands it into one task per
    # live service. The `query` is then service-agnostic; classify prepends each.
    fan_out_all: NotRequired[bool]


class RouterState(MessagesState):
    """Shared state for the whole graph. `messages` (from MessagesState) is the
    single conversation channel: the supervisor, the sub-agents and their tool
    calls all append to it, so the UI renders every agent's reasoning. `query`
    holds the supervisor's dispatched task; `classifications` the routing fan-out
    that `route_to_agents` turns into one parallel Send per specialist."""
    query: str
    classifications: list[Classification]
