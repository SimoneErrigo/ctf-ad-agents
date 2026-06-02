from typing import Literal, TypedDict

from langgraph.graph import MessagesState


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""
    source: Literal["traffic", "patch", "exploit"]
    query: str


class RouterState(MessagesState):
    """Shared state for the whole graph. `messages` (from MessagesState) is the
    single conversation channel: the supervisor, the sub-agents and their tool
    calls all append to it, so the UI renders every agent's reasoning. `query`
    holds the supervisor's dispatched task; `classifications` the routing fan-out."""
    query: str
    classifications: list[Classification]
