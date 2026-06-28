from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from graph.schema import State
from routers.router import router_node, route_next
from agents.researcher import research_node
from agents.orchestrator import orchestrator_node, fanout
from agents.worker import worker_node
from agents.reducer import reducer_subgraph

g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {
    "research": "research",
    "orchestrator": "orchestrator",
})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()