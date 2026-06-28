from __future__ import annotations

from langchain_core.messages import HumanMessage

from graph.schema import State, RouterDecision
from graph.llm import llm
from prompts.router_prompt import get_router_prompt


def router_node(state: State) -> dict:
    prompt = get_router_prompt()
    chain = prompt | llm.with_structured_output(RouterDecision)

    decision = chain.invoke({"topic": state["topic"]})

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"