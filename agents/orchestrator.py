from __future__ import annotations
 
from langgraph.types import Send
 
from graph.schema import State, Plan
from graph.llm import llm
from prompts.orchstrator_prompt import get_orchstrator_prompt

def orchestrator_node(state:State)->dict:
    prompt = get_orchstrator_prompt()
    chain = prompt | llm.with_structured_output(Plan)
    evidence = state.get("evidence") or []
    topic_with_context = (
        f"Topic: {state['topic']}\n"
        f"Mode: {state.get('mode', 'closed_book')}\n\n"
        f"Evidence (use only for fresh claims):\n"
        f"{[e.model_dump() for e in evidence][:16]}"
    )
 
    plan = chain.invoke({"topic": topic_with_context})
    return {"plan": plan}

def fanout(state: State):
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence") or []],
        })
        for task in state["plan"].tasks
    ]