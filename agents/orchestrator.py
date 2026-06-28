from __future__ import annotations

import json
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send

from graph.schema import State, Plan, Task
from graph.llm import llm
from prompts.orchstrator_prompt import get_orchestrator_prompt


def orchestrator_node(state: State) -> dict:
    evidence = state.get("evidence") or []
    mode = state.get("mode", "closed_book")

    evidence_text = "\n".join(
        f"- {e.title} | {e.url}" for e in evidence[:16]
    ) if evidence else "No evidence."

    response = llm.invoke([
        SystemMessage(content=get_orchestrator_prompt()),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n\n"
            f"Evidence:\n{evidence_text}"
        )),
    ])

    raw = response.content.strip()

    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                data = json.loads(part)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Orchestrator JSON parse failed:\n{raw}")
    else:
        data = json.loads(raw)

    valid_kinds = {"explainer", "tutorial", "news_roundup", "comparison", "system_design"}
    if data.get("blog_kind") not in valid_kinds:
        data["blog_kind"] = "explainer"

    # bullets fix — minimum 3 enforce karo
    for task in data.get("tasks", []):
        while len(task.get("bullets", [])) < 3:
            task["bullets"].append("Provide additional implementation details and examples.")

    plan = Plan(**data)
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