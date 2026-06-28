from __future__ import annotations

import json
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send

from graph.schema import State, Plan, Task
from graph.llm import llm

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a blog outline. Return ONLY valid JSON, no markdown, no backticks.

JSON format:
{
  "blog_title": "string",
  "audience": "string",
  "tone": "string",
  "blog_kind": "explainer|tutorial|news_roundup|comparison|system_design",
  "constraints": [],
  "tasks": [
    {
      "id": 1,
      "title": "string",
      "goal": "one sentence",
      "bullets": ["bullet1", "bullet2", "bullet3"],
      "target_words": 250,
      "tags": [],
      "requires_research": false,
      "requires_citations": false,
      "requires_code": false
    }
  ]
}

Rules:
- 5-9 tasks total
- Each task: MINIMUM 3 bullets, maximum 6, target_words 120-550
- At least 1 task with requires_code=true
- bullets must be actionable (build/compare/measure/debug)
- blog_kind must be one of the exact values listed above
"""


def orchestrator_node(state: State) -> dict:
    evidence = state.get("evidence") or []
    mode = state.get("mode", "closed_book")

    evidence_text = "\n".join(
        f"- {e.title} | {e.url}" for e in evidence[:16]
    ) if evidence else "No evidence."

    response = llm.invoke([
        SystemMessage(content=ORCH_SYSTEM),
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