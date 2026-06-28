from __future__ import annotations

import json
from langchain_core.messages import HumanMessage
from graph.schema import State
from graph.llm import llm
from prompts.router_prompt import get_router_prompt


def router_node(state: State) -> dict:
    prompt = get_router_prompt()
    formatted = prompt.format(topic=state["topic"])
    response = llm.invoke([HumanMessage(content=formatted)])
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
            data = {"mode": "closed_book", "needs_research": False, "queries": []}
    else:
        try:
            data = json.loads(raw)
        except Exception:
            data = {"mode": "closed_book", "needs_research": False, "queries": []}

    valid_modes = {"closed_book", "hybrid", "open_book"}
    if data.get("mode") not in valid_modes:
        data["mode"] = "closed_book"

    return {
        "needs_research": bool(data.get("needs_research", False)),
        "mode": data["mode"],
        "queries": data.get("queries", []),
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"