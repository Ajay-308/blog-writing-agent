from __future__ import annotations

from graph.schema import Task, Plan, EvidenceItem
from graph.llm import llm
from prompts.worker_prompt import get_worker_prompt
import time


def worker_node(payload: dict) -> dict:
    time.sleep(2)
    task = Task(**payload['task'])
    plan = Plan(**payload['plan'])
    evidence = [EvidenceItem(**e) for e in payload.get('evidence', [])]
    mode = payload.get('mode', 'closed_book')

    evidence_text = "\n".join(
        f"- {e.title} | {e.url}\n  {e.snippet or ''}"
        for e in evidence[:20]
    ) if evidence else "No evidence provided."

    prompt = get_worker_prompt()
    chain = prompt | llm

    section = chain.invoke({
        "topic": payload["topic"],
        "mode": mode,
        "research_results": evidence_text,
        "section_title": task.title,
        "goal": task.goal,
        "bullets": "\n".join(f"- {b}" for b in task.bullets),
        "target_words": task.target_words,
        "requires_code": task.requires_code,
        "requires_citations": task.requires_citations,
    }).content.strip()

    return {"sections": [(task.id, section)]}