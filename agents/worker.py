from __future__ import annotations

from graph.schema import Task, Plan, EvidenceItem
from graph.llm import llm
from prompts.worker_prompt import get_worker_prompt
import time
from groq import RateLimitError


# def worker_node(payload: dict) -> dict:
#     time.sleep(2)
#     task = Task(**payload['task'])
#     plan = Plan(**payload['plan'])
#     evidence = [EvidenceItem(**e) for e in payload.get('evidence', [])]
#     mode = payload.get('mode', 'closed_book')

#     evidence_text = "\n".join(
#         f"- {e.title} | {e.url}\n  {e.snippet or ''}"
#         for e in evidence[:20]
#     ) if evidence else "No evidence provided."

#     prompt = get_worker_prompt()
#     chain = prompt | llm

#     section = chain.invoke({
#         "topic": payload["topic"],
#         "mode": mode,
#         "research_results": evidence_text,
#         "section_title": task.title,
#         "goal": task.goal,
#         "bullets": "\n".join(f"- {b}" for b in task.bullets),
#         "target_words": task.target_words,
#         "requires_code": task.requires_code,
#         "requires_citations": task.requires_citations,
#     }).content.strip()

#     return {"sections": [(task.id, section)]}


def worker_node(payload: dict) -> dict:
    time.sleep(5)  # 2 → 5 seconds, parallel workers throttle honge

    task = Task(**payload['task'])
    plan = Plan(**payload['plan'])
    evidence = [EvidenceItem(**e) for e in payload.get('evidence', [])]
    mode = payload.get('mode', 'closed_book')

    evidence_text = "\n".join(
        f"- {e.title} | {e.url}\n  {e.snippet or ''}"
        for e in evidence[:10]  # 20 → 10, tokens bachao
    ) if evidence else "No evidence provided."

    prompt = get_worker_prompt()
    chain = prompt | llm

    # Retry with backoff
    for attempt in range(5):
        try:
            result = chain.invoke({
                "topic": payload["topic"],
                "mode": mode,
                "research_results": evidence_text,
                "section_title": task.title,
                "goal": task.goal,
                "bullets": "\n".join(f"- {b}" for b in task.bullets),
                "target_words": min(task.target_words, 400),  # cap karo
                "requires_code": task.requires_code,
                "requires_citations": task.requires_citations,
            })
            return {"sections": [(task.id, result.content.strip())]}
        except RateLimitError:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
            print(f"Rate limit hit (worker). {wait}s baad retry #{attempt+1}...")
            time.sleep(wait)

    raise Exception("Worker: max retries exceeded")