
from __future__ import annotations
 
from graph.schema import Task, Plan, EvidenceItem
from graph.llm import llm
from prompts.worker_prompt import get_worker_prompt
import time


# worker ke pass sara data aayega task ke liye payload mai 
def worker_node(payload:dict)->dict:
    time.sleep(2)
    task = Task(**payload['task'])
    plan = Plan(**payload['plan'])
    evidence = [EvidenceItem(**e) for e in payload.get('evidence', [])]
    mode = payload.get('mode', 'closed_book')
        # Build evidence text for prompt
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    ) if evidence else "No evidence provided."

    #combine task details into topic content
    task_content = (
        f"Section title: {task.title}\n"
        f"Goal: {task.goal}\n"
        f"Target words: {task.target_words}\n"
        f"Bullets:\n- " + "\n- ".join(task.bullets) + "\n"
        f"requires_code: {task.requires_code}\n"
        f"requires_citations: {task.requires_citations}\n"
        f"Blog title: {plan.blog_title}\n"
        f"Audience: {plan.audience}\n"
        f"Tone: {plan.tone}\n"
    )
    prompt = get_worker_prompt()
    chain = prompt | llm
    section = chain.invoke({"topic": task_content, "mode": mode, "research_results": evidence_text}).content.strip()

    return {"section":[(task.id,section)]}


