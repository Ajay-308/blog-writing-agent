from __future__ import annotations

from typing import List
import os

from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

from graph.schema import State, EvidencePack, EvidenceItem
from graph.llm import llm
from prompts.research_prompt import get_research_prompt


def tavily_search_node(query: str, max_results: int = 5) -> list[dict]:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    # seedha client.search use karo - TavilySearchResults deprecated hai
    response = client.search(query=query, max_results=max_results)
    results = response.get("results", [])
    
    normalized_results = []
    for r in results:
        normalized_results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "") or r.get("snippet", ""),  # 'content' key hai Tavily mein
            "published_at": r.get("published_date") or r.get("published_at"),
            "source": r.get("source"),
        })
    return normalized_results


def research_node(state: State) -> dict:
    queries = state.get("queries") or []
    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(tavily_search_node(q, max_results=5))

    if not raw_results:
        return {"evidence": []}

    prompt = get_research_prompt()
    chain = prompt | llm.with_structured_output(EvidencePack)

    # research_prompt mein sirf {topic} variable hai, isliye results ko saath merge karo
    pack = chain.invoke({
        "topic": f"Topic: {state['topic']}\n\nSearch Results:\n{raw_results}"
    })

    dedup = {e.url: e for e in pack.evidence if e.url}
    return {"evidence": list(dedup.values())}