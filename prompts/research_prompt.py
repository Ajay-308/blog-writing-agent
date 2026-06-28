from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts import PromptTemplate


research_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are a research assistant for a technical blog planner. 
Your task is to generate high-signal search queries for a given topic.
Topic:
{topic} 

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
Return JSON:

{{
    "results": [
        {{
            "title": "string",
            "url": "string",
            "snippet": "string",
            "published_at": "YYYY-MM-DD | null"
        }}
    ]
}}
"""
)   

def get_research_prompt() -> BasePromptTemplate:
    return research_prompt