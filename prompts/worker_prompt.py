from langchain_core.prompts import BasePromptTemplate, PromptTemplate

worker_prompt = PromptTemplate(
    input_variables=["topic", "mode", "research_results", "section_title", 
                     "goal", "bullets", "target_words", "requires_code", "requires_citations"],
    template="""You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Section: {section_title}
Goal: {goal}

Bullets — cover EVERY point below, minimum 3-4 sentences each, in this exact order:
{bullets}

Target words: {target_words}. Write the FULL amount. Do not stop early.
Requires code: {requires_code}
Requires citations: {requires_citations}

Hard constraints:
- Start with ## {section_title}
- Every bullet must become a real paragraph with explanation, not just a restatement.
- If requires_code is true: write a complete, runnable code example with explanation.
- If requires_citations is true: cite as ([Source](URL)) using only provided Evidence URLs.
- If mode is open_book: every factual claim needs a citation from Evidence.
- No fluff, no marketing. Be precise and implementation-oriented.

Topic: {topic}
Blog kind: {mode}

Evidence:
{research_results}

Write the full section now:""",
)

def get_worker_prompt() -> BasePromptTemplate:
    return worker_prompt