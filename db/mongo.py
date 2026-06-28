from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
import os
from dotenv import load_dotenv

load_dotenv()


def get_collection() -> Collection:
    uri = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "blog_db")
    collection_name = os.getenv("MONGODB_COLLECTION", "notes")

    client = MongoClient(uri)
    return client[db_name][collection_name]


def _extract_code_blocks(content: str) -> tuple[str, Optional[str]]:
    """
    Section content se code blocks extract karo.
    Returns: (content without code, code string or None)
    """
    code_pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
    codes = code_pattern.findall(content)

    # content se code blocks hata do
    clean_content = code_pattern.sub("", content).strip()

    code = "\n\n".join(c.strip() for c in codes) if codes else None
    return clean_content, code


def _parse_sections(sections: list[tuple[int, str]]) -> list[dict]:
    """
    Sections list se structured notes banao.
    Each section: (task_id, markdown_string)
    """
    parsed = []
    for _, md in sorted(sections, key=lambda x: x[0]):
        # pehli line heading hai (## Title)
        lines = md.strip().splitlines()
        subtopic = lines[0].lstrip("#").strip() if lines else "Untitled"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        sub_content, code = _extract_code_blocks(body)

        parsed.append({
            "subtopic": subtopic,
            "sub_content": sub_content,
            "code": code,
        })

    return parsed


def save_blog_to_mongo(out: dict) -> str:
    """
    LangGraph output dict leke MongoDB mein save karo.
    Returns: inserted document id (string)
    """
    plan = out["plan"]
    sections = out["sections"]

    document = {
        "topic": out["topic"],
        "blog_title": plan.blog_title,
        "blog_kind": plan.blog_kind,
        "audience": plan.audience,
        "tone": plan.tone,
        "created_at": date.today().isoformat(),
        "sections": _parse_sections(sections),
    }

    collection = get_collection()
    result = collection.insert_one(document)

    return str(result.inserted_id)