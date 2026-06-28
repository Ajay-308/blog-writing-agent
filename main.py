from __future__ import annotations

import sys
from datetime import date
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from graph.graph import app
from db.mongo import save_blog_to_mongo


def run(topic: str, as_of: Optional[str] = None) -> dict:
    if as_of is None:
        as_of = date.today().isoformat()

    print(f"\n🚀 Starting blog generation for: '{topic}'\n")

    out = app.invoke({
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    })

    print(f"\n✅ Done! Blog: {out['plan'].blog_title}.md\n")

    # MongoDB mein save karo
    try:
        doc_id = save_blog_to_mongo(out)
        print(f"📦 MongoDB mein save ho gaya! ID: {doc_id}\n")
    except Exception as e:
        print(f"⚠️  MongoDB save failed: {e}\n")

    return out


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Self Attention in Transformer Architecture"
    run(topic)