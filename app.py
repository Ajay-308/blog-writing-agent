from __future__ import annotations

import sys
import os
from datetime import date

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from graph.graph import app
from db.mongo import save_blog_to_mongo, get_collection

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blog Notes Generator",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Blog Notes Generator")
st.caption("Topic likho → AI blog banayega → MongoDB mein save hoga")

# ── Sidebar: past notes ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 Saved Notes")
    try:
        col = get_collection()
        past = list(col.find({}, {"topic": 1, "blog_title": 1, "created_at": 1}).sort("created_at", -1).limit(20))
        if past:
            for doc in past:
                st.markdown(f"**{doc.get('blog_title', doc.get('topic', 'Untitled'))}**")
                st.caption(f"🗓 {doc.get('created_at', '')}")
                st.divider()
        else:
            st.info("Abhi koi notes nahi hain.")
    except Exception as e:
        st.warning(f"MongoDB connect nahi hua: {e}")

# ── Main: topic input ─────────────────────────────────────────────────────────
topic = st.text_input(
    "📌 Topic likho:",
    placeholder="e.g. Self Attention in Transformer Architecture",
)

generate_btn = st.button("🚀 Generate Blog Notes", type="primary", disabled=not topic.strip())

# ── Generate ──────────────────────────────────────────────────────────────────
if generate_btn and topic.strip():
    with st.spinner("⏳ Blog generate ho raha hai... (1-2 min lag sakte hain)"):
        try:
            out = app.invoke({
                "topic": topic.strip(),
                "mode": "",
                "needs_research": False,
                "queries": [],
                "evidence": [],
                "plan": None,
                "as_of": date.today().isoformat(),
                "sections": [],
                "merged_md": "",
                "md_with_placeholders": "",
                "image_specs": [],
                "final": "",
            })
            st.success(f"✅ Blog ready: **{out['plan'].blog_title}**")
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            st.stop()

    # MongoDB save
    try:
        doc_id = save_blog_to_mongo(out)
        st.success(f"📦 MongoDB mein save! ID: `{doc_id}`")
    except Exception as e:
        st.warning(f"⚠️ MongoDB save failed: {e}")

    # ── Display sections ──────────────────────────────────────────────────────
    st.divider()
    plan = out["plan"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Audience", plan.audience)
    col2.metric("Blog Kind", plan.blog_kind)
    col3.metric("Tone", plan.tone)

    st.divider()

    sections = sorted(out["sections"], key=lambda x: x[0])
    for _, md in sections:
        lines = md.strip().splitlines()
        subtopic = lines[0].lstrip("#").strip() if lines else "Untitled"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        with st.expander(f"📄 {subtopic}", expanded=True):
            # code blocks alag dikhao
            import re
            code_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
            codes = code_pattern.findall(body)
            clean_body = code_pattern.sub("", body).strip()

            st.markdown(clean_body)

            for lang, code in codes:
                st.code(code.strip(), language=lang or "python")

    # Full markdown download
    st.divider()
    st.download_button(
        label="⬇️ Download Markdown",
        data=out.get("final", out.get("merged_md", "")),
        file_name=f"{plan.blog_title}.md",
        mime="text/markdown",
    )