from __future__ import annotations

import os
from bson import ObjectId
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Blog Notes API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL"))
db = client[os.getenv("MONGODB_DB", "blog_db")]
collection = db[os.getenv("MONGODB_COLLECTION", "notes")]


def serialize(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])
    return doc


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/notes")
def get_all_notes(topic: str = None, blog_kind: str = None):
    filt = {}
    if topic:
        filt["topic"] = {"$regex": topic, "$options": "i"}
    if blog_kind:
        filt["blog_kind"] = blog_kind

    notes = list(collection.find(filt).sort("created_at", -1).limit(50))
    return {"success": True, "data": [serialize(n) for n in notes]}


@app.get("/api/notes/{note_id}")
def get_note(note_id: str):
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=400, detail="Invalid ID")

    note = collection.find_one({"_id": ObjectId(note_id)})
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    return {"success": True, "data": serialize(note)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)