from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Blog Notes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

MONGODB_URL = os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL")

client = MongoClient(
    MONGODB_URL,
    tls=True,
    tlsAllowInvalidCertificates=True,  # SSL issue fix
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
)
db = client[os.getenv("MONGODB_DB", "blog_db")]
collection: Collection = db[os.getenv("MONGODB_COLLECTION", "notes")]


def serialize(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/")
def root():
    return {"status": "ok", "message": "Blog Notes API is running"}


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
    try:
        from bson.objectid import ObjectId
        oid = ObjectId(note_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")

    note = collection.find_one({"_id": oid})
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"success": True, "data": serialize(note)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)