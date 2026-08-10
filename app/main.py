from pathlib import Path
import json

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .rag import RAG
from .config import (
    PDF_PATH, EMBED_MODEL, RERANKER_MODEL, OLLAMA_MODEL, OLLAMA_URL,
    INDEX_DIR, RERANKER_DEVICE,
)

app = FastAPI(title="PPL Corporate Intelligence RAG v5", version="5.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
rag = None


class ChatRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup():
    global rag
    rag = RAG()


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


def _ollama_status():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if not r.ok:
            return {"ok": False, "detail": f"HTTP {r.status_code}"}
        names = [m.get("name") for m in r.json().get("models", [])]
        return {"ok": True, "model_configured": OLLAMA_MODEL in names, "models": names}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


@app.get("/health")
def health():
    if rag is None:
        return {"ok": False, "indexed": False}
    ollama = _ollama_status()
    return {
        "ok": True,
        "indexed": True,
        "schema_version": rag.meta.get("schema_version"),
        "records": rag.meta.get("records"),
        "embedding_device": str(rag.embedder.device),
        "reranker_device": rag.reranker_device,
        "ollama": ollama,
        "pdf_exists": PDF_PATH.exists(),
    }


@app.get("/api/status")
def status():
    return health()


@app.get("/api/about")
def about():
    meta = {}
    p = INDEX_DIR / "meta.json"
    if p.exists():
        try:
            meta = json.loads(p.read_text(encoding="utf8"))
        except Exception:
            pass
    ollama = _ollama_status()
    return {
        "name": "PPL Corporate Intelligence RAG v5",
        "document": PDF_PATH.name,
        "document_available": PDF_PATH.exists(),
        "index_schema": meta.get("schema_version", "missing"),
        "indexed_evidence_units": meta.get("records", "unknown"),
        "architecture": "Table/field-aware hybrid RAG: BGE-M3 + FAISS + BM25 + RRF + BGE Reranker + local Ollama",
        "embedding_model": EMBED_MODEL,
        "embedding_device": str(rag.embedder.device) if rag else "not loaded",
        "reranker_model": RERANKER_MODEL,
        "reranker_device": RERANKER_DEVICE,
        "generation_model": OLLAMA_MODEL,
        "ollama_endpoint": OLLAMA_URL,
        "ollama_online": ollama.get("ok", False),
        "ollama_model_available": ollama.get("model_configured", False),
        "huggingface_network": "disabled / local-only",
        "citation_policy": "Claims must cite supplied E# evidence; sources expose report page, PDF page, section, table, exact field, values, quote and PDF coordinates when available.",
        "v5_upgrade": "Intent-aware retrieval, stricter source-grounding, field/value cards, PDF coordinates, robust viewer navigation, health diagnostics and GPU reranking.",
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        return rag.answer(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/report")
def report():
    if not PDF_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {PDF_PATH}")
    return FileResponse(
        PDF_PATH,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline; filename=\"PPL_AR_2025.pdf\"",
            "Cache-Control": "no-store, max-age=0",
            "Accept-Ranges": "bytes",
        },
    )
