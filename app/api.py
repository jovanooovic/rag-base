from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core.config import Settings
from .core.providers import Message
from .core.trace import Trace
from .pipeline import RAGPipeline, build_store

app = FastAPI(title="RAG base", version="0.1.0",
              description="Retrieval-augmented answering over a client corpus.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


@lru_cache(maxsize=1)
def get_store():
    """One store for the process. Cheap to share; expensive to rebuild per request."""
    return build_store(get_settings())


def get_pipeline() -> RAGPipeline:
    """Fresh pipeline per request so the trace and the cost budget are per-request."""
    settings = get_settings()
    return RAGPipeline(settings, store=get_store(),
                       trace=Trace(enabled=settings.trace_enabled, out_dir=settings.trace_dir))


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    path: str


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {"status": "ok", "project": s.project_name, "llm_provider": s.llm_provider,
            "chunks_indexed": get_store().count()}


@app.post("/ask")
def ask(req: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    if pipeline.store.count() == 0:
        raise HTTPException(409, "index is empty -- run ingestion first (POST /ingest)")
    history = [Message(t.role, t.content) for t in req.history]
    result = pipeline.ask(req.question, history=history or None,
                          where=req.filters, top_k=req.top_k)
    return result.as_dict()


@app.post("/ingest")
def ingest(req: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    """Ingest a path on the server.

    Left as a server-side path rather than an upload endpoint on purpose: in
    every client deployment so far the corpus arrives via a mounted volume, an
    S3 sync, or a scheduled export -- not via someone POSTing a file. Add an
    upload route when a client actually needs one, and put auth in front of it.
    """
    try:
        report = pipeline.ingest(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    get_store.cache_clear()
    return report.as_dict()


@app.get("/documents")
def documents() -> dict[str, Any]:
    chunks = get_store().all_chunks()
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.source] = by_doc.get(c.source, 0) + 1
    return {"documents": len(by_doc), "chunks": len(chunks),
            "by_source": sorted(({"source": k, "chunks": v} for k, v in by_doc.items()),
                                key=lambda d: -d["chunks"])}


# Mounted last so it never shadows an API route above. Optional: a bare
# clone without web/ (e.g. an API-only deployment) still serves fine.
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
