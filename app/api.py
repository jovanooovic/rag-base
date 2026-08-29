from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core.config import Settings
from .core.providers import Message
from .core.trace import Trace
from .ingest.loaders import SUPPORTED as SUPPORTED_EXTENSIONS
from .pipeline import RAGPipeline, build_store

app = FastAPI(title="RAG base", version="0.1.0",
              description="Retrieval-augmented answering over a client corpus.")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # 15 MB per file
MAX_FILES_PER_UPLOAD = 10


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


# In-process answer cache. A repeated question -- the same suggestion chip
# clicked by a second visitor, a demo re-run -- costs one retrieve+rerank+
# generate the first time and nothing at all after that: no LLM calls, no
# cost, no wait. Keyed on everything that can change the answer (question,
# history, filters, top_k), not just the question text, so a follow-up with
# different conversation context never collides with an unrelated cache hit.
# Cleared on every successful ingest/upload -- new content can turn a cached
# refusal into a real answer, or vice versa, and a stale hit would hide that.
_ANSWER_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_ANSWER_CACHE_MAX = 200  # bounded so a long-running demo can't grow this forever


def _cache_key(req: AskRequest) -> str:
    payload = {
        "question": req.question,
        "history": [(t.role, t.content) for t in req.history],
        "top_k": req.top_k,
        "filters": req.filters,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_admin(x_admin_token: str = Header(default="", alias="X-Admin-Token")) -> None:
    """Gate write endpoints with a shared-secret header.

    A no-op when APP_ADMIN_TOKEN is unset, so local dev and the existing test
    suite keep working untouched. Set it before exposing the API publicly --
    /ingest and /upload otherwise let anyone who can reach the port index
    whatever they want into the corpus.
    """
    token = os.environ.get("APP_ADMIN_TOKEN", "")
    if token and x_admin_token != token:
        raise HTTPException(401, "missing or invalid admin token")


def _safe_filename(name: str) -> str:
    name = Path(name).name  # strip any directory components -- no path traversal
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "upload"
    return name


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {"status": "ok", "project": s.project_name, "llm_provider": s.llm_provider,
            "chunks_indexed": get_store().count(),
            "brand_accent": s.extra.get("brand_accent"),
            "brand_description": s.extra.get("brand_description"),
            "show_source_link": bool(s.extra.get("show_source_link", True))}


@app.post("/ask")
def ask(req: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    if pipeline.store.count() == 0:
        raise HTTPException(409, "index is empty -- run ingestion first (POST /ingest)")

    key = _cache_key(req)
    cached = _ANSWER_CACHE.get(key)
    if cached is not None:
        _ANSWER_CACHE.move_to_end(key)
        return {**cached, "cached": True}

    history = [Message(t.role, t.content) for t in req.history]
    result = pipeline.ask(req.question, history=history or None,
                          where=req.filters, top_k=req.top_k)
    body = result.as_dict()

    _ANSWER_CACHE[key] = body
    _ANSWER_CACHE.move_to_end(key)
    if len(_ANSWER_CACHE) > _ANSWER_CACHE_MAX:
        _ANSWER_CACHE.popitem(last=False)
    return {**body, "cached": False}


@app.post("/ingest", dependencies=[Depends(require_admin)])
def ingest(req: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    """Ingest a path already on the server -- a mounted volume, an S3 sync,
    a scheduled export. Auth-gated: see require_admin."""
    try:
        report = pipeline.ingest(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    get_store.cache_clear()
    _ANSWER_CACHE.clear()
    return report.as_dict()


@app.post("/upload", dependencies=[Depends(require_admin)])
async def upload(files: list[UploadFile] = File(...),
                  pipeline: RAGPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    """Accept files directly, for a demo corpus with no server filesystem
    access. Auth-gated -- this writes to disk and triggers embedding calls,
    neither of which a public client-facing deployment should hand out for free.
    """
    if not files:
        raise HTTPException(400, "no files in request")
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(400, f"at most {MAX_FILES_PER_UPLOAD} files per request")

    settings = get_settings()
    upload_dir = Path(settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                415, f"unsupported file type '{ext}' ({f.filename}) -- "
                     f"supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        body = await f.read(MAX_UPLOAD_BYTES + 1)
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"{f.filename} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
        dest = upload_dir / f"{uuid.uuid4().hex[:8]}_{_safe_filename(f.filename or 'upload')}"
        dest.write_bytes(body)
        saved.append(dest.name)

    try:
        report = pipeline.ingest(upload_dir)
    except RuntimeError as exc:  # e.g. a scanned .pdf that OCR'd to no text
        raise HTTPException(422, str(exc)) from exc
    get_store.cache_clear()
    _ANSWER_CACHE.clear()
    return {**report.as_dict(), "saved_files": saved}


@app.get("/source")
def source(path: str) -> FileResponse:
    """Serve an indexed document's original bytes, for in-browser preview.

    `path` is client-supplied, so it's never used to build a filesystem path
    from scratch -- it's only accepted if it exactly matches some chunk's
    `source` already in the index. That's a stronger check than a path-
    traversal guard: sources are only ever set by the (admin-gated) ingest
    pipeline, so this can't be tricked into serving a file nobody chose to
    index, regardless of where on disk it actually lives.
    """
    known_sources = {c.source for c in get_store().all_chunks()}
    if path not in known_sources:
        raise HTTPException(404, "not an indexed document")
    candidate = Path(path)
    if not candidate.is_file():
        raise HTTPException(404, "file not found on disk")
    return FileResponse(candidate)


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
