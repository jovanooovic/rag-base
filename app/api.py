from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Cookie, Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer.guardrails import redact
from .core import auth
from .core.config import Settings
from .core.ratelimit import RateLimiter
from .core.providers import Message
from .core.trace import Trace
from .ingest.loaders import SUPPORTED as SUPPORTED_EXTENSIONS
from .pipeline import RAGPipeline, build_store
from .store.access import AccessScope, DocumentACL

app = FastAPI(title="RAG base", version="0.1.0",
              description="Retrieval-augmented answering over a client corpus.")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # 15 MB per file
MAX_FILES_PER_UPLOAD = 10

# POST /feedback is the only write endpoint that is not admin-gated -- it has
# to be, since the whole point is that a reader can disagree without holding a
# token. That makes an unbounded append-only file a disk-fill DoS on anything
# publicly reachable, so the file is capped and further writes are refused
# loudly rather than silently filling the volume. This is a bound, not rate
# limiting; see Known Limitations before exposing this to the open internet.
FEEDBACK_MAX_BYTES = 5 * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


@lru_cache(maxsize=1)
def get_store():
    """One store for the process. Cheap to share; expensive to rebuild per request."""
    return build_store(get_settings())


@lru_cache(maxsize=1)
def get_directory():
    """Only built in the multi-user tier; the single-tenant one has no
    directory to consult."""
    settings = get_settings()
    if not settings.extra.get("multi_tenant"):
        return None
    from .store.directory import Directory
    return Directory(settings.extra["postgres_dsn"])


SESSION_COOKIE = "rag_session"

# Failed logins only; see RateLimiter for why successes are not counted.
_LOGIN_LIMITER = RateLimiter(max_attempts=10, window_seconds=300.0)


def current_user_id(session: str = Cookie(default="", alias=SESSION_COOKIE),
                    authorization: str = Header(default="")) -> str | None:
    """The user this request is signed in as, from cookie or bearer token.

    Cookie for the browser console, bearer for API clients. Both carry the same
    signed token; neither is trusted further than its signature.
    """
    token = session
    if not token and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        return None
    try:
        return auth.read_token(token, secret=auth.secret_key())
    except auth.AuthError:
        return None


def current_scope(user_id: str | None = Depends(current_user_id)) -> AccessScope | None:
    """Who this request runs as, as a retrieval boundary.

    Built from the directory on every request rather than carried in the token.
    A token that carried memberships would keep granting access to a department
    someone was removed from until it expired; deriving it means removal takes
    effect on the next request.
    """
    settings = get_settings()
    if not settings.extra.get("multi_tenant"):
        return None
    if not user_id:
        raise HTTPException(401, "not signed in")
    try:
        return get_directory().scope_for(user_id)
    except LookupError as exc:
        # The account was deleted while a valid token was still in the wild.
        raise HTTPException(401, "unknown user") from exc


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


class FeedbackRequest(BaseModel):
    """A reader disagreeing with an answer, which is the only signal that
    reliably finds the failures a golden set was never written to cover."""
    run_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1, max_length=4000)
    verdict: str = Field(pattern="^(up|down)$")
    note: str | None = Field(default=None, max_length=2000)


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


def _cache_key(req: AskRequest, scope: AccessScope | None = None) -> str:
    payload = {
        # Identity is part of the key, not an afterthought: two users asking
        # the same question have different corpora, so a shared entry would
        # serve one of them the other's answer and citations.
        "identity": None if scope is None else [scope.company_id, scope.user_id,
                                                sorted(scope.department_ids)],
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


class RegisterRequest(BaseModel):
    company_name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class CreateUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)
    department_id: str | None = None
    role: str = Field(default="member", pattern="^(member|manager)$")


def _require_multi_tenant() -> Settings:
    settings = get_settings()
    if not settings.extra.get("multi_tenant"):
        raise HTTPException(404, "authentication is only used in the multi-user tier")
    return settings


def _is_secure_request(request: Request) -> bool:
    """Whether this request actually arrived over TLS.

    A Secure cookie is never sent back over plain HTTP, so hardcoding
    secure=True breaks every http:// deployment silently: login returns 200,
    sets a cookie the browser then refuses to send, and every subsequent
    request is anonymous. Hardcoding False is worse -- it ships the session
    over the wire in the clear.

    So: mark it Secure exactly when the connection is secure. x-forwarded-proto
    covers the usual case of TLS terminated at a proxy or tunnel, where the app
    itself only ever sees http. A forged header can only make the cookie more
    restrictive, never less.
    """
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https"


def _set_session_cookie(response: Response, token: auth.SessionToken,
                        request: Request) -> None:
    response.set_cookie(
        SESSION_COOKIE, token.value,
        httponly=True,      # unreadable from JavaScript, so XSS cannot exfiltrate it
        samesite="strict",  # the CSRF defence: no cross-site request carries it
        secure=_is_secure_request(request),
        max_age=int(auth.TOKEN_TTL.total_seconds()),
        path="/",
    )


@app.post("/auth/register")
def register(req: RegisterRequest, response: Response, request: Request) -> dict[str, Any]:
    """Create a company and its first user, who manages it.

    Whoever signs the company up is by definition its first manager -- there is
    nobody to approve them, and an unmanaged company cannot approve its own
    first document. Everyone after them is created by a manager.
    """
    _require_multi_tenant()
    directory = get_directory()
    try:
        email = auth.validate_email(req.email)
        password_hash = auth.hash_password(req.password)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc

    if directory.find_user_by_email(email) is not None:
        raise HTTPException(409, "that email is already registered")

    company_id = directory.create_company(req.company_name.strip())
    user_id = directory.create_user(company_id, email, password_hash)
    department_id = directory.create_department(company_id, "General")
    directory.add_membership(user_id, department_id, "manager")

    token = auth.issue_token(user_id, secret=auth.secret_key())
    _set_session_cookie(response, token, request)
    return {"user_id": user_id, "company_id": company_id, "department_id": department_id}


@app.post("/auth/login")
def login(req: LoginRequest, response: Response, request: Request) -> dict[str, Any]:
    _require_multi_tenant()
    directory = get_directory()
    email = req.email.strip().lower()
    client_ip = request.client.host if request.client else "unknown"

    # Both keys must be under the limit: per-email alone lets an attacker lock
    # out any account on purpose, per-IP alone lets a botnet spread out.
    keys = (f"email:{email}", f"ip:{client_ip}")
    if not all(_LOGIN_LIMITER.check(k) for k in keys):
        raise HTTPException(429, "too many failed attempts -- wait a few minutes")

    record = directory.find_user_by_email(email)
    # Runs the hash either way; see verify_password on why a missing account
    # must not return faster than a wrong password.
    if not auth.verify_password(record["password_hash"] if record else None, req.password):
        for k in keys:
            _LOGIN_LIMITER.record(k)
        # One message for both cases, so this cannot be used to discover which
        # addresses have accounts.
        raise HTTPException(401, "incorrect email or password")

    for k in keys:
        _LOGIN_LIMITER.reset(k)
    if auth.needs_rehash(record["password_hash"]):
        directory.set_password_hash(record["id"], auth.hash_password(req.password))

    token = auth.issue_token(record["id"], secret=auth.secret_key())
    _set_session_cookie(response, token, request)
    return {"user_id": record["id"], "company_id": record["company_id"]}


@app.post("/auth/logout")
def logout(response: Response) -> dict[str, Any]:
    """Clears the cookie. The token stays valid until it expires -- honest
    limitation of stateless sessions, and the reason the TTL is 12 hours rather
    than a month. Real revocation needs a server-side session table."""
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"signed_out": True}


@app.get("/auth/me")
def me(user_id: str | None = Depends(current_user_id)) -> dict[str, Any]:
    _require_multi_tenant()
    if not user_id:
        raise HTTPException(401, "not signed in")
    directory = get_directory()
    scope = directory.scope_for(user_id)
    return {"user_id": user_id, "company_id": scope.company_id,
            "department_ids": list(scope.department_ids),
            "manages": directory.managed_departments(user_id)}


@app.post("/users")
def create_user(req: CreateUserRequest,
                user_id: str | None = Depends(current_user_id)) -> dict[str, Any]:
    """A manager adds someone to their own company.

    Not open registration: /auth/register creates a company, this adds people
    to one. The company is taken from the caller's own record rather than the
    request body, so a manager cannot create users inside somebody else's.
    """
    _require_multi_tenant()
    if not user_id:
        raise HTTPException(401, "not signed in")
    directory = get_directory()
    managed = directory.managed_departments(user_id)
    if not managed:
        raise HTTPException(403, "only a manager can add users")
    if req.department_id and req.department_id not in managed:
        raise HTTPException(403, "you do not manage that department")

    caller = directory.scope_for(user_id)
    try:
        email = auth.validate_email(req.email)
        password_hash = auth.hash_password(req.password)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    if directory.find_user_by_email(email) is not None:
        raise HTTPException(409, "that email is already registered")

    new_user = directory.create_user(caller.company_id, email, password_hash)
    if req.department_id:
        directory.add_membership(new_user, req.department_id, req.role)
    return {"user_id": new_user}


@app.get("/health")
def health(scope: AccessScope | None = Depends(current_scope)) -> dict[str, Any]:
    s = get_settings()
    # Scoped: an unscoped count tells a user how much exists that they cannot
    # see, which is a small leak but a leak.
    return {"status": "ok", "project": s.project_name, "llm_provider": s.llm_provider,
            "chunks_indexed": get_store().count(access=scope),
            "brand_accent": s.extra.get("brand_accent"),
            "brand_description": s.extra.get("brand_description"),
            "show_source_link": bool(s.extra.get("show_source_link", True))}


@app.post("/ask")
def ask(req: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline),
        scope: AccessScope | None = Depends(current_scope)) -> dict[str, Any]:
    if pipeline.store.count(access=scope) == 0:
        raise HTTPException(409, "index is empty -- run ingestion first (POST /ingest)")

    key = _cache_key(req, scope)
    cached = _ANSWER_CACHE.get(key)
    if cached is not None:
        _ANSWER_CACHE.move_to_end(key)
        return {**cached, "cached": True}

    history = [Message(t.role, t.content) for t in req.history]
    result = pipeline.ask(req.question, history=history or None,
                          where=req.filters, top_k=req.top_k, access=scope)
    body = result.as_dict()

    _ANSWER_CACHE[key] = body
    _ANSWER_CACHE.move_to_end(key)
    if len(_ANSWER_CACHE) > _ANSWER_CACHE_MAX:
        _ANSWER_CACHE.popitem(last=False)
    return {**body, "cached": False}


@app.post("/ingest", dependencies=[Depends(require_admin)])
def ingest(req: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline),
           scope: AccessScope | None = Depends(current_scope)) -> dict[str, Any]:
    """Ingest a path already on the server -- a mounted volume, an S3 sync,
    a scheduled export. Auth-gated: see require_admin."""
    try:
        report = pipeline.ingest(req.path, acl=_acl_for_upload(scope))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    get_store.cache_clear()
    _ANSWER_CACHE.clear()
    return report.as_dict()


def _acl_for_upload(scope: AccessScope | None) -> DocumentACL | None:
    """An upload belongs to whoever sent it, and starts private.

    Nothing becomes department-visible by being uploaded -- that takes a
    manager's approval, which is the point of the queue. Without this stamp a
    chunk carries no owner and no company, so it matches nobody's scope and is
    invisible even to the person who just uploaded it.
    """
    if scope is None:
        return None
    return DocumentACL(company_id=scope.company_id, owner_id=scope.user_id)


@app.post("/upload", dependencies=[Depends(require_admin)])
async def upload(files: list[UploadFile] = File(...),
                  pipeline: RAGPipeline = Depends(get_pipeline),
                  scope: AccessScope | None = Depends(current_scope)) -> dict[str, Any]:
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
        report = pipeline.ingest(upload_dir, acl=_acl_for_upload(scope))
    except RuntimeError as exc:  # e.g. a scanned .pdf that OCR'd to no text
        raise HTTPException(422, str(exc)) from exc
    get_store.cache_clear()
    _ANSWER_CACHE.clear()
    return {**report.as_dict(), "saved_files": saved}


@app.get("/source")
def source(path: str, scope: AccessScope | None = Depends(current_scope)) -> FileResponse:
    """Serve an indexed document's original bytes, for in-browser preview.

    `path` is client-supplied, so it's never used to build a filesystem path
    from scratch -- it's only accepted if it exactly matches some chunk's
    `source` already in the index. That's a stronger check than a path-
    traversal guard: sources are only ever set by the (admin-gated) ingest
    pipeline, so this can't be tricked into serving a file nobody chose to
    index, regardless of where on disk it actually lives.
    """
    # Scoped: unscoped, this serves any indexed file to anyone who can name
    # it, and /documents used to hand out the names.
    known_sources = {c.source for c in get_store().all_chunks(access=scope)}
    if path not in known_sources:
        raise HTTPException(404, "not an indexed document")
    candidate = Path(path)
    if not candidate.is_file():
        raise HTTPException(404, "file not found on disk")
    return FileResponse(candidate)


def _feedback_path(settings: Settings) -> Path:
    return Path(settings.data_dir) / "feedback.jsonl"


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    """Record a reader's verdict on an answer.

    Deliberately not admin-gated: the value of this endpoint is that the person
    who spotted the wrong answer can say so, and they will not have a token.
    That openness is also its risk, hence the size cap and the field limits on
    FeedbackRequest.

    `note` is free text a human typed about a real answer, which is exactly
    where a name or an order number ends up -- so it goes through the same
    redaction as the answer itself when redact_pii is on.
    """
    settings = get_settings()
    path = _feedback_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size >= FEEDBACK_MAX_BYTES:
        raise HTTPException(
            507, f"feedback log is full ({FEEDBACK_MAX_BYTES // (1024 * 1024)}MB). "
                 "Export and rotate it: python -m app.cli feedback-export")

    clean = redact if settings.extra.get("redact_pii", False) else (lambda s: s)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": req.run_id,
        "verdict": req.verdict,
        "question": clean(req.question),
        "note": clean(req.note) if req.note else None,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"recorded": True}


def read_feedback(settings: Settings) -> list[dict[str, Any]]:
    """Rows from the feedback log, skipping any that are unreadable.

    A corrupt line -- a half-written row from a killed process, say -- must not
    take down the admin panel that exists to read this file.
    """
    path = _feedback_path(settings)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@app.get("/feedback", dependencies=[Depends(require_admin)])
def feedback_summary(limit: int = 20) -> dict[str, Any]:
    """Counts plus the most recent negatives -- admin-gated, because it is a
    log of what real users asked, which is client data."""
    rows = read_feedback(get_settings())
    down = [r for r in rows if r.get("verdict") == "down"]
    return {
        "total": len(rows),
        "up": sum(1 for r in rows if r.get("verdict") == "up"),
        "down": len(down),
        "recent_negative": list(reversed(down[-max(0, min(limit, 200)):])),
    }


@app.get("/documents")
def documents(scope: AccessScope | None = Depends(current_scope)) -> dict[str, Any]:
    chunks = get_store().all_chunks(access=scope)
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
