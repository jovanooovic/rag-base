from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Document:
    """One source document, before chunking."""
    doc_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


def load_html(raw: str) -> str:
    p = _HTMLText()
    p.feed(raw)
    return p.text()


def load_csv(raw: str) -> str:
    """Render a CSV as one 'Field: value' record per block.

    Row-per-chunk beats naive text splitting here: a retrieved chunk stays a
    complete record instead of half of two rows.
    """
    reader = csv.DictReader(io.StringIO(raw))
    blocks = []
    for i, row in enumerate(reader):
        body = "\n".join(f"{k}: {v}" for k, v in row.items() if v not in (None, ""))
        blocks.append(f"### record {i + 1}\n{body}")
    return "\n\n".join(blocks)


def load_pdf(path: Path) -> str:
    """PDF text via pypdf if present.

    Left as an optional dependency on purpose: half of client PDFs are scans and
    need OCR anyway, which is a scoping conversation, not a default install.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PDF support needs pypdf: pip install pypdf. "
            "If the client's PDFs are scans you need OCR too -- scope that separately."
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join(f"[page {i + 1}]\n{(p.extract_text() or '').strip()}"
                       for i, p in enumerate(reader.pages))


LOADERS = {
    ".txt": lambda p: p.read_text(errors="replace"),
    ".md": lambda p: p.read_text(errors="replace"),
    ".markdown": lambda p: p.read_text(errors="replace"),
    ".html": lambda p: load_html(p.read_text(errors="replace")),
    ".htm": lambda p: load_html(p.read_text(errors="replace")),
    ".csv": lambda p: load_csv(p.read_text(errors="replace")),
    ".json": lambda p: json.dumps(json.loads(p.read_text()), indent=2),
    ".pdf": load_pdf,
}

SUPPORTED = tuple(LOADERS)


def load_path(path: str | Path) -> Iterator[Document]:
    """Yield Documents from a file or (recursively) a directory."""
    p = Path(path)
    if not p.exists():
        # Fail loudly. A silent zero-document ingest is the worst possible
        # outcome: the index looks fine and every answer becomes a refusal.
        raise FileNotFoundError(f"nothing to ingest at {p}")
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    if not files:
        raise FileNotFoundError(f"no files found under {p}")
    for f in files:
        ext = f.suffix.lower()
        if ext not in LOADERS:
            continue
        text = LOADERS[ext](f)
        if not text.strip():
            continue
        yield Document(
            doc_id=f.as_posix(),
            text=text,
            source=f.as_posix(),
            metadata={"filename": f.name, "ext": ext, "bytes": f.stat().st_size},
        )
