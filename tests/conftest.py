import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings          # noqa: E402
from app.pipeline import RAGPipeline          # noqa: E402
from app.store.sqlite_store import SQLiteStore  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(
        project_name="test",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dim=256,
        data_dir=str(tmp_path),
        trace_enabled=False,
        extra={"use_reranker": False, "top_k": 5, "fetch_k": 20},
    )


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "index.db")


@pytest.fixture
def pipeline(settings, store):
    p = RAGPipeline(settings, store=store)
    p.ingest("data/sample")
    return p
