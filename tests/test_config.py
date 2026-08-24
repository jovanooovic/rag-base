import pytest

from app.core.config import Settings
from app.core.errors import ConfigError
from app.core.providers import build_embeddings, build_llm


def test_validate_accepts_openrouter_for_llm_and_embeddings():
    s = Settings(llm_provider="openrouter", embedding_provider="openrouter")
    s.validate()  # must not raise


def test_validate_rejects_an_unknown_llm_provider():
    with pytest.raises(ConfigError):
        Settings(llm_provider="not-a-real-provider").validate()


def test_api_key_demands_openrouter_key_when_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings().api_key("openrouter")


def test_api_key_returns_the_openrouter_key_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert Settings().api_key("openrouter") == "sk-or-test"


def test_build_llm_points_openrouter_at_the_right_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    s = Settings(llm_provider="openrouter", llm_model="openai/gpt-4o-mini")
    llm = build_llm(s)
    assert str(llm.inner._client.base_url) == "https://openrouter.ai/api/v1/"


def test_build_embeddings_points_openrouter_at_the_right_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    s = Settings(embedding_provider="openrouter", embedding_model="qwen/qwen3-embedding-8b",
                embedding_dim=4096)
    emb = build_embeddings(s)
    assert str(emb._client.base_url) == "https://openrouter.ai/api/v1/"
    assert emb.dim == 4096
