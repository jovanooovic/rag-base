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


def test_validate_rejects_an_unknown_extra_key_with_a_suggestion():
    """A typo'd safety knob is the dangerous case: redact_pii2 reads as
    'redaction off' and nothing anywhere says so."""
    s = Settings(extra={"redact_pii2": True})
    with pytest.raises(ConfigError, match="redact_pii2"):
        s.validate()
    try:
        s.validate()
    except ConfigError as exc:
        assert "redact_pii" in str(exc)  # the suggestion


def test_validate_accepts_every_documented_extra_key():
    from app.core.config import KNOWN_EXTRA_KEYS
    extra = {k: None for k in KNOWN_EXTRA_KEYS}
    extra.update({"store_backend": "sqlite", "min_top_score": None, "use_reranker": True})
    Settings(extra=extra).validate()  # must not raise


def test_pgvector_without_a_dsn_is_a_clear_config_error_not_a_keyerror():
    with pytest.raises(ConfigError, match="postgres_dsn"):
        Settings(extra={"store_backend": "pgvector"}).validate()


def test_min_top_score_outside_zero_to_one_is_rejected():
    """The old raw-score scale invited values like 0.15 that meant nothing."""
    with pytest.raises(ConfigError, match="between 0.0 and 1.0"):
        Settings(extra={"min_top_score": 7.5}).validate()


def test_min_top_score_without_a_reranker_is_rejected():
    """Fusion confidence measures leg agreement, not relevance -- a floor on it
    would be a false sense of protection."""
    with pytest.raises(ConfigError, match="use_reranker"):
        Settings(extra={"min_top_score": 0.5, "use_reranker": False}).validate()
