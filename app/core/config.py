from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError

DEFAULT_CONFIG_NAME = "project.config.json"


def _find_config(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for project.config.json."""
    here = (start or Path.cwd()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / DEFAULT_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


@dataclass
class Settings:
    """Everything the base needs to run, assembled from three layers.

    Precedence, lowest to highest: dataclass defaults -> project.config.json
    (the client's intake-form answers) -> environment variables. The intake form
    describes *the project*; env vars carry *secrets and deployment specifics*.
    Never put a key in project.config.json.
    """

    # --- identity -------------------------------------------------------
    project_name: str = "unnamed-project"
    client_name: str = ""

    # --- providers ------------------------------------------------------
    # "mock" runs the whole system offline with deterministic fake responses,
    # which is what the test-suite and CI use. Nothing else has to change.
    llm_provider: str = "mock"           # mock | openai | anthropic | ollama
    llm_model: str = "gpt-4.1-mini"
    embedding_provider: str = "mock"      # mock | openai | ollama
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 256

    max_output_tokens: int = 1024
    temperature: float = 0.0
    request_timeout_s: float = 60.0

    # --- ollama -----------------------------------------------------------
    # Local, no key required. Ollama serves an OpenAI-compatible API, so the
    # existing OpenAI client classes are reused with this base_url instead.
    ollama_base_url: str = "http://localhost:11434/v1"

    # --- cost guardrails ------------------------------------------------
    # Postings repeatedly mention runaway spend. Refuse rather than surprise.
    max_cost_usd_per_run: float = 1.00
    max_llm_calls_per_run: int = 25

    # --- storage --------------------------------------------------------
    data_dir: str = "./data"

    # --- observability --------------------------------------------------
    trace_enabled: bool = True
    trace_dir: str = "./traces"

    # --- anything the specific base adds -------------------------------
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, *, env: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        raw: dict[str, Any] = {}

        cfg_path = Path(path) if path else _find_config()
        if cfg_path is not None:
            if not Path(cfg_path).is_file():
                raise ConfigError(f"config file not found: {cfg_path}")
            raw = json.loads(Path(cfg_path).read_text())

        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}

        for name in known:
            env_key = f"APP_{name.upper()}"
            if env_key in env:
                kwargs[name] = _coerce(cls.__dataclass_fields__[name].type, env[env_key])

        settings = cls(**kwargs)
        settings.extra.update(extra)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.llm_provider not in {"mock", "openai", "anthropic", "ollama", "openrouter"}:
            raise ConfigError(f"unknown llm_provider: {self.llm_provider!r}")
        if self.embedding_provider not in {"mock", "openai", "ollama", "openrouter"}:
            raise ConfigError(f"unknown embedding_provider: {self.embedding_provider!r}")
        if self.embedding_dim <= 0:
            raise ConfigError("embedding_dim must be positive")
        if self.max_cost_usd_per_run <= 0:
            raise ConfigError("max_cost_usd_per_run must be positive")

    def api_key(self, provider: str, env: dict[str, str] | None = None) -> str:
        env = dict(os.environ if env is None else env)
        key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
              "openrouter": "OPENROUTER_API_KEY"}.get(provider)
        if key is None:
            return ""
        value = env.get(key, "")
        if not value:
            raise ConfigError(
                f"{key} is not set. Either export it, or set the provider to 'mock' "
                f"in {DEFAULT_CONFIG_NAME} to run offline."
            )
        return value


def _coerce(type_: Any, value: str) -> Any:
    name = getattr(type_, "__name__", str(type_))
    if name == "bool":
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if name == "int":
        return int(value)
    if name == "float":
        return float(value)
    return value
