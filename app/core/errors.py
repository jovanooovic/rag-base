class BaseError(Exception):
    """Root of every error this codebase raises on purpose."""


class ConfigError(BaseError):
    """The project config is missing something or contradicts itself."""


class ProviderError(BaseError):
    """An upstream model provider failed in a way we could not recover from."""


class RetryableError(ProviderError):
    """Transient upstream failure. `retry.call` will back off and try again."""


class GuardrailError(BaseError):
    """A guardrail refused to let the run continue."""
