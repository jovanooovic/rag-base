"""Shared core used identically by rag-base, agent-base and automation-base.

Keep this package byte-identical across the three repos. When you improve it on a
client branch, backport it to all three mains.
"""
__all__ = ["config", "providers", "trace", "retry", "errors"]
