"""Runnable check for observability/tracing.py: tracing stays a no-op unless
both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
"""

from mastermind.observability import tracing


def test_no_op_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert tracing.get_callbacks() == []
    tracing.shutdown()  # must not raise / must not try to reach a client


def test_callback_built_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    callbacks = tracing.get_callbacks()

    assert len(callbacks) == 1
    tracing.shutdown()  # flushing an unreachable host must not raise
