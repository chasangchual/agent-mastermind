"""Langfuse setup — the only place that touches the `langfuse` package.

Per CLAUDE.md: LangChain/LangGraph need Langfuse's `CallbackHandler` passed
in explicitly as a callback, there's no env-var auto-instrumentation like
LangSmith. This module just builds that handler from the `LANGFUSE_*` env
vars; `agent/agent_runtime.py` passes it through on `.astream()` calls.
"""

from __future__ import annotations

import os

from langchain_core.callbacks import BaseCallbackHandler


def _is_configured() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def get_callbacks() -> list[BaseCallbackHandler]:
    """Langfuse callback if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, else none.

    Tracing is opt-in via env vars rather than required config, so the agent
    runs the same with or without Langfuse configured.
    """
    if not _is_configured():
        return []

    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def shutdown() -> None:
    """Flush any pending Langfuse spans; call this once on app exit.

    Langfuse's v3+ client batches spans over OpenTelemetry and exports them
    on a background thread/timer — exiting without a final flush can drop
    whatever hasn't been sent yet, most visibly the very last chat turn.
    """
    if not _is_configured():
        return

    from langfuse import get_client

    get_client().flush()
