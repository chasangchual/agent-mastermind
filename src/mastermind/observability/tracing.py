"""Langfuse setup — the only place that touches the `langfuse` package.

Per CLAUDE.md: LangChain/LangGraph need Langfuse's `CallbackHandler` passed
in explicitly as a callback, there's no env-var auto-instrumentation like
LangSmith. This module just builds that handler from the `LANGFUSE_*` env
vars; `agent/graph.py` passes it through on `.compile()`/`.astream()` calls.
"""

from __future__ import annotations

import os

from langchain_core.callbacks import BaseCallbackHandler


def get_callbacks() -> list[BaseCallbackHandler]:
    """Langfuse callback if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, else none.

    Tracing is opt-in via env vars rather than required config, so the agent
    runs the same with or without Langfuse configured.
    """
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        return []

    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]
