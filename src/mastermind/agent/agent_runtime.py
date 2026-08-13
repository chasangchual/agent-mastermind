"""The seam between the TUI and LangGraph.

Per CLAUDE.md's Architecture Layering, Textual talks only to `AgentRuntime`
and only ever sees the `AgentEvent` types from `agent/events.py` — never a
LangGraph chunk/event object or a LangChain message type.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from mastermind.agent.events import (
    AgentError,
    AgentEvent,
    AssistantCompleted,
    AssistantStarted,
    AssistantToken,
)
from mastermind.agent.graph import build_graph
from mastermind.config.settings import Config
from mastermind.llm import build_chat_model
from mastermind.observability.tracing import get_callbacks


class AgentRuntime:
    # ponytail: one fixed thread_id, so there's a single running conversation
    # and no multi-session support — add real per-session thread ids once
    # /sessions (deferred, see CLAUDE.md's "don't jump ahead" list) exists.
    _THREAD_ID = "default"

    def __init__(self, config: Config) -> None:
        assert config.model_config is not None
        llm = build_chat_model(config.model_config)
        self._graph = build_graph(llm, draw_mermaid=config.draw_mermaid)

    async def astream(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Run one turn, yielding AgentEvents as the reply streams in."""
        yield AssistantStarted()
        buffer = ""
        try:
            async for message_chunk, _metadata in self._graph.astream(
                {"messages": [HumanMessage(content=user_text)]},
                config={
                    "configurable": {"thread_id": self._THREAD_ID},
                    "callbacks": get_callbacks(),
                    # Langfuse tracing (see observability/tracing.py): one trace
                    # per turn, grouped into one session per running thread —
                    # https://langfuse.com/docs/observability/best-practices.
                    # `run_name` becomes the trace name; verb-first per that
                    # guide's naming convention, not the model name (that's
                    # already a generation-level attribute) or a per-turn
                    # value (names should stay stable across traces).
                    "run_name": "generate-response",
                    "metadata": {"langfuse_session_id": self._THREAD_ID},
                },
                stream_mode="messages",
            ):
                # `.content` is typed as `str | list[...]` because LangChain
                # messages can carry multimodal content blocks; plain text
                # chat models (all we support so far) only ever produce str.
                if (
                    isinstance(message_chunk, AIMessageChunk)
                    and isinstance(message_chunk.text, str)
                    and message_chunk.text
                ):
                    buffer += message_chunk.text
                    yield AssistantToken(message_chunk.text)
        except Exception as exc:  # noqa: BLE001 -- any provider/network error must surface to the TUI, not crash it
            yield AgentError(str(exc))
            return
        yield AssistantCompleted(buffer)

    def _get_state(self) -> list[HumanMessage | AIMessage]:
        """Return the current conversation history, for display in the TUI."""
        snapshot = self._graph.get_state({"configurable": {"thread_id": self._THREAD_ID}})
        return snapshot.values.get("messages", [])
