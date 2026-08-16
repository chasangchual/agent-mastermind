"""The seam between the TUI and LangGraph.

Per CLAUDE.md's Architecture Layering, Textual talks only to `AgentRuntime`
and only ever sees the `AgentEvent` types from `agent/events.py` — never a
LangGraph chunk/event object or a LangChain message type.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from logging import Logger
import uuid 

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    trim_messages,
)
from langchain_protocol import Any
from langgraph.types import StateSnapshot
from langgraph.graph.message import REMOVE_ALL_MESSAGES
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
    """Runs the graph for one turn and relays it as `AgentEvent`s.

    `astream()` opens the graph run via `astream_events(version="v3")`, which multiplexes every event the run produces, not just chat tokens.

    Two `method`s show up on the raw event log:
    - "values": a full `{"messages": [...]}` state snapshot after each step (start of turn, end of turn, ...). Powers the `.values`  projection, not `.messages`.
    - "messages": one event per step of a chat model's content-block lifecycle, `payload = (event_dict, metadata)`. For a single text response, `event_dict["event"]` is always this sequence:

        | event_dict["event"]     | payload                                                               | meaning                    |
        |-------------------------|-----------------------------------------------------------------------|----------------------------|
        | message-start           | {event, id, role: "ai"}                                               | model run began            |
        | content-block-start     | {event, index: 0, content: {type: "text", text: ""}}                  | block 0 opened             |
        | content-block-delta ×N  | {event, index: 0, delta: {type: "text-delta", text: "hello"}}         | one per token/chunk        |
        | content-block-finish    | {event, index: 0, content: {type: "text", text: "hello world this"}}  | block 0 closed, full text  |
        | message-finish          | {event}                                                               | model run done             |

    `run.messages` groups "messages" events by `run_id` into one
    `ChatModelStream` per LLM call; `stream.text` is fed only by
    `content-block-delta`, so `astream()`'s token loop naturally ignores
    the start/finish markers and the "values" snapshots.
    """

    # ponytail: one fixed thread_id, so there's a single running conversation
    # and no multi-session support — add real per-session thread ids once
    # /sessions (deferred, see CLAUDE.md's "don't jump ahead" list) exists.
    _DEFAULT_THREAD_ID = "default"

    def __init__(self, config: Config, thread_id: str = _DEFAULT_THREAD_ID) -> None:
        assert config.model_config is not None

        self._llm = build_chat_model(config.model_config)
        self._graph = build_graph(self._llm, draw_mermaid=config.draw_mermaid)
        self._thread_id = thread_id
        self._compact_max_token = config.compact_max_token
        self._max_iterations = config.max_iterations

    async def astream(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Run one turn, yielding AgentEvents as the reply streams in."""
        yield AssistantStarted()
        buffer = ""
        try:
            async with await self._graph.astream_events(
                {"messages": [HumanMessage(content=user_text)]},
                config={
                    "configurable": self._get_thread(self._thread_id),
                    "callbacks": get_callbacks(),
                    "run_name": "generate-response",
                    # Langfuse tracing (see observability/tracing.py): one trace
                    # per turn, grouped into one session per running thread —
                    # https://langfuse.com/docs/observability/best-practices.
                    # `run_name` becomes the trace name; verb-first per that
                    # guide's naming convention, not the model name (that's
                    # already a generation-level attribute) or a per-turn
                    # value (names should stay stable across traces).
                    "metadata": {"langfuse_session_id": self._thread_id},
                },
                version="v3",
            ) as run:
                async for stream in run.messages:
                    async for text in stream.text:
                        buffer += text
                        yield AssistantToken(text)
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- any provider/network error must surface to the TUI, not crash it
            yield AgentError(repr(ex))
            return

        yield AssistantCompleted(buffer)

    def compact_history(self) -> int:
        thread = self._get_thread(self._thread_id)
        messages = self._get_state_messages(self._thread_id)
        trimmed = trim_messages(
            messages,
            max_tokens=self._compact_max_token,
            token_counter=self._llm,
            strategy="last",
            start_on="human",
        )
        droppedCount = len(messages) - len(trimmed)
        if droppedCount > 0:
            self._graph.update_state(
                {"configurable": thread},
                {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]},
            )
        return droppedCount

    def clear_history(self) -> None:
        thread = self._get_thread(self._thread_id)
        self._graph.update_state(
            {"configurable": thread},
            {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]},
        )
        self._thread_id = str(uuid.uuid4())  # new thread id, so a new session in Langfuse


    def _get_state_message_for_default_thread(self) -> list[HumanMessage | AIMessage]:
        snapshot = self._get_state_snapshot_for_default_thread()
        return snapshot.values.get("messages", [])

    def _get_state_messages(self, thread_id: str) -> list[HumanMessage | AIMessage]:
        snapshot = self._get_state_snapshot(thread_id)
        return snapshot.values.get("messages", [])

    def _get_state_snapshot_for_default_thread(self) -> StateSnapshot:
        return self._get_state_snapshot(self._DEFAULT_THREAD_ID)

    def _get_state_snapshot(self, thread_id: str) -> StateSnapshot:
        return self._graph.get_state({"configurable": self._get_thread(thread_id)})

    def _get_default_thread(self) -> Any:
        return self._get_thread(self._DEFAULT_THREAD_ID)

    def _get_thread(self, thread_id: str) -> Any:
        return {"thread_id": thread_id}
