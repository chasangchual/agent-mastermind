"""The LangGraph agent graph: START -> agent -> END.

`MessagesState` is LangGraph's built-in state schema (a `messages` list with
an `add_messages` reducer that appends/merges by message id) — reused as-is
rather than hand-rolling a state schema, since a plain running chat history
is all this graph needs so far. A checkpointer (`InMemorySaver`) is what
gives `AgentRuntime` multi-turn memory: it snapshots state per `thread_id`
between `.astream()` calls, so the caller only ever supplies the *new*
human message, not the whole history.

Single node today; this is the seam where Tool/Skill/MCP/sub-agent/human-
approval nodes get added later (Block 5) without the graph's shape needing
to change — same idea CLAUDE.md's Architecture Layering describes.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    async def call_model(state: MessagesState) -> dict:
        # `llm.astream` (not `.ainvoke`) is what makes token-by-token chunks
        # observable outside this node: LangGraph's `stream_mode="messages"`
        # (see agent/runtime.py) surfaces each chunk as it's produced here,
        # not just the node's final return value.
        full = None
        async for chunk in llm.astream(state["messages"]):
            full = chunk if full is None else full + chunk
        return {"messages": [full]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=InMemorySaver())
