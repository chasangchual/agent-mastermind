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
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langchain_protocol import Annotated
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

class State(TypedDict):
    messages : Annotated[list[AnyMessage], add_messages]
    
    
def build_graph(llm: BaseChatModel, *, draw_mermaid: bool = False) -> CompiledStateGraph:
    async def call_model(state: State, config: RunnableConfig) -> dict:
        # `llm.ainvoke` (not `.astream`) is deliberate under `astream_events`
        # `version="v3"` (see agent_runtime.py): `.astream()` is a legacy
        # `BaseChatModel` code path that only ever fires the old
        # `on_llm_new_token` callback and never touches the v2/v3
        # content-block machinery — v3 protocol-event deltas are only
        # emitted from inside `_agenerate_with_cache`/`.ainvoke()`, driven
        # by a `_V2StreamingCallbackHandler` LangGraph attaches to `config`
        # for a v3 run. `config` must be threaded through so that handler
        # reaches the model call.
        message = await llm.ainvoke(state["messages"], config)
        return {"messages": [message]}

    builder = StateGraph(State)
    builder.add_node("chat", call_model)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    
    _graph = builder.compile(checkpointer=InMemorySaver())
    if draw_mermaid:  # config.draw_mermaid — debug aid, off by default
        _graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    return _graph
