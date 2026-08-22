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

from langchain.embeddings import Embeddings
from langchain.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from textual import log


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    selected_tools: list[str]


"""
How old was the 30th president of the United States when he died?
"""


def build_chat_agent_graph(
    llm: BaseChatModel,
    embeddings: Embeddings,
    tools: list[BaseTool],
    *,
    draw_mermaid: bool = False,
) -> CompiledStateGraph:
    def build_retriever(embeddings: Embeddings) -> VectorStoreRetriever:
        return InMemoryVectorStore.from_documents(
            [
                Document(tool.description, metadata={"name": tool.name})
                for tool in tools
            ],
            embeddings,
        ).as_retriever(search_kwargs={"k": 2})

    tool_rag_retriver = build_retriever(embeddings)
    tools_by_name = {tool.name: tool for tool in tools}

    def select_tools(state: State) -> dict:
        query = state["messages"][-1].content
        tool_docs = tool_rag_retriver.invoke(query)
        selected = [doc.metadata["name"] for doc in tool_docs]
        log("selected_tools:", selected)
        return {"selected_tools": selected}

    async def llm_call_node(state: State, config: RunnableConfig) -> dict:
        selected = [tools_by_name[name] for name in state["selected_tools"]]
        message = await llm.bind_tools(selected).ainvoke(state["messages"], config)
        return {"messages": [message]}

    builder = StateGraph(State)
    builder.add_node("selected_tools", select_tools)
    builder.add_node("llm_call", llm_call_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "selected_tools")
    builder.add_edge("selected_tools", "llm_call")
    # tools_condition (from langgraph.prebuilt) already returns either "tools" or the END sentinel itself
    builder.add_conditional_edges("llm_call", tools_condition)
    builder.add_edge("tools", "llm_call")

    _graph = builder.compile(checkpointer=InMemorySaver())
    if draw_mermaid:
        _graph.get_graph().draw_mermaid_png(output_file_path="model_tool_graph.png")

    return _graph


"""
Write an essay about the relevance of 'The Little Prince' today.
"""

def build_essay_writer_agent_graph(
    llm: BaseChatModel,
    embeddings: Embeddings,
    tools: list[BaseTool],
    *,
    draw_mermaid: bool = False,
) -> CompiledStateGraph:
    def build_retriever(embeddings: Embeddings) -> VectorStoreRetriever:
        return InMemoryVectorStore.from_documents(
            [
                Document(tool.description, metadata={"name": tool.name})
                for tool in tools
            ],
            embeddings,
        ).as_retriever(search_kwargs={"k": 2})

    tool_rag_retriver = build_retriever(embeddings)
    tools_by_name = {tool.name: tool for tool in tools}

    def select_tools(state: State) -> dict:
        query = state["messages"][-1].content
        tool_docs = tool_rag_retriver.invoke(query)
        selected = [doc.metadata["name"] for doc in tool_docs]
        log("selected_tools:", selected)
        return {"selected_tools": selected}

    async def llm_call_node(state: State, config: RunnableConfig) -> dict:
        selected = [tools_by_name[name] for name in state["selected_tools"]]
        message = await llm.bind_tools(selected).ainvoke(state["messages"], config)
        return {"messages": [message]}

    generate_prompt = SystemMessage(
        "You are an essay assistant tasked with writing excellent 3-paragraph essays."
        " Generate the best essay possible for the user's request."
        " If the user provides critique, respond with a revised version of your previous attempts."
    )

    reflection_prompt = SystemMessage(
        "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
        " Provide detailed recommendations, including requests for length, depth, style, etc."
    )

    def generate(state: State) -> State:
        answer = llm.invoke([generate_prompt] + state['messages'])
        return {'messages': [answer]}

    def reflect(state: State) -> State:
        cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        # First message is the original user request. We hold it the same for all nodes
        translated = [reflection_prompt, state['messages'][0]] + [
            cls_map[msg.__class__](content=msg.content) for msg in state['messages'][1:]
        ]
        answer = llm.invoke(translated)
        return {'messages': [HumanMessage(content=answer.content)]}

    def should_continue(state: State) -> str:
        if len(state['messages']) > 6:
            return END
        else:
            return "reflect"
    
    builder = StateGraph(State)
    builder.add_node("generate", generate)
    builder.add_node("reflect", reflect)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "generate")
    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("reflect", "generate")

    _graph = builder.compile(checkpointer=InMemorySaver())
    if draw_mermaid:
        _graph.get_graph().draw_mermaid_png(output_file_path="generate_reflect_graph.png")

    return _graph
