"""Runnable check for the agent loop: AgentRuntime -> LangGraph -> a fake chat
model, verifying the AgentEvent contract (AssistantStarted -> AssistantToken*
-> AssistantCompleted, or AgentError on failure) without needing a real
provider/API key.
"""

import uuid

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from mastermind.agent import agent as runtime_module
from mastermind.agent.agent import Agent
from mastermind.agent.events import (
    AgentError,
    AssistantCompleted,
    AssistantStarted,
    AssistantToken,
)
from mastermind.config.settings import Config, EmbeddingConfig, ModelConfig

_FAKE_EMBEDDING_CONFIG = EmbeddingConfig(provider="ollama", model="test")


class ToolBindableFakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel has no real tool-calling model behind it, so its
    inherited bind_tools() raises NotImplementedError - the graph always
    calls bind_tools() on the selected subset, so tests need a no-op here.
    """

    def bind_tools(self, tools: object, **kwargs: object) -> "ToolBindableFakeChatModel":
        return self


@pytest.mark.asyncio
async def test_agent_runtime_streams_then_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "build_chat_model",
        lambda cfg: ToolBindableFakeChatModel(messages=iter(["hello world"])),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_embedding",
        lambda cfg: DeterministicFakeEmbedding(size=8),
    )
    agent = Agent(
        Config(
            model_config=ModelConfig(provider="ollama", model="test"),
            embedding_config=_FAKE_EMBEDDING_CONFIG,
        ),
        str(uuid.uuid4()),
    )

    events = [event async for event in agent.astream("hi")]

    assert isinstance(events[0], AssistantStarted)
    assert isinstance(events[-1], AssistantCompleted)
    assert events[-1].text == "hello world"
    tokens = [e.text for e in events if isinstance(e, AssistantToken)]
    assert "".join(tokens) == "hello world"


@pytest.mark.asyncio
async def test_agent_runtime_yields_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenLLM:
        def bind_tools(self, tools: object, **kwargs: object) -> "BrokenLLM":
            return self

        async def ainvoke(self, messages: object, config: object) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr(runtime_module, "build_chat_model", lambda cfg: BrokenLLM())
    monkeypatch.setattr(
        runtime_module,
        "build_embedding",
        lambda cfg: DeterministicFakeEmbedding(size=8),
    )
    agent = Agent(
        Config(
            model_config=ModelConfig(provider="ollama", model="test"),
            embedding_config=_FAKE_EMBEDDING_CONFIG,
        ),
        str(uuid.uuid4()),
    )

    events = [event async for event in agent.astream("hi")]

    assert isinstance(events[-1], AgentError)
    assert "boom" in events[-1].error


@pytest.mark.asyncio
async def test_agent_dump_history_renders_thread_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "build_chat_model",
        lambda cfg: ToolBindableFakeChatModel(messages=iter(["hello world"])),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_embedding",
        lambda cfg: DeterministicFakeEmbedding(size=8),
    )
    agent = Agent(
        Config(
            model_config=ModelConfig(provider="ollama", model="test"),
            embedding_config=_FAKE_EMBEDDING_CONFIG,
        ),
        str(uuid.uuid4()),
    )

    assert agent.dump_history() == "(no messages yet)"

    [_ async for _ in agent.astream("hi")]

    dump = agent.dump_history()
    assert "[human]\nhi" in dump
    assert "[ai]\nhello world" in dump
