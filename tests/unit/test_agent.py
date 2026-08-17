"""Runnable check for the agent loop: AgentRuntime -> LangGraph -> a fake chat
model, verifying the AgentEvent contract (AssistantStarted -> AssistantToken*
-> AssistantCompleted, or AgentError on failure) without needing a real
provider/API key.
"""

import pytest
import uuid
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from mastermind.agent import agent as runtime_module
from mastermind.agent.events import (
    AgentError,
    AssistantCompleted,
    AssistantStarted,
    AssistantToken,
)
from mastermind.agent.agent import Agent
from mastermind.config.settings import Config, ModelConfig


@pytest.mark.asyncio
async def test_agent_runtime_streams_then_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "build_chat_model",
        lambda cfg: GenericFakeChatModel(messages=iter(["hello world"])),
    )
    agent = Agent(Config(model_config=ModelConfig(provider="ollama", model="test")), str(uuid.uuid4()))

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
        async def astream(self, messages: object) -> object:
            raise RuntimeError("boom")
            yield  # pragma: no cover -- unreachable, keeps this an async generator

    monkeypatch.setattr(runtime_module, "build_chat_model", lambda cfg: BrokenLLM())
    agent = Agent(Config(model_config=ModelConfig(provider="ollama", model="test")), str(uuid.uuid4()))

    events = [event async for event in agent.astream("hi")]

    assert isinstance(events[-1], AgentError)
    assert "boom" in events[-1].error
