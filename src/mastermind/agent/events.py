"""Application-level events AgentRuntime yields to its caller.

Per CLAUDE.md's Architecture Layering, this is the only vocabulary the TUI is
allowed to know about — it must never see a raw LangGraph chunk/event object.
Only the assistant-turn events exist so far; ToolStarted/ToolCompleted/
AgentApprovalRequested get added once tools/skills/MCP (Block 5) give the
graph something to call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssistantStarted:
    """The agent has started producing a reply."""


@dataclass(frozen=True)
class AssistantToken:
    """One streamed chunk of the assistant's reply."""

    text: str


@dataclass(frozen=True)
class AssistantCompleted:
    """The assistant's reply finished; `text` is the full message."""

    text: str


@dataclass(frozen=True)
class AgentError:
    """The agent run failed; `error` is a user-facing message."""

    error: str


AgentEvent = AssistantStarted | AssistantToken | AssistantCompleted | AgentError
