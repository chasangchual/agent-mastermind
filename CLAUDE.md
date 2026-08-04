# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Status:** basic scaffolding only — `pyproject.toml` and a minimal `mastermind` package/TUI shell exist, but the agent loop, LLM layer, tools, and skills described below are not yet implemented.

**Agent/LLM framework:** the agent loop is built on **LangGraph** (graph-based agent runtime) and the LLM/tool layer on **LangChain** (model integrations, tool binding), not fully custom code. Prefer LangChain/LangGraph primitives over hand-rolled equivalents in `agent/` and `llm/` — e.g. `langchain`'s chat model classes instead of a bespoke provider abstraction, LangGraph's graph/state/checkpoint APIs instead of a hand-rolled loop. Only write custom code where LangChain/LangGraph doesn't already cover the need (e.g. the TUI itself, session persistence).

**Tracing/debug logging:** use **Langfuse** for LLM/agent-run tracing rather than custom tracing code. Unlike LangSmith, Langfuse isn't purely env-var auto-instrumentation: LangChain/LangGraph need Langfuse's `CallbackHandler` (from the `langfuse` package) passed in as a callback on graph/chain invocations. `observability/tracing.py` should own constructing that `CallbackHandler` (reading the `LANGFUSE_*` env vars below) and exposing it for `agent/graph.py` to pass through — keep it limited to that setup, not a custom tracer. `observability/logging.py` is still for ordinary application logs, which Langfuse doesn't cover.

**Textual code comments:** the user is new to Textual. Any code using Textual (`app.py`, everything under `tui/`, and anywhere else `textual` is imported) should carry detailed comments explaining the Textual-specific concepts in play, not just what the line does — e.g. why `compose()`/`on_mount()` are called automatically and when, how `BINDINGS` maps to `action_*` methods, the `on_<widget>_<event>` message-dispatch naming convention, what a given CSS unit or theme variable means, and why blocking calls in event handlers are a problem (use `@work`/`asyncio` instead). This applies to new Textual code and to edits that touch existing Textual code — update or extend the comments rather than leaving them stale.

## Project Overview

**mastermind** is a Python-based interactive AI agent application with a terminal user interface (TUI).

The application provides an agent experience similar to Hermes Agent, including:

* Interactive terminal chat
* Streaming LLM responses
* Tool and skill execution
* Multi-step agent loops
* Conversation and session management
* User approval for sensitive operations
* Local and remote LLM provider support
* Extensible tools, skills, and commands

Python dependencies, virtual environments, and command execution are managed exclusively with `uv`.

## Project Naming

Use the following naming conventions consistently:

* Project name: `mastermind`
* CLI command: `mastermind`
* Python package: `mastermind`
* Environment-variable prefix: `MASTERMIND_`
* Configuration directory: `mastermind`
* User data directory: `mastermind`

Do not use `agent-tui` or `agent_tui` in new code.

## Common Development Commands

### Install dependencies

```bash
uv sync
```

### Install development dependencies

```bash
uv sync --group dev
```

### Run the application

```bash
uv run mastermind
```

Alternative module entry point:

```bash
uv run python -m mastermind
```

### Run tests

```bash
uv run pytest
```

### Run tests with coverage

```bash
uv run pytest --cov=mastermind --cov-report=term-missing
```

### Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

### Type checking

```bash
uv run pyright
```

### Final verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Recommended Project Structure

```text
mastermind/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── mastermind/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── app.py
│       │
│       ├── agent/                  # LangGraph graph(s), state, checkpointing
│       │   ├── graph.py
│       │   ├── state.py
│       │   └── events.py
│       │
│       ├── llm/                    # thin config/factory around langchain chat models
│       │   └── factory.py
│       │
│       ├── tools/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── executor.py
│       │   └── builtin/
│       │       ├── filesystem.py
│       │       ├── shell.py
│       │       └── web.py
│       │
│       ├── skills/
│       │   ├── loader.py
│       │   ├── registry.py
│       │   └── models.py
│       │
│       ├── sessions/
│       │   ├── manager.py
│       │   ├── repository.py
│       │   └── models.py
│       │
│       ├── tui/
│       │   ├── screens/
│       │   ├── widgets/
│       │   ├── messages.py
│       │   ├── keybindings.py
│       │   └── theme.py
│       │
│       ├── config/
│       │   ├── settings.py
│       │   └── paths.py
│       │
│       └── observability/
│           ├── logging.py
│           └── tracing.py
│
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

Follow the existing repository structure when it differs. Do not reorganize working code solely to match this recommendation.

## Configuration

Suggested environment variables:

```text
MASTERMIND_PROVIDER
MASTERMIND_MODEL
MASTERMIND_LOG_LEVEL
MASTERMIND_CONFIG_PATH
MASTERMIND_DATA_DIR
MASTERMIND_MAX_ITERATIONS

ANTHROPIC_API_KEY
OPENAI_API_KEY
OLLAMA_BASE_URL
LLAMA_CPP_BASE_URL

LANGFUSE_SECRET_KEY
LANGFUSE_PUBLIC_KEY
LANGFUSE_HOST
```

## Exception Naming

Project-specific exceptions should use the application name:

```python
class MastermindError(Exception):
    """Base exception for expected mastermind failures."""


class ProviderError(MastermindError):
    """Raised when an LLM provider operation fails."""


class ToolExecutionError(MastermindError):
    """Raised when a tool cannot complete successfully."""


class ConfigurationError(MastermindError):
    """Raised when application configuration is invalid."""
```

## `pyproject.toml` Entry Point

The project should expose the `mastermind` command:

```toml
[project]
name = "mastermind"

[project.scripts]
mastermind = "mastermind.cli:main"
```

The corresponding entry function should be similar to:

```python
# src/mastermind/cli.py

from __future__ import annotations


def main() -> None:
    """Start the mastermind terminal application."""
    from mastermind.app import MastermindApp

    app = MastermindApp()
    app.run()
```

The module entry point should support:

```bash
uv run python -m mastermind
```

Example:

```python
# src/mastermind/__main__.py

from mastermind.cli import main

if __name__ == "__main__":
    main()
```

