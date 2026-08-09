# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Status:** TUI shell (Header/RichLog transcript/Input/Footer), a slash-command registry (`tui/commands.py`), the `/model` provider+model config dialog (`tui/screens/model_screen.py`, `config/settings.py`), and a LangChain chat-model factory (`llm/`) exist. The LangGraph agent loop (`agent/`), the top-level `commands/`/`skills/`/`mcp/` packages, and observability/tracing are not yet implemented — chat input currently echoes instead of calling an LLM.

**Agent/LLM framework:** the agent loop is built on **LangGraph** (graph-based agent runtime) and the LLM/tool layer on **LangChain** (model integrations, tool binding), not fully custom code. Prefer LangChain/LangGraph primitives over hand-rolled equivalents in `agent/` and `llm/` — e.g. `langchain`'s chat model classes instead of a bespoke provider abstraction, LangGraph's graph/state/checkpoint APIs instead of a hand-rolled loop. Only write custom code where LangChain/LangGraph doesn't already cover the need (e.g. the TUI itself, session persistence).

**Tracing/debug logging:** use **Langfuse** for LLM/agent-run tracing rather than custom tracing code. Unlike LangSmith, Langfuse isn't purely env-var auto-instrumentation: LangChain/LangGraph need Langfuse's `CallbackHandler` (from the `langfuse` package) passed in as a callback on graph/chain invocations. `observability/tracing.py` should own constructing that `CallbackHandler` (reading the `LANGFUSE_*` env vars below) and exposing it for `agent/graph.py` to pass through — keep it limited to that setup, not a custom tracer. `observability/logging.py` is still for ordinary application logs, which Langfuse doesn't cover.

**Textual code comments:** the user is new to Textual. Any code using Textual (`app.py`, everything under `tui/`, and anywhere else `textual` is imported) should carry detailed comments explaining the Textual-specific concepts in play, not just what the line does — e.g. why `compose()`/`on_mount()` are called automatically and when, how `BINDINGS` maps to `action_*` methods, the `on_<widget>_<event>` message-dispatch naming convention, what a given CSS unit or theme variable means, and why blocking calls in event handlers are a problem (use `@work`/`asyncio` instead). This applies to new Textual code and to edits that touch existing Textual code — update or extend the comments rather than leaving them stale.

## Development Philosophy:

You are a lazy senior developer. Lazy means efficient, not careless. The best code is the code never written.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase? Reuse the helper, util, or pattern that's already here, don't re-write it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can this be one line? Make it one line.
7. Only then: write the minimum code that works.

The ladder runs after you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end, then climb.

Bug fix = root cause, not symptom: a report names a symptom. Grep every caller of the function you touch and fix the shared function once — one guard there is a smaller diff than one per caller, and patching only the path the ticket names leaves a sibling caller still broken.

Rules:

- No abstractions that weren't explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Deletion over addition. Boring over clever. Fewest files possible.
- Shortest working diff wins, but only once you understand the problem. The smallest change in the wrong place isn't lazy, it's a second bug.
- Question complex requests: "Do you actually need X, or does Y cover it?"
- Pick the edge-case-correct option when two stdlib approaches are the same size, lazy means less code, not the flimsier algorithm.
- Mark deliberate simplifications that cut a real corner with a known ceiling (global lock, O(n²) scan, naive heuristic) with a `ponytail:` comment naming the ceiling and upgrade path.

Not lazy about: understanding the problem (read it fully and trace the real flow before picking a rung, a small diff you don't understand is just laziness dressed up as efficiency), input validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs (the platform is never the spec ideal, a clock drifts, a sensor reads off), anything explicitly requested. Lazy code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks (an assert-based demo/self-check or one small test file; no frameworks, no fixtures). Trivial one-liners need no test.

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

Build incrementally, like Lego blocks, in this order — each block should run on its own before the next one starts:

1. **TUI** — layout, theme, widgets, dialogs, keyboard handling. Runs with no LLM configured.
2. **Commands** — `/help`, `/model`, `/exit` only, dispatched through a registry (see Architecture Layering below).
3. **LLM** — LangChain integrations for OpenAI, Gemini, Ollama, llama.cpp behind one factory interface.
4. **Agent** — LangGraph conversational graph wired to the selected LLM, with streaming.
5. **Extensibility** — the initial `skills/`, `commands/`, `mcp/` registries/interfaces.

Don't jump ahead: no multi-agent orchestration, persistent memory/sessions, RAG, browser tools, complex approval workflows, autonomous planning, or background tasks until the blocks above work and the user asks for them.

## Architecture Layering

```text
Textual TUI
     │
     ▼
AgentRuntime
     │
     ▼
LangGraph
     │
     ▼
LLM / Skills / MCP / Tools
```

* Textual must not know LangChain or LangGraph implementation details — it talks to `AgentRuntime`, which emits application-level events (`AssistantStarted`, `AssistantToken`, `AssistantCompleted`, `ToolStarted`, `ToolCompleted`, `AgentError`), not raw LangGraph event objects.
* LLM providers must not know anything about Textual.
* Skills and MCP must not depend on Textual.
* Slash commands are dispatched through a registry, never an `if/elif` chain on the command name — new commands are added by registering a handler, not by editing the dispatcher.

## TUI Design

Layout to match (Header shows title + model/status, Footer shows available slash commands + status):

```text
┌───────────────────────────────────────────────────────┐
│ Mastermind                             Model / Status │
├───────────────────────────────────────────────────────┤
│ Conversation                                          │
│                                                       │
│ User                                                  │
│ > Hello                                               │
│                                                       │
│ Assistant                                             │
│ Hello! How can I help?                                │
├───────────────────────────────────────────────────────┤
│ > Type a message...                                   │
├───────────────────────────────────────────────────────┤
│ /help   /model   /skills   /mcp            Ready      │
└───────────────────────────────────────────────────────┘
```

Only `/help`, `/model`, and `/exit` need to be functional first (Block 2 above); other commands can stay registered as stubs. Support markdown rendering, code blocks, scrolling, multiline input, streaming assistant text, modal dialogs, and responsive resizing. Keep TUI logic independent of the LLM/agent implementation — see Architecture Layering.

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
│       ├── llm/                    # LangChain chat models behind one factory interface
│       │   ├── factory.py          # LLMFactory.create(model_config) -> BaseChatModel
│       │   └── openai.py, gemini.py, ollama.py, llamacpp.py   # only split out of factory.py
│       │                                                       # once a provider needs bespoke
│       │                                                       # handling init_chat_model doesn't cover
│       │                                                       # (e.g. llama.cpp via ChatOpenAI+base_url)
│       │
│       ├── commands/               # slash-command registry, decoupled from tui/
│       │   ├── base.py
│       │   ├── registry.py
│       │   └── builtin/            # HelpCommand, ModelCommand, ExitCommand, ...
│       │
│       ├── skills/                 # SKILL.md directories + registry
│       │   ├── loader.py
│       │   ├── registry.py
│       │   └── models.py
│       │
│       ├── mcp/                    # MCP server config, connect/discover, expose tools to the agent
│       │   ├── client.py
│       │   ├── config.py
│       │   └── registry.py
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

Follow the existing repository structure when it differs. Do not reorganize working code solely to match this recommendation — e.g. slash commands currently live in `tui/commands.py` via a dict-based registry (already satisfies "no if/elif", see Architecture Layering); moving them into a top-level `commands/` package with `Command` classes is only worth doing once there's a reason beyond matching this diagram. `tools/` and `sessions/` are deferred (see the "don't jump ahead" list above) — add them when a block actually needs them, not preemptively.

## Development Workflow

Before implementing a change: inspect the existing repo (`pyproject.toml`, `src/mastermind`) and reuse what's already there rather than re-implementing it, then state the proposed changes before writing code. Implement one block from the Project Overview's build order at a time; after each block, run:

```bash
uv run ruff check .
uv run pytest
```

Fix issues before moving to the next block.

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

