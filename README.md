# mastermind

An interactive AI agent with a terminal user interface (TUI), built on
[Textual](https://textual.textualize.io/) for the UI and
[LangGraph](https://langchain-ai.github.io/langgraph/)/[LangChain](https://python.langchain.com/)
for the agent loop and LLM integrations.

## Overview

mastermind runs a chat loop against a configurable LLM provider (Claude,
OpenAI, Gemini, Ollama, or a local llama.cpp server), with tool calling and
retrieval-based tool selection:

- **TUI** — a Textual app (Header/transcript/Input/Footer) with a slash-command
  registry and a `/model` dialog for picking a provider/model at runtime.
- **Agent loop** — a LangGraph graph: an embedding-backed retriever picks the
  top-2 most relevant tools for the latest message, the LLM is invoked with
  just that subset bound, and a `ToolNode` executes whatever it calls before
  looping back.
- **LLM/embeddings** — one factory per model type (`llm/`) wrapping LangChain's
  `init_chat_model`/`init_embeddings`, so adding a provider means adding an
  entry to a lookup table, not a new class.
- **Tracing** — optional [Langfuse](https://langfuse.com/) tracing: one trace
  per chat turn, grouped by session, active whenever `LANGFUSE_PUBLIC_KEY`/
  `LANGFUSE_SECRET_KEY` are set.

Architecture is layered so each piece only knows about the one below it:

```text
Textual TUI
     │
     ▼
AgentRuntime (agent/agent.py)
     │
     ▼
LangGraph (agent/graph_builder.py)
     │
     ▼
LLM / Tools (llm/, agent/tool.py)
```

The TUI never sees a raw LangGraph event or LangChain message — only the
`AgentEvent` types in `agent/events.py` (`AssistantStarted`, `AssistantToken`,
`AssistantCompleted`, `AgentError`).

## Project structure

```text
mastermind/
├── pyproject.toml
├── .env.example
├── src/
│   └── mastermind/
│       ├── cli.py                  # entry point: `mastermind` / `python -m mastermind`
│       ├── app.py                  # the Textual App
│       │
│       ├── agent/
│       │   ├── agent.py            # Agent: runs the graph for one turn, yields AgentEvents
│       │   ├── graph_builder.py    # the LangGraph graph (tool-selection -> llm -> tools loop)
│       │   ├── events.py           # AgentEvent types the TUI is allowed to see
│       │   └── tool.py             # tool definitions (web search, calculator)
│       │
│       ├── llm/
│       │   └── __init__.py         # build_chat_model / build_embedding factories
│       │
│       ├── tui/
│       │   ├── commands.py         # slash-command registry + dispatch
│       │   ├── screens/            # /model dialog
│       │   └── widgets/            # conversation transcript, message rendering
│       │
│       ├── config/
│       │   ├── settings.py         # ModelConfig/EmbeddingConfig/Config, load/save
│       │   └── paths.py            # config file location (~/.config/mastermind/)
│       │
│       └── observability/
│           └── tracing.py          # Langfuse CallbackHandler setup
│
└── tests/
    └── unit/
```

`commands/`, `skills/`, and `mcp/` from the long-term design aren't built yet
— slash commands are registered directly in `tui/commands.py` for now.

## Running it

Dependencies and the venv are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # install runtime deps
uv sync --group dev     # + pytest/ruff/pyright/textual-dev

uv run mastermind               # run the app
uv run python -m mastermind     # equivalent module entry point
```

### Configuration

Copy `.env.example` to `.env` for local dev (gitignored) and fill in what you
need — real environment variables work the same way in production.

```text
MASTERMIND_PROVIDER, MASTERMIND_MODEL, ...   # see CLAUDE.md for the full list
ANTHROPIC_API_KEY / OPENAI_API_KEY / OLLAMA_BASE_URL / LLAMA_CPP_BASE_URL
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
```

Model/embedding provider choice is otherwise picked interactively via `/model`
and persisted to `~/.config/mastermind/config.json` (override with
`MASTERMIND_CONFIG_PATH`).

### Slash commands

`/help`, `/model`, `/clear`, `/compact`, `/dump`, and `/exit` are implemented;
the rest (`/new`, `/provider`, `/tools`, `/skills`, `/sessions`, `/save`,
`/load`, `/config`) are registered as stubs for future work.

## Watching the Textual devtools log

Because the TUI takes over the whole terminal, `print()`/stdout doesn't work
for debugging — code (e.g. `graph_builder.py`'s tool-selection logging) uses
Textual's `log()` helper instead, which streams to a separate devtools
console instead of the app's screen.

1. In one terminal, start the console:

   ```bash
   uv run textual console
   ```

2. In another terminal, run the app in dev mode so it connects to that
   console (setting the `TEXTUAL=devtools` env var has the same effect, which
   is how the VS Code launch config wires it up):

   ```bash
   uv run textual run --dev mastermind.app:MastermindApp
   ```

Log calls anywhere in the app (e.g. the retrieved tool names for each turn)
will show up in the first terminal as you use the app in the second.

## Tests

```bash
uv run pytest
uv run pytest --cov=mastermind --cov-report=term-missing
```

## Lint, format, type-check

```bash
uv run ruff check .
uv run ruff format .
uv run pyright
```

See `CLAUDE.md` for the full development philosophy, build order, and
architecture conventions this project follows.
