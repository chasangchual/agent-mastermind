from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input

from mastermind.agent.agent_runtime import AgentRuntime
from mastermind.agent.events import AgentError, AssistantCompleted, AssistantToken
from mastermind.config.settings import Config, ModelConfig, load_config, save_config
from mastermind.tui.commands import run_command
from mastermind.tui.screens.model_screen import ModelScreen
from mastermind.tui.widgets.conversation import Conversation

# Textual app lifecycle: __init__ -> compose() (build widget tree) ->
# on_mount() (tree attached) -> event loop (input/timers/workers) -> on_unmount().
# https://textual.textualize.io/guide/app-lifecycle/


class MastermindApp(App):
    """Terminal application for mastermind."""

    # `App` is Textual's root object, one per process. Class attributes like
    # TITLE/CSS/BINDINGS are how an App declares static config.
    TITLE = "mastermind"

    # BINDINGS maps a key to an `action_<name>` method: ctrl+q -> action_quit
    # (built into App), ctrl+l -> action_clear_transcript below. The third
    # element is just the label Footer displays for it.
    BINDINGS: ClassVar = [
        ("ctrl+q", "quit", "Exit"),
        ("ctrl+l", "clear_transcript", "Clear"),
    ]

    def action_clear_transcript(self) -> None:
        self.clear_transcript()

    # Textual CSS: same selector/property idea as browser CSS, scaled to a
    # terminal grid. `1fr` gives Conversation the vertical space left over
    # after Header/Footer/Input size themselves. `$accent`/`$foreground-muted`/
    # `$background` are theme tokens, not literal colors.
    #
    # Input's own DEFAULT_CSS draws a "tall" (blocky) border and adds a
    # background-tint on focus; both are overridden here (`border: none` then
    # a plain top/bottom border, `background-tint: $foreground 0%` on focus)
    # so it reads as a flat divider matching Conversation instead of a boxed
    # widget that changes color when focused.
    CSS = """
    Conversation {
        height: 1fr;
        border-top: solid $accent;
    }

    Input {
        border: none;
        border-top: solid $foreground-muted;
        border-bottom: solid $foreground-muted;
        background: $background;
    }

    Input:focus {
        border: none;
        border-top: solid $foreground-muted;
        border-bottom: solid $foreground-muted;
        background-tint: $foreground 0%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # `load_config()` returns whatever was last saved via /model, or
        # defaults on a fresh install.
        self._config: Config = load_config()
        # AgentRuntime (agent/agent_runtime.py) is the seam to LangGraph —
        # this module never imports langchain/langgraph directly, per
        # CLAUDE.md's Architecture Layering. Built in on_mount(), not here,
        # since building it can fail and report the error via write_line(),
        # which needs the widget tree mounted first.
        self._runtime: AgentRuntime | None = None

    def on_mount(self) -> None:
        # on_mount fires once compose()'s widget tree is attached — the
        # earliest point it's safe to query it (compose() is still busy
        # yielding while it runs).
        self.query_one(Input).focus()
        self._refresh_status()
        # Deferred from __init__: build the runtime now that a model was
        # already configured from a previous run, once write_line()'s
        # query_one(Conversation) has something mounted to find.
        if self._config.model_config is not None:
            self._build_runtime(self._config)

    def compose(self) -> ComposeResult:
        # compose() is a generator Textual calls once to build the initial
        # widget tree; each yield mounts one child, in order. Header/Footer
        # dock themselves to the top/bottom edge; Conversation and Input
        # stack vertically between them. Header shows `self.title` centered
        # and `self.sub_title` on the right (see _refresh_status below).
        yield Header()
        yield Conversation()
        yield Input(placeholder="Type a message and press Enter…")
        yield Footer()

    def _build_runtime(self, cfg: Config) -> None:
        assert cfg.model_config is not None
        try:
            self._runtime = AgentRuntime(cfg)
        except Exception as exc:  # noqa: BLE001 -- an unsupported/misconfigured provider must be reported, not crash startup
            self._runtime = None
            # `escape()` because `exc`'s message is untrusted (e.g. a
            # Pydantic error containing literal "[...]" text), which Rich's
            # markup parser would otherwise try to read as a tag.
            self.write_line(
                f"[b red]Could not initialize {cfg.model_config.provider}: {escape(str(exc))}[/b red]"
            )
            self._refresh_status("Error")
            return
        self._refresh_status()

    def _refresh_status(self, state: str = "Ready") -> None:
        # `self.sub_title` is a Textual reactive that Header watches, so
        # assigning it here is all that's needed to update the header bar.
        model_config = self._config.model_config
        if model_config is None:
            self.sub_title = "No model configured"
        else:
            self.sub_title = f"{model_config.provider}/{model_config.model} · {state}"

    def open_model_dialog(self, current: ModelConfig | None) -> None:
        self.push_screen(ModelScreen(current), callback=self._on_model_configured)

    def _on_model_configured(self, newModelCfg: ModelConfig | None) -> None:
        if newModelCfg is None:
            self.write_line("[b red]Model selection canceled.[/b red]")
            return

        self._config = replace(self._config, model_config=newModelCfg)
        save_config(self._config)
        self._build_runtime(self._config)
        # `newModelCfg.model` can be user-typed (the /model dialog's "Custom model
        # name" field), so it gets the same escape() treatment as above.
        self.write_line(f"[b]Model configured:[/b] {newModelCfg.provider}/{escape(newModelCfg.model)}")

    # --- CommandContext protocol (see commands.py): handlers take this
    # narrow interface instead of the whole App so they're unit-testable
    # without booting Textual. Nothing declares the relationship explicitly —
    # Protocols are structural, pyright just checks the method shapes match. ---
    def write_line(self, text: str) -> None:
        self.query_one(Conversation).write_line(text)

    def clear_transcript(self) -> None:
        self.query_one(Conversation).clear()

    def exit(self, *args: Any, **kwargs: Any) -> None:
        # Named `exit` to match CommandContext's method name, not Textual's
        # `action_exit`. This overrides App.exit() itself, so `super().exit()`
        # (not `self.exit()`) is what actually reaches Textual's shutdown.
        super().exit(*args, **kwargs)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Textual dispatches an Input's `Input.Submitted` message (posted on
        # Enter) to any ancestor method named `on_input_submitted` — that
        # naming convention is why this is found and called automatically.
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text.startswith("/"):
            result = run_command(text, self)
            # Empty message means the handler already did everything it's
            # going to (e.g. /exit tearing down the app) — most visibly,
            # writing a status line into a screen mid-shutdown would raise.
            if result.message:
                style = "b" if result.ok else "b red"
                self.write_line(f"[{style}]{escape(result.message)}[/{style}]")
            return

        self.query_one(Conversation).add_message("user", text)
        if self._runtime is None:
            self.write_line("[b red]No model configured. Run /model first.[/b red]")
            return

        # `@work` below schedules this as a background asyncio task instead
        # of running it inline — needed because it awaits a network call, and
        # Textual runs one event loop for the whole UI, so a blocking call
        # here would freeze keystrokes/redraws until it returned.
        self._run_agent(text)

    @work(exclusive=True)
    async def _run_agent(self, text: str) -> None:
        # `exclusive=True` cancels any previous `_run_agent` task first, so
        # submitting a new message while one is still streaming can't run two
        # replies concurrently into the same message widget.
        assert self._runtime is not None
        # One ChatMessage is mounted up front and repainted in place as
        # tokens arrive via update_text() (see tui/widgets/message.py).
        message = self.query_one(Conversation).add_message("assistant")
        self._refresh_status("Thinking…")
        buffer = ""
        async for event in self._runtime.astream(text):
            if isinstance(event, AssistantToken):
                buffer += event.text
                await message.update_text(buffer)
            elif isinstance(event, AssistantCompleted):
                await message.update_text(event.text)
            elif isinstance(event, AgentError):
                await message.update_text(f"**Error:** {event.error}")
        self._refresh_status()
