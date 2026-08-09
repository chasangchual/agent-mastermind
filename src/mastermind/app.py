from __future__ import annotations

from typing import Any, ClassVar

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input

from mastermind.agent.events import AgentError, AssistantCompleted, AssistantToken
from mastermind.agent.runtime import AgentRuntime
from mastermind.config.settings import ModelConfig, load_config, save_config
from mastermind.tui.commands import run_command
from mastermind.tui.screens.model_screen import ModelScreen
from mastermind.tui.widgets.conversation import Conversation


class MastermindApp(App):
    """Terminal application for mastermind."""

    # `App` is Textual's root object: one instance per process, created once
    # in cli.py and started with `.run()`. Everything on screen lives inside it.
    # Class attributes (not __init__) are how Textual apps declare static config —
    # TITLE, plus things like CSS/BINDINGS/SCREENS if we add them later.
    TITLE = "mastermind"

    # BINDINGS maps a key to an "action" by name: pressing ctrl+q calls
    # `self.action_quit()`, ctrl+l calls `self.action_clear_transcript()`
    # (see below). Textual builds the method name itself as `action_<name>`,
    # so the string here ("quit", "clear_transcript") must match a method
    # that exists on this class. `action_quit` isn't defined below because
    # Textual's base `App` already provides it (it calls `self.exit()`,
    # which resolves to *this* class's override below since Python looks up
    # methods on the instance's actual class first) — every `App` can quit.
    # There's no built-in `action_exit` to bind to, which is why this isn't
    # named "exit": that name is reserved for the `exit()` method below.
    # The third tuple element (the "Exit"/"Clear" text) is only for display:
    # the Footer widget reads BINDINGS and renders it as a hint bar, so you
    # never have to hand-write "press ctrl+q to exit" anywhere.
    #
    # Declaring keys here — instead of, say, checking `event.key == "ctrl+q"`
    # inside a widget's own key-event handler — is Textual's recommended way
    # to keep every shortcut in one place, discoverable and in the Footer,
    # rather than scattered across widgets.
    #
    # `ClassVar` tells the type checker this is a plain class-level constant,
    # not a per-instance attribute Textual might reactively track (like a
    # `reactive[...]` field would be) — pyright otherwise flags a mutable
    # class attribute as a possible bug.
    BINDINGS: ClassVar = [
        ("ctrl+q", "quit", "Exit"),
        ("ctrl+l", "clear_transcript", "Clear"),
    ]

    # Textual widgets are styled with CSS (a subset of real CSS: same
    # selector/property idea, tailored to a terminal grid instead of a
    # browser). This can also live in an external `.tcss` file loaded via
    # `CSS_PATH`; inline `CSS` is fine while there's only one small rule.
    #
    # `1fr` is a "fraction" unit (like CSS Grid's `fr`): "give this widget 1
    # share of whatever vertical space is left over" after Header/Footer/
    # Input (which size themselves) take theirs — so Conversation stretches
    # to fill the remaining space and resizes automatically with the terminal.
    # `$accent` is a theme variable, not a literal color — Textual defines a
    # small palette of these per theme so widgets stay consistent if the
    # user switches themes.
    #
    # `border: solid $accent` (a shorthand for all four sides) drew a full
    # box, so every row had a "│" glued to its left/right edge. `border-top`
    # sets only that one side, giving a horizontal divider between Header
    # and Conversation with no vertical bars and no bottom line — Input
    # (right below Conversation) supplies its own top divider, so a
    # Conversation bottom border would just double up with it. Input gets
    # the equivalent treatment: its own `DEFAULT_CSS` uses a "tall" border
    # (thick/thin blocks on the *sides*, not just lines), which is what was
    # drawing verticals down the input box; `border: none` clears that
    # default first (a bare `border-top`/`border-bottom` here would only add
    # to it, not replace it), for both the unfocused and focused (`:focus`)
    # states — Input's own CSS sets a different border again on focus, so it
    # needs overriding separately or the verticals would reappear the moment
    # the box is focused. `$foreground-muted` (a translucent foreground, not
    # a literal grey hex) is Textual's own light-grey token — theme-aware
    # like `$accent` above, so it still looks right if the theme changes.
    # "solid" is already Textual's thinnest border style (a single-cell "─"
    # per row); there's no thinner variant, only heavier ones ("heavy",
    # "thick", "double", ...) — the lighter color is what reads as "thinner"
    # here, since a dim line has less visual weight than a bright one.
    #
    # Input's `background: $background` matches Conversation's background —
    # Conversation has no `background` rule of its own, so it's transparent
    # and just shows the Screen through it, which is `$background`; Input's
    # own `DEFAULT_CSS` instead sets `background: $surface` (a lighter
    # shade), which is what made the two boxes look like different blacks.
    #
    # `background-tint` is a separate compositing step, not `background`
    # itself — Input's own `DEFAULT_CSS` layers `background-tint: $foreground
    # 5%` on top *only* while focused, which is what was still lightening
    # the box on focus even with `background` pinned above; `0%` cancels
    # that layer so focusing leaves the background untouched.
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
        # Plain instance state, not a Textual `reactive` — nothing on
        # screen re-renders off it. `load_config()` returns whatever was
        # last saved via /model, or None on a fresh install.
        self._model_config: ModelConfig | None = load_config()
        # AgentRuntime (agent/runtime.py) is the seam to LangGraph — this
        # module never imports langchain/langgraph directly, per CLAUDE.md's
        # Architecture Layering. None until a model is configured, so the
        # app still runs (per the TUI requirement) with nothing selected yet.
        # Built in on_mount(), not here: building it can fail and report the
        # error via write_line(), which needs the widget tree mounted first
        # (see on_mount()'s docstring below for why __init__ is too early).
        self._runtime: AgentRuntime | None = None

    def _build_runtime(self, cfg: ModelConfig) -> None:
        try:
            self._runtime = AgentRuntime(cfg)
        except Exception as exc:  # noqa: BLE001 -- an unsupported/misconfigured provider must be reported, not crash startup
            self._runtime = None
            # `escape()` (not a plain f-string) because `exc`'s message is
            # untrusted content, not markup we wrote — provider SDK errors
            # (e.g. a Pydantic validation error) routinely contain literal
            # "[...]" text ("[type=string_type, input_type=dict]"), which
            # Rich's markup parser would otherwise try to read as a tag and
            # raise `MarkupError` on, crashing the app instead of reporting
            # the original error.
            self.write_line(
                f"[b red]Could not initialize {cfg.provider}: {escape(str(exc))}[/b red]"
            )
            self._refresh_status("Error")
            return
        self._refresh_status()

    def _refresh_status(self, state: str = "Ready") -> None:
        # `self.sub_title` is a Textual `App` reactive: `Header` (mounted in
        # compose()) watches it internally and re-renders itself whenever it
        # changes, so assigning the string here is all that's needed to
        # update what's shown on the right of the header bar.
        if self._model_config is None:
            self.sub_title = "No model configured"
        else:
            self.sub_title = (
                f"{self._model_config.provider}/{self._model_config.model} · {state}"
            )

    def open_model_dialog(self, current: ModelConfig | None) -> None:
        # This is the one real side effect in this module: it calls back into
        # whatever `ctx` was passed (the live App in normal use, `_FakeContext`
        # in the demo/tests) rather than touching the Conversation widget
        # directly — this function has no idea Textual exists.
        self.push_screen(ModelScreen(current), callback=self._on_model_configured)

    def _on_model_configured(self, cfg: ModelConfig | None) -> None:
        if cfg is None:
            self.write_line("[b red]Model selection canceled.[/b red]")
            return

        self._model_config = cfg
        save_config(cfg)
        self._build_runtime(cfg)
        # `cfg.provider` is always one of `SUPPORTED_PROVIDERS` (config/
        # settings.py), never user-typed — but `cfg.model` can be, via the
        # /model dialog's "Custom model name" field, so it gets the same
        # `escape()` treatment as the exception message above.
        self.write_line(f"[b]Model configured:[/b] {cfg.provider}/{escape(cfg.model)}")

    def compose(self) -> ComposeResult:
        # compose() builds the initial widget tree. Textual calls this once,
        # automatically, right after the app starts (also again if you mount a
        # new Screen) — we never call it ourselves.
        #
        # It's a generator: each `yield` adds one widget as a direct child of
        # this App/Screen, in the order yielded. Yielding (instead of
        # returning a list) lets Textual start mounting widgets as they're
        # produced rather than waiting for the whole tree to be built.
        #
        # There's no explicit layout call here: Header/Footer dock themselves
        # to the top/bottom edge (that's built into those widgets), and plain
        # widgets in between stack vertically in yield order by default —
        # Conversation then Input, top to bottom.
        #
        # Header renders `self.title` centered and `self.sub_title` on the
        # right (see `_refresh_status` below) — no separate widget needed to
        # show "provider/model · status" per the TUI Design mockup.
        yield Header()
        yield Conversation()
        yield Input(placeholder="Type a message and press Enter…")
        yield Footer()  # bottom bar showing active key bindings

    def on_mount(self) -> None:
        # `on_mount` is one of Textual's lifecycle hooks: it fires once the
        # widget tree from compose() is actually attached and ready, which is
        # the earliest safe point to reach into it (compose() itself is still
        # busy yielding widgets, so querying the tree there wouldn't work).
        #
        # `query_one` does a CSS-selector-style lookup over the mounted tree —
        # here, "the Input widget" (by type, since there's exactly one). It
        # raises if zero or more-than-one match, so a typo/duplicate fails
        # loudly instead of silently doing nothing.
        self.query_one(Input).focus()
        self._refresh_status()
        # Deferred from __init__ (see the comment there): if a model was
        # already configured from a previous run, build its AgentRuntime now
        # that write_line()'s query_one(Conversation) has something mounted
        # to find.
        if self._model_config is not None:
            self._build_runtime(self._model_config)

    def action_clear_transcript(self) -> None:
        # This is what BINDINGS' ("ctrl+l", "clear_transcript", ...) entry
        # actually calls. Kept as a one-line wrapper around clear_transcript()
        # below so both the keybinding *and* the `/clear` slash command
        # (dispatched through CommandContext) share one implementation
        # instead of two copies of "clear the conversation" existing.
        self.clear_transcript()

    # --- CommandContext protocol, so tui/commands.py handlers stay decoupled
    # from this widget tree (see CommandContext in commands.py). Slash-command
    # handlers are plain functions that take this narrow interface instead of
    # the whole `App`, so they can be unit-tested with a fake object instead
    # of booting Textual. These three methods are what makes MastermindApp
    # satisfy that Protocol — nothing declares the relationship explicitly
    # (Protocols are structural/"duck-typed"), pyright just checks the method
    # shapes line up wherever `self` is passed as a CommandContext. ---
    def write_line(self, text: str) -> None:
        self.query_one(Conversation).write_line(text)

    def clear_transcript(self) -> None:
        self.query_one(Conversation).clear()

    def exit(self, *args: Any, **kwargs: Any) -> None:
        # Named `exit` (not `action_exit`) purely to match the CommandContext
        # Protocol's method name — this is unrelated to Textual's built-in
        # `action_exit` that the ctrl+q binding above calls. This method
        # *overrides* `App.exit()` (same name, same class), so calling
        # `self.exit()` here would call itself again, not the base class —
        # infinite recursion, not a shutdown. `super().exit()` is what
        # actually reaches Textual's real shutdown: it tears down the app
        # and returns control to whatever called `app.run()` in cli.py.
        # `*args/**kwargs` (rather than a zero-arg override) just forwards
        # App.exit()'s own optional result/return_code/message straight
        # through — CommandContext only ever calls this with none of them.
        super().exit(*args, **kwargs)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Textual dispatches widget events to methods named
        # `on_<snake_case_widget_class>_<snake_case_event_name>` on any
        # ancestor in the tree — Input posts an `Input.Submitted` message
        # when the user presses Enter in it, so `on_input_submitted` is found
        # and called automatically here on the App. (We never call this
        # method ourselves, same as compose()/on_mount() above.) `event` is
        # that message: `event.value` is the text that was in the box,
        # `event.input` is the Input instance that fired it, useful here
        # since it's how we clear the box back out below.
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text.startswith("/"):
            # Slash commands are parsed/dispatched in tui/commands.py, kept
            # separate from ordinary chat text per the TUI requirement that
            # commands get dedicated, independently testable handlers rather
            # than being handled inline as just another chat message.
            result = run_command(text, self)
            # An empty message means the handler already caused everything
            # it's going to (e.g. `/exit` tearing down the app via
            # ctx.exit()) and has nothing left to report — most visibly,
            # trying to mount a status line into a screen that's already
            # mid-shutdown would raise, not just be pointless.
            if result.message:
                style = "b" if result.ok else "b red"
                # write_line()'s Conversation.write_line() renders this via a
                # `Static`, which (like RichLog, Label, etc.) accepts Rich's
                # `[tag]...[/tag]` markup for inline styling — "b" is bold,
                # "b red" is bold+red; this isn't Textual-specific, it's the
                # Rich library Textual is built on top of for all its text
                # rendering. Command output uses this plain-markup path
                # rather than a `ChatMessage`'s Markdown body since it's a
                # status line, not a chat turn. `escape()` because
                # `result.message` can echo raw user input — e.g. the
                # "Unknown command: /{name}" message embeds whatever the
                # user typed after the slash — and no handler currently
                # relies on it containing real markup, so it's always safe
                # to escape.
                self.write_line(f"[{style}]{escape(result.message)}[/{style}]")
            return

        self.query_one(Conversation).add_message("user", text)
        if self._runtime is None:
            self.write_line("[b red]No model configured. Run /model first.[/b red]")
            return

        # `_run_agent` is an async method decorated with `@work` below, so
        # calling it here just schedules it as a background asyncio Task
        # Textual manages — it does NOT run inline and does NOT block this
        # handler. That matters because it awaits a real network call to the
        # LLM provider: Textual runs one asyncio event loop for the whole
        # UI, so a blocking/slow call made directly inside an event handler
        # (instead of via `@work`) would freeze keystrokes and redraws for
        # everyone until it returned.
        self._run_agent(text)

    @work(exclusive=True)
    async def _run_agent(self, text: str) -> None:
        # `@work(exclusive=True)` (from `textual.work`) is what makes this
        # coroutine run as a managed background task instead of a plain
        # method call: Textual schedules it on the same event loop but
        # doesn't await it from the caller, and `exclusive=True` cancels any
        # previous `_run_agent` task first — so submitting a new message
        # while one is still streaming can't run two replies concurrently
        # into the same message widget.
        assert self._runtime is not None
        # One ChatMessage is mounted up front and repainted in place as
        # tokens arrive (via update_text(), see tui/widgets/message.py) —
        # that's the streaming behavior the old `#streaming` Static hack
        # used to provide, now handled by the message widget itself.
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
