from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from mastermind.tui.commands import run_command


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
    # Textual's base `App` already provides it — every `App` can quit.
    # The third tuple element (the "Quit"/"Clear" text) is only for display:
    # the Footer widget reads BINDINGS and renders it as a hint bar, so you
    # never have to hand-write "press ctrl+q to quit" anywhere.
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
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_transcript", "Clear"),
    ]

    # Textual widgets are styled with CSS (a subset of real CSS: same
    # selector/property idea, tailored to a terminal grid instead of a
    # browser). This can also live in an external `.tcss` file loaded via
    # `CSS_PATH`; inline `CSS` is fine while there's only one small rule.
    #
    # `1fr` is a "fraction" unit (like CSS Grid's `fr`): "give this widget 1
    # share of whatever vertical space is left over" after Header/Footer/
    # Input (which size themselves) take theirs — so RichLog stretches to
    # fill the remaining space and resizes automatically with the terminal.
    # `$accent` is a theme variable, not a literal color — Textual defines a
    # small palette of these per theme so widgets stay consistent if the
    # user switches themes.
    CSS = """
    RichLog {
        height: 1fr;
        border: solid $accent;
    }
    """

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
        # RichLog then Input, top to bottom.
        yield Header()  # top bar showing TITLE + a clock
        # `id="transcript"` gives this specific RichLog a CSS/query handle —
        # useful once there's more than one RichLog on screen (e.g. a
        # separate debug pane); right now `query_one(RichLog)` below matches
        # by type alone since it's the only one, but the id is there for when
        # that stops being true.
        yield RichLog(wrap=True, highlight=True, id="transcript")
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

    def action_clear_transcript(self) -> None:
        # This is what BINDINGS' ("ctrl+l", "clear_transcript", ...) entry
        # actually calls. Kept as a one-line wrapper around clear_transcript()
        # below so both the keybinding *and* the `/clear` slash command
        # (dispatched through CommandContext) share one implementation
        # instead of two copies of "clear the RichLog" existing.
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
        self.query_one(RichLog).write(text)

    def clear_transcript(self) -> None:
        self.query_one(RichLog).clear()

    def quit(self) -> None:
        # Named `quit` (not `action_quit`) purely to match the CommandContext
        # Protocol's method name — this is unrelated to Textual's built-in
        # `action_quit` that the ctrl+q binding above calls. `self.exit()` is
        # Textual's real shutdown call: it tears down the app and returns
        # control to whatever called `app.run()` in cli.py.
        self.exit()

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
            style = "b" if result.ok else "b red"
            # RichLog (and Static, Log, etc.) accept Rich's `[tag]...[/tag]`
            # markup for inline styling — "b" is bold, "b red" is bold+red;
            # this isn't Textual-specific, it's the Rich library Textual is
            # built on top of for all its text rendering.
            self.write_line(f"[{style}]{result.message}[/{style}]")
            return

        log = self.query_one(RichLog)
        log.write(f"[b]you:[/b] {text}")
        # ponytail: no agent loop yet, so this is a placeholder echo — replace
        # once agent/graph.py exists to route the message through the LLM.
        # That's also where streaming/cancellation/background workers belong
        # (Textual @work + asyncio, per the TUI requirements) — nothing to
        # cancel or stream until there's a real provider call running.
        #
        # Note for later: whatever replaces this must NOT block here. Textual
        # runs one asyncio event loop for the whole UI — a slow/blocking call
        # made directly inside an event handler like this one would freeze
        # keystrokes and redraws for everyone until it returns. The fix, when
        # this becomes a real LLM call, is Textual's `@work` decorator (runs
        # a method as a background async task) or plain `asyncio.create_task`,
        # not a direct blocking call in-line here.
        log.write("[b]mastermind:[/b] (agent loop not wired up yet)")
