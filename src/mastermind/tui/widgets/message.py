"""A single chat turn: a role label above its content.

Kept as its own mounted widget instead of a `RichLog` line so a turn can be
targeted and repainted individually — that's what lets a streaming assistant
reply update in place via `update_text()` rather than growing one new log
line per token.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Markdown, Static

# (label, color) per role; anything else falls back to the role name as-is.
_ROLES: dict[str, tuple[str, str]] = {
    "user": ("User", "$text"),
    "assistant": ("Assistant", "$accent"),
}


class ChatMessage(Vertical):
    """A `Vertical` (not a plain compose() of two top-level widgets) so one
    `ChatMessage` mounts as a single unit per turn — `Conversation` below
    mounts/queries one of these per message rather than juggling loose
    label/body widget pairs itself.
    """

    DEFAULT_CSS = """
    ChatMessage {
        height: auto;
        margin-bottom: 1;
    }
    """

    def __init__(self, role: str, text: str = "") -> None:
        super().__init__()
        self._role = role
        self._initial_text = text

    def compose(self) -> ComposeResult:
        label, color = _ROLES.get(self._role, (self._role.title(), "$text"))
        # Static's Rich markup (`[b color]...[/b color]`) is unrelated to the
        # Markdown body below it — this is just a bold, colored role label.
        yield Static(f"[b {color}]{label}[/b {color}]")
        # `Markdown` (not `Static`/`RichLog`) is what gives the TUI Design
        # requirement of rendered code blocks/lists/etc. in message bodies —
        # RichLog and Static only understand Rich markup, not CommonMark.
        yield Markdown(self._initial_text)

    async def update_text(self, text: str) -> None:
        # Markdown.update() re-renders the whole body from `text` each call —
        # fine at chat-message length/frequency, nothing to optimize here.
        await self.query_one(Markdown).update(text)
