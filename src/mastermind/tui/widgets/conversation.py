"""The scrollable chat history: one `ChatMessage` per turn, plus plain status
lines for command output/errors — replacing the single `RichLog` transcript
so each chat turn can be styled by role and repainted in place for streaming.
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static

from mastermind.tui.widgets.message import ChatMessage


class Conversation(VerticalScroll):
    """`VerticalScroll` adds a scrollbar and clips content to its box;
    children stack top-to-bottom and can be added/removed after mount via
    `mount()`/`remove_children()`.
    """

    def add_message(self, role: str, text: str = "") -> ChatMessage:
        """Mount a new chat turn and return it (so callers can stream into it)."""
        message = ChatMessage(role, text)
        self.mount(message)
        # animate=False since this fires on every streamed token, not once.
        self.scroll_end(animate=False)
        return message

    def write_line(self, text: str, *, markup: bool = True) -> None:
        """A plain status/command-result line — Rich markup, not Markdown."""
        self.mount(Static(text, markup=markup))
        self.scroll_end(animate=False)

    def clear(self) -> None:
        self.remove_children()
