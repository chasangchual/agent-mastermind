"""The scrollable chat history: one `ChatMessage` per turn, plus plain status
lines for command output/errors — replacing the single `RichLog` transcript
so each chat turn can be styled by role and repainted in place for streaming.
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static

from mastermind.tui.widgets.message import ChatMessage


class Conversation(VerticalScroll):
    """`VerticalScroll` is a container that adds a scrollbar and clips its
    content to its own box — children just stack top-to-bottom in mount
    order, same as `compose()`'s yield order elsewhere, except children can
    be added/removed after the fact via `mount()`/`remove_children()`.
    """

    def add_message(self, role: str, text: str = "") -> ChatMessage:
        """Mount a new chat turn and return it (so callers can stream into it)."""
        message = ChatMessage(role, text)
        self.mount(message)
        # `scroll_end` (not `scroll_visible`) jumps the whole container to
        # its bottom — chat transcripts should track the latest turn, same
        # as a terminal or messaging app; `animate=False` since this fires
        # on every token while streaming, not just once per message.
        self.scroll_end(animate=False)
        return message

    def write_line(self, text: str) -> None:
        """A plain status/command-result line — Rich markup, not Markdown."""
        self.mount(Static(text, markup=True))
        self.scroll_end(animate=False)

    def clear(self) -> None:
        self.remove_children()
