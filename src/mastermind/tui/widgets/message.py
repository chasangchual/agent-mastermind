"""A single chat turn: role conveyed by background color, not a text label.

Kept as its own mounted widget instead of a `RichLog` line so a turn can be
targeted and repainted individually — that's what lets a streaming assistant
reply update in place via `update_text()` rather than growing one new log
line per token.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Markdown

# Roles that get a background color; anything else (including "assistant")
# mounts with no override, so it shows the Conversation area's own background
# via Widget's default `background: transparent`.
_STYLED_ROLES = ("user",)


class ChatMessage(Vertical):
    """A `Vertical` so the body mounts as its own unit per turn, letting
    `Conversation` mount/query a single widget per message.

    `Widget`'s own default CSS sets `background: transparent`, so the body
    `Markdown` (which doesn't set its own background) paints directly over
    whatever background `-user` puts on this container — or, for any
    unstyled role, straight through to the Conversation area behind it.
    """

    DEFAULT_CSS = """
    ChatMessage {
        height: auto;
        margin-bottom: 1;
        padding: 0 0;
    }

    ChatMessage.-user {
        background: #404040;
        color: white;
    }

    /* Markdown sets its own `color: $foreground` in its DEFAULT_CSS, which
       otherwise wins over the parent's `color: white` above. */
    ChatMessage.-user Markdown {
        color: white;
    }

    /* MarkdownParagraph has a built-in `margin: 0 0 1 0` (see Textual's
       _markdown.py) that applies even to a lone/last paragraph — since
       ChatMessage's height is auto, that row gets absorbed into this box's
       own painted background instead of showing as the gap *between*
       messages (that gap already comes from ChatMessage's margin-bottom
       above). */
    ChatMessage MarkdownParagraph {
        margin: 0 !important;
    }
    """

    def __init__(self, role: str, text: str = "") -> None:
        super().__init__()
        self._role = role
        self._initial_text = text
        if role in _STYLED_ROLES:
            self.add_class(f"-{role}")

    def compose(self) -> ComposeResult:
        # Markdown (not Static/RichLog) renders CommonMark — code blocks,
        # lists, etc. — in the message body.
        yield Markdown(self._initial_text)

    async def update_text(self, text: str) -> None:
        # Re-renders the whole body each call; fine at chat-message frequency.
        await self.query_one(Markdown).update(text)
