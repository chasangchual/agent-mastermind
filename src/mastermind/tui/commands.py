"""Slash command parsing and dispatch, kept separate from the TUI widget tree.

Handlers only see a `CommandContext` (a Protocol), not the Textual `App` —
so they're callable and testable without rendering anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class CommandContext(Protocol):
    """Side effects a command handler is allowed to perform.

    This is a `typing.Protocol`, not an ABC you inherit from — any object
    with these three methods satisfies it automatically ("structural typing",
    same idea as Go interfaces). `MastermindApp` in app.py never says
    "I implement CommandContext" anywhere; it just happens to define
    `write_line`/`clear_transcript`/`quit` with matching signatures, and
    pyright accepts passing `self` wherever a `CommandContext` is expected.
    The `demo()` below exploits the same thing with a plain `_FakeContext`
    class that has nothing to do with Textual at all.

    The `...` bodies mean "no implementation here, this is only a shape
    description" — Protocol methods are never actually called on this class.
    """

    def write_line(self, text: str) -> None: ...
    def clear_transcript(self) -> None: ...
    def quit(self) -> None: ...


@dataclass(frozen=True)
class CommandResult:
    """What a command handler reports back, independent of how it's displayed.

    `frozen=True` makes instances immutable and gives free `__eq__`/`__hash__`
    based on the field values — that's what lets the demo/tests below compare
    results with plain `==` instead of checking each field by hand.
    """

    ok: bool
    message: str


# A type alias, not a runtime object: this just gives a name to "a function
# shaped like (args, ctx) -> CommandResult" so the COMMANDS dict and function
# signatures below don't have to spell that shape out repeatedly.
CommandHandler = Callable[[list[str], CommandContext], CommandResult]


def _help(args: list[str], ctx: CommandContext) -> CommandResult:
    names = ", ".join(f"/{name}" for name in sorted(COMMANDS))
    return CommandResult(True, f"Available commands: {names}")


def _clear(args: list[str], ctx: CommandContext) -> CommandResult:
    # This is the one real side effect in this module: it calls back into
    # whatever `ctx` was passed (the live App in normal use, `_FakeContext`
    # in the demo/tests) rather than touching a RichLog directly — this
    # function has no idea Textual exists.
    ctx.clear_transcript()
    return CommandResult(True, "Transcript cleared.")


def _quit(args: list[str], ctx: CommandContext) -> CommandResult:
    ctx.quit()
    return CommandResult(True, "Quitting.")


def _not_implemented(args: list[str], ctx: CommandContext) -> CommandResult:
    return CommandResult(False, "Not implemented yet.")


# ponytail: most of these are stubs until the pieces they depend on exist
# (agent loop for /new, LLM factory for /model+/provider, tools/registry.py
# for /tools, skills/registry.py for /skills, sessions/manager.py for
# /sessions+/save+/load, config/settings.py for /config).
#
# This dict is the whole "dedicated handler per command" requirement: adding
# a new slash command later means adding one function above plus one entry
# here — nothing in app.py or the dispatch logic below needs to change.
COMMANDS: dict[str, CommandHandler] = {
    "help": _help,
    "clear": _clear,
    "quit": _quit,
    "new": _not_implemented,
    "model": _not_implemented,
    "provider": _not_implemented,
    "tools": _not_implemented,
    "skills": _not_implemented,
    "sessions": _not_implemented,
    "save": _not_implemented,
    "load": _not_implemented,
    "config": _not_implemented,
}


def run_command(line: str, ctx: CommandContext) -> CommandResult:
    """Parse and dispatch a slash command line, e.g. '/model gpt-4'.

    `line[1:]` strips the leading "/", then `.split()` breaks on whitespace:
    "model gpt-4".split() -> ["model", "gpt-4"]. The starred assignment
    `name, *args = [...]` takes the first element as `name` and collects
    everything else into the `args` list ("gpt-4" and any further words),
    the same destructuring you'd use to peel one item off the front of any
    sequence.
    """
    name, *args = line[1:].split()
    handler = COMMANDS.get(name)
    if handler is None:
        return CommandResult(False, f"Unknown command: /{name}")
    return handler(args, ctx)


def demo() -> None:
    # A minimal stand-in for CommandContext with no Textual/App involved at
    # all — proof that command dispatch is testable without rendering the
    # TUI, per the "testable without rendering the full TUI" requirement.
    class _FakeContext:
        def __init__(self) -> None:
            self.lines: list[str] = []
            self.cleared = False
            self.quit_called = False

        def write_line(self, text: str) -> None:
            self.lines.append(text)

        def clear_transcript(self) -> None:
            self.cleared = True

        def quit(self) -> None:
            self.quit_called = True

    ctx = _FakeContext()
    help_result = run_command("/help", ctx)
    assert (
        help_result.ok
        and "/model" in help_result.message
        and "/clear" in help_result.message
    )
    assert run_command("/clear", ctx) == CommandResult(True, "Transcript cleared.")
    assert ctx.cleared is True
    assert run_command("/nope", ctx) == CommandResult(False, "Unknown command: /nope")
    assert run_command("/model gpt-4", ctx).ok is False
    print("commands.py demo OK")


if __name__ == "__main__":
    demo()
