import pytest

from mastermind.app import MastermindApp


# Textual apps normally take over the whole terminal, which a test can't do.
# `App.run_test()` is the framework's answer: it drives the app in a
# "headless" mode (no real terminal, no blocking event loop takeover) inside
# an async context manager, and hands back a Pilot you could use to simulate
# key presses / clicks if the test needed to. Because it's async, the test
# function must be async too — hence pytest-asyncio (configured via
# asyncio_mode = "auto" in pyproject.toml, so no per-test marker needed).
@pytest.mark.asyncio
async def test_app_composes_and_runs() -> None:
    app = MastermindApp()
    async with app.run_test():
        # Inside the `async with` block the app has mounted and is "running";
        # once the block exits, Textual shuts it down automatically.
        assert app.title == "mastermind"
