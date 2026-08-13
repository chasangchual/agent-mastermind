import pytest

from mastermind.app import MastermindApp
from mastermind.config.settings import Config, ModelConfig


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


# write_line() renders through a `Static(markup=True)` (see
# tui/widgets/conversation.py), which parses "[...]" as Rich markup tags —
# any of these three call sites embedding *untrusted* text (an exception
# message, a user-typed model name, raw user input echoed back) into a
# markup-tagged string will raise `MarkupError` and crash the app if that
# text happens to contain unescaped brackets, e.g. a Pydantic error's
# "[type=string_type, input_type=dict]". None of these should raise.
@pytest.mark.asyncio
async def test_bracketed_dynamic_text_does_not_crash_markup_rendering() -> None:
    app = MastermindApp()
    async with app.run_test() as pilot:
        # An init failure whose exception message contains brackets.
        bad_cfg = ModelConfig(provider="gemini", model="gemini-2.5-flash", api_key=None)
        app._build_runtime(Config(model_config=bad_cfg))

        # A slash command containing brackets.
        await pilot.click("Input")
        await pilot.press(*"/foo[bar]", "enter")

        # A custom model name containing brackets.
        app._on_model_configured(
            ModelConfig(provider="ollama", model="weird[model]", base_url="http://x")
        )
