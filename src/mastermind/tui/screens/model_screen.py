"""Modal dialog for `/model`: pick a provider + model, then supply whatever
that provider needs (a base URL for local servers, an API key for hosted
ones), and hand the result back to whoever opened the dialog.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from mastermind.config.settings import (
    PROVIDER_MODELS,
    SUPPORTED_PROVIDERS,
    ModelConfig,
    needs_api_key,
    needs_base_url,
)

# Sentinel for the "type your own model name" option in the model Select —
# distinct from any real model name so it can never collide with one.
_CUSTOM = "__custom__"


def _model_options(provider: str) -> list[tuple[str, str]]:
    options = [(name, name) for name in PROVIDER_MODELS[provider]]
    options.append(("Custom…", _CUSTOM))
    return options


class ModelScreen(ModalScreen[ModelConfig | None]):
    """A `ModalScreen` overlays the current screen instead of replacing it —
    the main screen stays mounted underneath until this one is popped.

    `ModalScreen[ModelConfig | None]` declares that dismissing this screen
    hands back a `ModelConfig | None`, delivered via `self.dismiss(...)`
    below to `MastermindApp._on_model_configured` (app.py), which pushed it
    with `push_screen(ModelScreen(...), callback=...)`.
    """

    # CSS here is scoped to this screen only. `align: center middle` centers
    # the dialog; `#dialog` gets a fixed width + border to read as a floating
    # box rather than filling the screen.
    CSS = """
    ModelScreen {
        align: center middle;
    }

    #dialog {
        width: 50;
        height: auto;
        border: solid $accent;
        padding: 1 2;
    }

    #dialog Label {
        margin-top: 1;
    }

    #buttons {
        margin-top: 1;
        height: auto;
        align: right middle;
    }

    #buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, current: ModelConfig | None = None) -> None:
        super().__init__()
        # Preselect whatever's already configured, so re-opening /model to
        # tweak one field doesn't force retyping everything.
        self._current = current

    def compose(self) -> ComposeResult:
        provider = self._current.provider if self._current else SUPPORTED_PROVIDERS[0]
        model = self._current.model if self._current else PROVIDER_MODELS[provider][0]
        is_known_model = model in PROVIDER_MODELS[provider]

        # yield inside a `with` block nests that widget under the container —
        # Vertical/Horizontal membership is just whatever's yielded while
        # their `with` block is open.
        with Vertical(id="dialog"):
            yield Label("Provider")
            yield Select(
                [(p, p) for p in SUPPORTED_PROVIDERS],
                value=provider,
                allow_blank=False,
                id="provider",
            )
            yield Label("Model")
            yield Select(
                _model_options(provider),
                value=model if is_known_model else _CUSTOM,
                allow_blank=False,
                id="model",
            )
            yield Input(
                value="" if is_known_model else model,
                placeholder="Custom model name",
                id="custom_model",
            )
            yield Label("Base URL")
            yield Input(
                value=(self._current.base_url or "") if self._current else "",
                placeholder="http://localhost:11434",
                id="base_url",
            )
            yield Label("API key")
            yield Input(
                value=(self._current.api_key or "") if self._current else "",
                placeholder="sk-...",
                password=True,
                id="api_key",
            )
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", variant="primary", id="save")

    def on_mount(self) -> None:
        # Set visibility to match whichever provider ended up preselected.
        self._sync_field_visibility(self._provider_select.value)
        self._sync_custom_model_visibility(self._model_select.value)

    @property
    def _provider_select(self) -> Select[str]:
        return self.query_one("#provider", Select)

    @property
    def _model_select(self) -> Select[str]:
        return self.query_one("#model", Select)

    def _sync_field_visibility(self, provider: object) -> None:
        assert isinstance(provider, str)
        # `.display = False` skips layout entirely (unlike CSS visibility),
        # so hidden fields take no space.
        self.query_one("#base_url", Input).display = needs_base_url(provider)
        self.query_one("#api_key", Input).display = needs_api_key(provider)

    def _sync_custom_model_visibility(self, model: object) -> None:
        self.query_one("#custom_model", Input).display = model == _CUSTOM

    def on_select_changed(self, event: Select.Changed) -> None:
        # Both Selects route through this one handler (on_<widget>_<event>
        # only distinguishes type+event, not instance) — event.select.id
        # tells them apart.
        if event.select.id == "provider":
            provider = event.value
            assert isinstance(provider, str)
            # set_options() resets the selection, so it's set explicitly
            # right after — that assignment posts another Select.Changed,
            # which updates the custom-model visibility below on its own.
            self._model_select.set_options(_model_options(provider))
            self._model_select.value = PROVIDER_MODELS[provider][0]
            self._sync_field_visibility(provider)
        elif event.select.id == "model":
            self._sync_custom_model_visibility(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return

        provider = self._provider_select.value
        model_value = self._model_select.value
        assert isinstance(provider, str)
        assert isinstance(model_value, str)
        model = (
            self.query_one("#custom_model", Input).value
            if model_value == _CUSTOM
            else model_value
        )

        base_url = self.query_one("#base_url", Input).value or None
        api_key = self.query_one("#api_key", Input).value or None
        self.dismiss(
            ModelConfig(
                provider=provider,
                model=model,
                base_url=base_url if needs_base_url(provider) else None,
                api_key=api_key if needs_api_key(provider) else None,
            )
        )
