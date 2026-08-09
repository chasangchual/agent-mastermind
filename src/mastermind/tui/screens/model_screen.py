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
    """A `ModalScreen` is a Textual `Screen` that overlays the current one
    instead of replacing it — the main screen (transcript + input) stays
    mounted underneath, dimmed, until this one is popped.

    `ModalScreen[ModelConfig | None]` is Textual's typed way of declaring
    "closing this screen hands back a ModelConfig or None". That value is
    whatever gets passed to `self.dismiss(...)` below, and it's delivered
    to whoever did `push_screen(ModelScreen(...), callback=...)` — that's
    `MastermindApp._on_model_configured` in app.py.
    """

    # Scoped to this screen only (Textual CSS cascades per-screen, not
    # globally). `align: center middle` centers the dialog box in the
    # terminal; `#dialog` gets a fixed width + border so it reads as a
    # floating dialog rather than filling the screen.
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
        # compose() runs once, right after this screen is pushed, and
        # builds its widget tree the same way MastermindApp.compose() does
        # for the main screen — see app.py for the general explanation of
        # why this is a generator and why layout needs no separate call.
        provider = self._current.provider if self._current else SUPPORTED_PROVIDERS[0]
        model = self._current.model if self._current else PROVIDER_MODELS[provider][0]
        is_known_model = model in PROVIDER_MODELS[provider]

        # `yield` inside a `with` block nests widgets under that container —
        # Vertical/Horizontal have no explicit "add child" call, membership
        # is just whatever gets yielded while their `with` block is open.
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
        # on_mount fires once these widgets are actually attached — the
        # earliest point it's safe to query/hide them (compose() is still
        # busy yielding while it runs). Sets visibility to match whichever
        # provider ended up preselected above.
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
        # `.display` controls whether Textual renders/lays out a widget at
        # all (a hidden widget takes no space, unlike CSS visibility) —
        # this is how the dialog shows only the field the provider needs.
        self.query_one("#base_url", Input).display = needs_base_url(provider)
        self.query_one("#api_key", Input).display = needs_api_key(provider)

    def _sync_custom_model_visibility(self, model: object) -> None:
        self.query_one("#custom_model", Input).display = model == _CUSTOM

    def on_select_changed(self, event: Select.Changed) -> None:
        # Textual routes every Select's change through this one handler —
        # the `on_<widget>_<event>` naming convention only distinguishes
        # widget *type* + event, not which instance fired it. `event.select`
        # is that instance, so `.id` is how the two Selects are told apart
        # (same pattern as `event.input` on Input.Submitted in app.py).
        if event.select.id == "provider":
            provider = event.value
            assert isinstance(provider, str)
            # set_options() replaces the model list for the new provider;
            # Textual resets the selection when options change, so the
            # value is set explicitly right after — that assignment itself
            # posts another Select.Changed, which is what updates the
            # custom-model field's visibility below without duplicating
            # that logic here.
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
