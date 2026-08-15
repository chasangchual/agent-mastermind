"""App configuration: known providers/models and on-disk persistence.

Values come from what the user picks in the /model dialog (tui/screens/
model_screen.py), not env vars — this module just holds the provider/model
catalog and the load/save side of the persisted Config.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from mastermind.config.paths import config_file_path


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class Config:
    model_config: ModelConfig | None = None
    draw_mermaid: bool = False
    prompt_log: bool = False
    max_iterations: int | None = None  # None = no limit, else max iterations per turn
    compact_max_token: int = 4096


# Local/self-hosted providers need a base URL; hosted APIs need an API key.
# Drives which Input field the dialog shows for a given provider.
_URL_PROVIDERS = ("ollama", "llama.cpp")

# Curated choices shown in the picker - not exhaustive, the dialog alws offers a 'Custom .." option for anything not listed here
PROVIDER_MODELS: dict[str, tuple[str, ...]] = {
    "claude": (
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
    ),
    "openai": (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5-pro",
        "gpt-5.5",
    ),
    "gemini": ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-pro"),
    "ollama": ("llama3", "qwen2.5", "mistral"),
    "llama.cpp": ("local-model",),
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_MODELS)


def needs_base_url(provider: str) -> bool:
    return provider in _URL_PROVIDERS


def needs_api_key(provider: str) -> bool:
    return provider not in _URL_PROVIDERS


def load_config() -> Config:
    """The persisted app config, or defaults on a fresh install."""
    path = config_file_path()
    if not path.exists():
        return Config()

    data = json.loads(path.read_text())
    if "model_config" not in data:  # pre-Config file: bare ModelConfig fields
        return Config(model_config=ModelConfig(**data))

    model_config = data["model_config"]
    return Config(
        model_config=ModelConfig(**model_config) if model_config else None,
        draw_mermaid=data.get("draw_mermaid", False),
        prompt_log=data.get("prompt_log", False),
    )


def save_config(config: Config) -> None:
    """Persist `config` so it survives restarts.

    May contain an API key in plain text, so the file is written user-only
    (0600) — that keeps it out of other local accounts' reach, it is not
    encryption. Use a real secrets manager if that's not enough.
    """
    path = config_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2))
    path.chmod(0o600)
