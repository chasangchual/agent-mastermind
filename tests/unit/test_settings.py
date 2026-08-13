"""Runnable check for config persistence: defaults, round-trip, and loading a
pre-Config config.json (bare ModelConfig fields, no "model_config" key).
"""

import json

from mastermind.config.settings import Config, ModelConfig, load_config, save_config


def test_load_config_defaults_on_fresh_install() -> None:
    assert load_config() == Config()


def test_save_then_load_round_trips() -> None:
    config = Config(
        model_config=ModelConfig(provider="ollama", model="llama3"),
        draw_mermaid=True,
        prompt_log=True,
    )
    save_config(config)
    assert load_config() == config


def test_load_config_migrates_legacy_flat_model_config(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"provider": "ollama", "model": "llama3"}))
    monkeypatch.setenv("MASTERMIND_CONFIG_PATH", str(path))

    assert load_config() == Config(model_config=ModelConfig(provider="ollama", model="llama3"))
