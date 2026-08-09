"""Filesystem local mastermind reads/writes, overridable via env vars."""

from __future__ import annotations

import os
from pathlib import Path


def config_file_path() -> Path:
    override = os.environ.get("MASTERMIND_CONFIG_PATH")
    if override:
        return Path(override)
    return Path.home() / ".config" / "mastermind" / "config.json"
