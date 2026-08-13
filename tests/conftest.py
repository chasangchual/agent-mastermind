"""Shared test fixtures.

Any test that boots MastermindApp or calls save_config()/load_config()
touches config.config_file_path(), which defaults to the real
~/.config/mastermind/config.json. Autouse + session-independent: this runs
for every test in the suite so no test (present or future) can clobber the
developer's actual saved config, without each test file needing its own copy
of this fixture.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_config_path(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MASTERMIND_CONFIG_PATH", str(tmp_path / "config.json"))
