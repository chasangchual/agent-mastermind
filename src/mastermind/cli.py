from __future__ import annotations


def main() -> None:
    """Start the mastermind terminal application."""
    from mastermind.app import MastermindApp

    app = MastermindApp()
    # App.run() takes over the terminal (alternate screen, raw input mode) and
    # blocks here until the app exits — the real-terminal counterpart to the
    # headless App.run_test() used in tests/unit/test_app.py.
    app.run()
