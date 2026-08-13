from __future__ import annotations


def main() -> None:
    """Start the mastermind terminal application."""
    # Loaded first, before anything below can read an env var (e.g.
    # observability/tracing.py's LANGFUSE_* checks): `.env` fills in local
    # dev values, but never overrides a var already set in the real
    # environment (e.g. in production), which is `load_dotenv()`'s default.
    from dotenv import load_dotenv

    load_dotenv()

    from mastermind.app import MastermindApp

    app = MastermindApp()
    # App.run() takes over the terminal (alternate screen, raw input mode) and
    # blocks here until the app exits — the real-terminal counterpart to the
    # headless App.run_test() used in tests/unit/test_app.py.
    app.run()
