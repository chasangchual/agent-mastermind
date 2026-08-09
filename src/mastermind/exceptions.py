class MastermindError(Exception):
    """Base exception for expected mastermind failures."""


class ProviderError(MastermindError):
    """Raised when an LLM provider operation fails."""
