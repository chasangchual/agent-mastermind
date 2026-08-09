"""Thin factory turning a ModelConfig into a LangChain chat model.

Prefer LangChain's own `init_chat_model` where it already knows the
provider; llama.cpp gets a manual branch because it has no dedicated
LangChain integration — its server mode just speaks the OpenAI
chat-completions API, so ChatOpenAI pointed at it works directly.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from mastermind.config.settings import ModelConfig
from mastermind.exceptions import ProviderError

_INIT_CHAT_MODEL_PROVIDERS = {
    "claude": "anthropic",
    "openai": "openai",
    "gemini": "google_genai",
    "ollama": "ollama",
}


def build_chat_model(cfg: ModelConfig) -> BaseChatModel:
    if cfg.provider == "llama.cpp":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model,
            base_url=cfg.base_url or "http://localhost:8080/v1",
            api_key=cfg.api_key or "not-needed",
        )

    provider = _INIT_CHAT_MODEL_PROVIDERS.get(cfg.provider)
    if provider is None:
        raise ProviderError(f"Unsupported provider: {cfg.provider}")

    kwargs: dict[str, str] = {}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    return init_chat_model(cfg.model, model_provider=provider, **kwargs)
