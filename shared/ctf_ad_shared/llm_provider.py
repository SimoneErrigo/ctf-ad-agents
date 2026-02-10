# Multi-provider LLM manager

# Builds a Semantic Kernel "Kernel" with the best available chat-completion
# service, following the fallback chain: Anthropic -> OpenAI -> Ollama

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion

from ctf_ad_shared.config import LLMProviderConfig, Settings, get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _build_anthropic_service(config: LLMProviderConfig) -> ChatCompletionClientBase:
    logger.info("Using Anthropic provider — model=%s", config.model)
    return AnthropicChatCompletion(
        ai_model_id=config.model,
        api_key=config.api_key,
    )


def _build_openai_service(config: LLMProviderConfig) -> ChatCompletionClientBase:
    logger.info("Using OpenAI provider — model=%s", config.model)
    return OpenAIChatCompletion(
        ai_model_id=config.model,
        api_key=config.api_key,
    )


def _build_ollama_service(config: LLMProviderConfig) -> ChatCompletionClientBase:
    logger.info("Using Ollama provider — model=%s base_url=%s", config.model, config.base_url)
    return OllamaChatCompletion(
        ai_model_id=config.model,
        host=config.base_url,
    )


_BUILDERS: dict[str, callable] = {
    "anthropic": _build_anthropic_service,
    "openai": _build_openai_service,
    "ollama": _build_ollama_service,
}


class LLMProviderManager:
    """Creates and configures Semantic Kernel kernels with the best available LLM"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._providers = self._settings.get_available_providers()
        if not self._providers:
            raise RuntimeError(
                "No LLM providers configured. "
                "Set at least ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
            )

    @property
    def primary_provider(self) -> LLMProviderConfig:
        return self._providers[0]

    def build_kernel(self, service_id: str = "default") -> Kernel:
        """
        Build a Kernel with the highest-priority available provider
        Tries providers in order, if one fails to initialise, falls back to the next
        """

        kernel = Kernel()
        last_error: Exception | None = None

        for config in self._providers:
            builder = _BUILDERS.get(config.name)
            if builder is None:
                logger.warning("Unknown provider '%s', skipping", config.name)
                continue
            try:
                service = builder(config)
                kernel.add_service(service, service_id=service_id)
                logger.info(
                    "Kernel ready — provider=%s model=%s", config.name, config.model
                )
                return kernel
            except Exception as exc:
                logger.warning(
                    "Failed to init provider '%s': %s — trying next", config.name, exc
                )
                last_error = exc

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    def build_service(self) -> ChatCompletionClientBase:
        """Build just the chat-completion service (no kernel)"""
        
        for config in self._providers:
            builder = _BUILDERS.get(config.name)
            if builder is None:
                continue
            try:
                return builder(config)
            except Exception as exc:
                logger.warning("Provider '%s' failed: %s", config.name, exc)

        raise RuntimeError("All LLM providers failed")
