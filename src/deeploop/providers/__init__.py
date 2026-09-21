"""Provider registry."""

from __future__ import annotations

from ..contract import ProviderConfig
from .base import ChatMessage, Completion, Provider, ProviderError, ToolCall, Usage
from .mock import MockProvider, MockTurn
from .openai_compat import OpenAICompatProvider

__all__ = [
    "ChatMessage",
    "Completion",
    "MockProvider",
    "MockTurn",
    "OpenAICompatProvider",
    "Provider",
    "ProviderError",
    "ToolCall",
    "Usage",
    "build_provider",
]


def build_provider(config: ProviderConfig, *, mock_turns=None) -> Provider:
    if config.name == "mock":
        return MockProvider(turns=mock_turns)
    return OpenAICompatProvider(config)
