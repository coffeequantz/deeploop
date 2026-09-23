"""Provider catalogue: base URLs, key environment variables, suggested models.

Kept dependency-free so both the config layer and the HTTP client can import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_BASE_URLS: Dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "mock": "",
}

DEFAULT_KEY_ENVS: Dict[str, Optional[str]] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
    "mock": None,
}


@dataclass
class ProviderOption:
    name: str
    label: str
    needs_key: bool
    base_url: str
    models: Dict[str, str] = field(default_factory=dict)
    key_env: Optional[str] = None
    key_url: str = ""
    notes: str = ""

    @property
    def default_model(self) -> str:
        return self.models.get("actor", "")


PROVIDER_OPTIONS: List[ProviderOption] = [
    ProviderOption(
        name="deepseek",
        label="DeepSeek  · cheapest, best value for long loops",
        needs_key=True,
        base_url=DEFAULT_BASE_URLS["deepseek"],
        key_env=DEFAULT_KEY_ENVS["deepseek"],
        key_url="https://platform.deepseek.com/api_keys",
        models={
            "planner": "deepseek-reasoner",
            "actor": "deepseek-chat",
            "critic": "deepseek-chat",
            "judge": "deepseek-reasoner",
        },
        notes="~$0.27/M input, $1.10/M output. The actor model must support function calling.",
    ),
    ProviderOption(
        name="openrouter",
        label="OpenRouter  · one key, many models (incl. vision)",
        needs_key=True,
        base_url=DEFAULT_BASE_URLS["openrouter"],
        key_env=DEFAULT_KEY_ENVS["openrouter"],
        key_url="https://openrouter.ai/keys",
        models={
            "planner": "deepseek/deepseek-chat",
            "actor": "deepseek/deepseek-chat",
            "critic": "deepseek/deepseek-chat",
            "judge": "deepseek/deepseek-chat",
        },
        notes="Any OpenRouter model id works; vision models can describe brief images.",
    ),
    ProviderOption(
        name="ollama",
        label="Ollama  · local, free, no key",
        needs_key=False,
        base_url=DEFAULT_BASE_URLS["ollama"],
        key_env=None,
        models={
            "planner": "qwen2.5-coder:7b",
            "actor": "qwen2.5-coder:7b",
            "critic": "qwen2.5-coder:7b",
            "judge": "qwen2.5-coder:7b",
        },
        notes="Requires `ollama serve` and a model that supports tool calling.",
    ),
    ProviderOption(
        name="mock",
        label="Mock  · scripted, no network (tests and demos)",
        needs_key=False,
        base_url="",
        key_env=None,
        models={"planner": "mock", "actor": "mock", "critic": "mock", "judge": "mock"},
        notes="Runs the bundled example without spending anything.",
    ),
]


def provider_options() -> List[ProviderOption]:
    return list(PROVIDER_OPTIONS)


def get_option(name: str) -> Optional[ProviderOption]:
    for option in PROVIDER_OPTIONS:
        if option.name == name:
            return option
    return None


def provider_names() -> List[str]:
    return [option.name for option in PROVIDER_OPTIONS]
