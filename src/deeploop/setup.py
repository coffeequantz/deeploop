"""Provider setup: choose a provider, paste a key, verify it works.

The non-UI logic lives here so it can be tested without a terminal; the TUI and
console front ends are thin wrappers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .catalog import DEFAULT_BASE_URLS, get_option, provider_names, provider_options
from .config import GlobalConfig, is_configured, key_env_name, load_config, save_config
from .contract import ProviderConfig, TaskContract
from .providers.base import ChatMessage, ProviderError
from .providers.openai_compat import OpenAICompatProvider

Prompt = Callable[[str], Optional[str]]


def validate_inputs(provider: str, api_key: str = "", base_url: str = "") -> List[str]:
    errors: List[str] = []
    option = get_option(provider)
    if option is None:
        return [f"unknown provider {provider!r} (choose from {', '.join(provider_names())})"]
    if option.needs_key and not api_key:
        hint = f" (get one at {option.key_url})" if option.key_url else ""
        errors.append(f"an API key is required for {provider}{hint}")
    if base_url and not base_url.startswith(("http://", "https://")):
        errors.append(f"base URL must start with http:// or https://: {base_url!r}")
    return errors


def apply_setup(
    provider: str,
    api_key: str = "",
    base_url: str = "",
    models: Optional[Dict[str, str]] = None,
    path: Optional[Path] = None,
) -> Path:
    config = load_config(path)
    option = get_option(provider)
    config.provider = provider
    config.onboarded = True
    if api_key:
        config.api_keys[provider] = api_key.strip()
    if base_url and base_url != (option.base_url if option else ""):
        config.base_urls[provider] = base_url.strip()
    elif base_url and provider in config.base_urls:
        config.base_urls.pop(provider, None)
    if models is None and option is not None:
        models = dict(option.models)
    if models:
        config.models[provider] = {role: model for role, model in models.items() if model}
    return save_config(config, path)


async def test_connection(
    provider: str,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Tuple[bool, str]:
    option = get_option(provider)
    if provider == "mock":
        return True, "mock provider needs no connection"
    if option is None:
        return False, f"unknown provider {provider!r}"
    config = ProviderConfig(
        name=provider,  # type: ignore[arg-type]
        base_url=base_url or None,
        timeout_seconds=30.0,
    )
    try:
        client = OpenAICompatProvider(config, api_key=api_key or None)
    except ProviderError as exc:
        return False, str(exc)
    try:
        completion = await client.complete(
            [ChatMessage.user("Reply with the single word: ready")],
            model=model or option.default_model,
            role="setup",
            max_tokens=8,
        )
        text = (completion.message.content or "").strip().replace("\n", " ")[:60]
        return True, f"{completion.model} answered in {completion.latency_ms}ms: {text!r}"
    except ProviderError as exc:
        hint = ""
        if exc.status in (401, 403):
            hint = " — the key was rejected"
        elif exc.status == 404:
            hint = " — model or base URL not found"
        return False, f"{exc}{hint}"
    except Exception as exc:  # noqa: BLE001 - report anything as a failed test
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        await client.aclose()


def console_setup(
    prompt: Prompt,
    secret_prompt: Prompt,
    default_provider: str = "deepseek",
    test: bool = True,
    path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Interactive setup over plain prompts. Returns the saved settings, or None
    if the human cancelled. `prompt` returning None cancels."""
    options = provider_options()
    listing = "\n".join(f"  {index + 1}. {option.label}" for index, option in enumerate(options))
    answer = prompt(f"Choose a provider:\n{listing}\n[1-{len(options)}]")
    if answer is None:
        return None
    answer = answer.strip() or "1"
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        provider = options[int(answer) - 1].name
    elif answer in provider_names():
        provider = answer
    else:
        return None

    option = get_option(provider)
    assert option is not None
    api_key = ""
    if option.needs_key:
        stored = load_config(path).api_keys.get(provider, "")
        hint = f" (enter to keep {stored[:4]}…)" if stored else ""
        raw = secret_prompt(f"Paste your {provider} API key{hint}")
        if raw is None:
            return None
        api_key = raw.strip() or stored

    base_url = ""
    if provider != "mock":
        raw = prompt(f"Base URL [{option.base_url}]")
        if raw is None:
            return None
        base_url = raw.strip()

    errors = validate_inputs(provider, api_key, base_url)
    if errors:
        return {"error": "; ".join(errors), "provider": provider}

    result: Dict[str, Any] = {"provider": provider, "api_key": api_key, "base_url": base_url}
    if test:
        import asyncio

        ok, message = asyncio.run(test_connection(provider, api_key, base_url))
        result["test_ok"] = ok
        result["test_message"] = message
        if not ok:
            return result
    saved = apply_setup(provider, api_key, base_url, path=path)
    result["saved_to"] = str(saved)
    return result


def resolve_provider(
    contract: TaskContract, override: Optional[str] = None, path: Optional[Path] = None
) -> Tuple[str, str]:
    """Pick the provider to use, returning (provider, note).

    The contract wins when it has a usable key; otherwise the provider chosen
    during `deeploop setup` takes over, with a note explaining the swap.
    """
    requested = override or contract.provider.name
    if is_configured(requested, path):
        return requested, ""
    configured = load_config(path).provider
    if configured and configured != requested and is_configured(configured, path):
        return (
            configured,
            f"contract asks for {requested} but no key is configured for it; "
            f"using {configured} from `deeploop setup`",
        )
    return requested, ""


def apply_to_contract(
    contract: TaskContract, provider_override: Optional[str] = None, path: Optional[Path] = None
) -> None:
    """Seed a contract with the machine's provider choice, models and base URL."""
    config = load_config(path)
    name = provider_override or (config.provider if config.onboarded else contract.provider.name)
    contract.provider.name = name  # type: ignore[assignment]
    option = get_option(name)
    models = config.models.get(name) or (dict(option.models) if option else {})
    for role in ("planner", "actor", "critic", "judge"):
        model = models.get(role)
        if model:
            setattr(contract.model, role, model)
    url = config.base_urls.get(name) or (option.base_url if option else "")
    if url and url != DEFAULT_BASE_URLS.get(name):
        contract.provider.base_url = url


def config_summary(path: Optional[Path] = None) -> str:
    config: GlobalConfig = load_config(path)
    option = get_option(config.provider)
    lines = [
        f"provider: {config.provider}"
        + (f" ({option.label.split('·')[0].strip()})" if option else ""),
        f"key env: {key_env_name(config.provider) or 'not required'}",
    ]
    for name, key in sorted(config.api_keys.items()):
        shown = f"{key[:4]}…{key[-4:]}" if len(key) > 8 else "set"
        lines.append(f"stored key for {name}: {shown}")
    return "\n".join(lines)
