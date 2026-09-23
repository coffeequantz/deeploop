"""Global config so `deeploop` works immediately after install.

Provider choice, API keys and base-URL overrides live in a 0600 YAML file:

    $XDG_CONFIG_HOME/deeploop/config.yaml   (default ~/.config/deeploop/config.yaml)

Environment variables always take precedence over stored keys, and
`DEEPLOOP_CONFIG` points at a different config file (used by tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .catalog import DEFAULT_KEY_ENVS, get_option

CONFIG_VERSION = 1
CONFIG_ENV = "DEEPLOOP_CONFIG"


@dataclass
class GlobalConfig:
    version: int = CONFIG_VERSION
    provider: str = "deepseek"
    api_keys: Dict[str, str] = field(default_factory=dict)
    base_urls: Dict[str, str] = field(default_factory=dict)
    models: Dict[str, Dict[str, str]] = field(default_factory=dict)
    onboarded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "provider": self.provider,
            "api_keys": dict(self.api_keys),
            "base_urls": dict(self.base_urls),
            "models": {name: dict(roles) for name, roles in self.models.items()},
            "onboarded": self.onboarded,
        }


def config_path() -> Path:
    override = os.environ.get(CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "deeploop" / "config.yaml"


def load_config(path: Optional[Path] = None) -> GlobalConfig:
    target = Path(path) if path else config_path()
    if not target.exists():
        return GlobalConfig()
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return GlobalConfig()
    if not isinstance(data, dict):
        return GlobalConfig()
    config = GlobalConfig()
    config.version = int(data.get("version", CONFIG_VERSION) or CONFIG_VERSION)
    config.provider = str(data.get("provider") or config.provider)
    config.onboarded = bool(data.get("onboarded", False))
    keys = data.get("api_keys")
    if isinstance(keys, dict):
        config.api_keys = {str(k): str(v) for k, v in keys.items() if v}
    urls = data.get("base_urls")
    if isinstance(urls, dict):
        config.base_urls = {str(k): str(v) for k, v in urls.items() if v}
    models = data.get("models")
    if isinstance(models, dict):
        config.models = {
            str(name): {str(role): str(model) for role, model in roles.items()}
            for name, roles in models.items()
            if isinstance(roles, dict)
        }
    return config


def save_config(config: GlobalConfig, path: Optional[Path] = None) -> Path:
    target = Path(path) if path else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.parent.chmod(0o700)
    except OSError:
        pass
    target.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return target


def key_env_name(provider: str) -> Optional[str]:
    option = get_option(provider)
    if option is not None:
        return option.key_env
    return DEFAULT_KEY_ENVS.get(provider)


def resolve_api_key(
    provider: str, api_key_env: Optional[str] = None, path: Optional[Path] = None
) -> Optional[str]:
    """Explicit env var, then the provider's default env var, then stored config."""
    for name in (api_key_env, key_env_name(provider)):
        if not name:
            continue
        value = os.environ.get(name)
        if value:
            return value.strip()
    stored = load_config(path).api_keys.get(provider)
    return stored.strip() if stored else None


def is_configured(provider: str, path: Optional[Path] = None) -> bool:
    option = get_option(provider)
    if option is not None and not option.needs_key:
        return True
    if option is None:
        return True
    return resolve_api_key(provider, path=path) is not None


def masked(secret: Optional[str], keep: int = 4) -> str:
    if not secret:
        return "(none)"
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}…{secret[-keep:]}"


def describe(path: Optional[Path] = None) -> str:
    config = load_config(path)
    target = Path(path) if path else config_path()
    lines = [f"config file: {target}"]
    lines.append(f"provider: {config.provider}")
    lines.append(f"onboarded: {'yes' if config.onboarded else 'no'}")
    for name in sorted(set(list(config.api_keys) + [config.provider])):
        env_name = key_env_name(name)
        env_state = "set" if env_name and os.environ.get(env_name) else "unset"
        lines.append(
            f"  {name}: key {masked(config.api_keys.get(name))} (env {env_name or 'n/a'}: {env_state})"
        )
    if config.base_urls:
        for name, url in sorted(config.base_urls.items()):
            lines.append(f"  {name} base_url: {url}")
    return "\n".join(lines)
