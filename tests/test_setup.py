import asyncio
from pathlib import Path

from deeploop.config import GlobalConfig, is_configured, load_config, save_config
from deeploop.contract import TaskContract
from deeploop.setup import (
    apply_setup,
    apply_to_contract,
    config_summary,
    console_setup,
    resolve_provider,
    validate_inputs,
)
from deeploop.setup import test_connection as probe_connection


def base_contract(provider: str = "deepseek") -> TaskContract:
    return TaskContract.model_validate(
        {
            "goal": "g",
            "success_criteria": [{"id": "x", "check": "true"}],
            "provider": {"name": provider},
        }
    )


def test_validate_inputs() -> None:
    assert validate_inputs("nope") != []
    assert validate_inputs("deepseek", "", "") != []
    assert validate_inputs("deepseek", "sk-x", "") == []
    assert validate_inputs("ollama", "", "") == []
    assert validate_inputs("deepseek", "sk-x", "ftp://bad") != []
    assert validate_inputs("openrouter", "sk-x", "https://openrouter.ai/api/v1") == []


def test_apply_setup_stores_provider_models_and_key(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    saved = apply_setup("deepseek", "sk-secret", path=path)
    assert saved == path
    config = load_config(path)
    assert config.provider == "deepseek"
    assert config.onboarded is True
    assert config.api_keys["deepseek"] == "sk-secret"
    assert config.models["deepseek"]["actor"] == "deepseek-chat"
    assert oct(path.stat().st_mode)[-3:] == "600"
    assert is_configured("deepseek", path=path) is True


def test_apply_setup_keeps_custom_base_url_only(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    apply_setup("ollama", path=path)
    assert "ollama" not in load_config(path).base_urls
    apply_setup("ollama", base_url="http://192.168.1.5:11434/v1", path=path)
    assert load_config(path).base_urls["ollama"] == "http://192.168.1.5:11434/v1"


def test_console_setup_writes_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    replies = iter(["1", ""])
    secrets = iter(["sk-console-key"])
    result = console_setup(
        lambda text: next(replies, None),
        lambda text: next(secrets, None),
        test=False,
        path=path,
    )
    assert result is not None
    assert result["provider"] == "deepseek"
    assert load_config(path).api_keys["deepseek"] == "sk-console-key"


def test_console_setup_cancel_and_bad_choice(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    assert console_setup(lambda text: None, lambda text: "", test=False, path=path) is None
    assert console_setup(lambda text: "99", lambda text: "", test=False, path=path) is None


def test_console_setup_reports_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    replies = iter(["1", ""])
    result = console_setup(
        lambda text: next(replies, None),
        lambda text: "",  # empty key and nothing stored
        test=False,
        path=path,
    )
    assert result is not None and "error" in result
    assert not path.exists()


def test_test_connection_for_mock() -> None:
    ok, message = asyncio.run(probe_connection("mock"))
    assert ok is True
    assert "no connection" in message


def test_test_connection_reports_unreachable_provider() -> None:
    ok, message = asyncio.run(
        probe_connection("ollama", base_url="http://127.0.0.1:9/v1", model="qwen2.5-coder:7b")
    )
    assert ok is False
    assert message


def test_resolve_provider_prefers_contract_when_configured(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    save_config(GlobalConfig(provider="openrouter", api_keys={"openrouter": "sk-or"}), path)
    contract = base_contract("mock")
    provider, note = resolve_provider(contract, path=path)
    assert provider == "mock"
    assert note == ""


def test_resolve_provider_falls_back_to_configured(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    save_config(GlobalConfig(provider="openrouter", api_keys={"openrouter": "sk-or"}), path)
    contract = base_contract("deepseek")
    provider, note = resolve_provider(contract, path=path)
    assert provider == "openrouter"
    assert "no key is configured" in note


def test_resolve_provider_override_wins(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    save_config(GlobalConfig(provider="openrouter", api_keys={"openrouter": "sk-or"}), path)
    provider, note = resolve_provider(base_contract("deepseek"), override="ollama", path=path)
    assert provider == "ollama"
    assert note == ""


def test_apply_to_contract_seeds_models_and_provider(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    save_config(
        GlobalConfig(
            provider="openrouter",
            onboarded=True,
            models={"openrouter": {"actor": "vendor/actor", "planner": "vendor/planner"}},
            base_urls={"openrouter": "https://proxy.example/v1"},
        ),
        path,
    )
    contract = base_contract("deepseek")
    apply_to_contract(contract, path=path)
    assert contract.provider.name == "openrouter"
    assert contract.model.actor == "vendor/actor"
    assert contract.model.planner == "vendor/planner"
    assert contract.provider.base_url == "https://proxy.example/v1"


def test_apply_to_contract_ignores_config_until_onboarded(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    save_config(GlobalConfig(provider="openrouter"), path)
    contract = base_contract("deepseek")
    apply_to_contract(contract, path=path)
    assert contract.provider.name == "deepseek"


def test_config_summary_mentions_provider(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    save_config(GlobalConfig(provider="ollama"), path)
    assert "provider: ollama" in config_summary(path)
