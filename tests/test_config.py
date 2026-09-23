from pathlib import Path

from deeploop.config import (
    GlobalConfig,
    config_path,
    describe,
    is_configured,
    key_env_name,
    load_config,
    masked,
    resolve_api_key,
    save_config,
)


def test_config_path_honours_env(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "custom.yaml"
    monkeypatch.setenv("DEEPLOOP_CONFIG", str(target))
    assert config_path() == target


def test_config_path_uses_xdg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPLOOP_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_path() == tmp_path / "deeploop" / "config.yaml"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.yaml"
    config = GlobalConfig(provider="openrouter", onboarded=True)
    config.api_keys["openrouter"] = "sk-or-secret-value"
    config.base_urls["ollama"] = "http://localhost:9999/v1"
    config.models["openrouter"] = {"actor": "deepseek/deepseek-chat"}
    save_config(config, path)

    assert path.exists()
    assert oct(path.stat().st_mode)[-3:] == "600"
    loaded = load_config(path)
    assert loaded.provider == "openrouter"
    assert loaded.onboarded is True
    assert loaded.api_keys["openrouter"] == "sk-or-secret-value"
    assert loaded.base_urls["ollama"] == "http://localhost:9999/v1"
    assert loaded.models["openrouter"]["actor"] == "deepseek/deepseek-chat"


def test_missing_and_corrupt_config_are_tolerated(tmp_path: Path) -> None:
    assert load_config(tmp_path / "nope.yaml").provider == "deepseek"
    bad = tmp_path / "bad.yaml"
    bad.write_text("provider: [unclosed", encoding="utf-8")
    assert load_config(bad).provider == "deepseek"
    not_a_mapping = tmp_path / "list.yaml"
    not_a_mapping.write_text("- just\n- a list\n", encoding="utf-8")
    assert load_config(not_a_mapping).provider == "deepseek"


def test_api_key_precedence(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    save_config(GlobalConfig(api_keys={"deepseek": "from-config"}), path)

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert resolve_api_key("deepseek", path=path) == "from-config"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    assert resolve_api_key("deepseek", path=path) == "from-env"

    assert resolve_api_key("deepseek", api_key_env="CUSTOM_KEY", path=path) == "from-env"
    monkeypatch.setenv("CUSTOM_KEY", "from-custom")
    assert resolve_api_key("deepseek", api_key_env="CUSTOM_KEY", path=path) == "from-custom"


def test_is_configured(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert is_configured("ollama", path=path) is True
    assert is_configured("mock", path=path) is True
    assert is_configured("deepseek", path=path) is False
    save_config(GlobalConfig(api_keys={"deepseek": "sk-x"}), path)
    assert is_configured("deepseek", path=path) is True


def test_key_env_name_and_masking() -> None:
    assert key_env_name("deepseek") == "DEEPSEEK_API_KEY"
    assert key_env_name("ollama") is None
    assert masked(None) == "(none)"
    assert masked("short") == "*****"
    assert masked("sk-1234567890") == "sk-1…7890"


def test_describe_masks_keys(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    save_config(GlobalConfig(provider="deepseek", api_keys={"deepseek": "sk-supersecret"}), path)
    text = describe(path)
    assert "deepseek" in text
    assert "sk-supersecret" not in text
    assert "env DEEPSEEK_API_KEY: unset" in text
