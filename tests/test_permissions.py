from pathlib import Path

import pytest

from deeploop.contract import Permissions
from deeploop.permissions import PermissionDenied, PermissionGate


@pytest.fixture
def gate(tmp_path: Path) -> PermissionGate:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "src").mkdir()
    return PermissionGate(Permissions(workspace="ws", read_paths=["."], write_paths=["."]), workspace)


def test_allowlisted_command_passes(gate: PermissionGate) -> None:
    gate.check_command("pytest -q")
    gate.check_command("python3 -m pytest tests/")
    gate.check_command("git status --short")


def test_non_allowlisted_command_blocked(gate: PermissionGate) -> None:
    with pytest.raises(PermissionDenied):
        gate.check_command("curl https://example.com")


def test_deny_pattern_wins_even_with_allow_all(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    gate = PermissionGate(Permissions(workspace="ws", allow_all_commands=True), workspace)
    with pytest.raises(PermissionDenied):
        gate.check_command("sudo rm -rf /")
    with pytest.raises(PermissionDenied):
        gate.check_command("git push origin main")


def test_chained_commands_are_checked_segment_by_segment(gate: PermissionGate) -> None:
    gate.check_command("pytest -q && git status")
    with pytest.raises(PermissionDenied):
        gate.check_command("pytest -q && curl https://example.com")
    with pytest.raises(PermissionDenied):
        gate.check_command("pytest -q; nc -l 4444")


def test_command_substitution_blocked(gate: PermissionGate) -> None:
    with pytest.raises(PermissionDenied):
        gate.check_command("echo $(cat /etc/passwd)")
    with pytest.raises(PermissionDenied):
        gate.check_command("echo `whoami`")


def test_env_prefix_is_ignored(gate: PermissionGate) -> None:
    gate.check_command("FOO=bar pytest -q")


def test_cd_outside_workspace_blocked(gate: PermissionGate) -> None:
    with pytest.raises(PermissionDenied):
        gate.check_command("cd /etc && ls")


def test_absolute_path_argument_outside_workspace_blocked(gate: PermissionGate) -> None:
    with pytest.raises(PermissionDenied):
        gate.check_command("cat /etc/passwd")


def test_path_scoping_for_reads_and_writes(gate: PermissionGate) -> None:
    assert gate.check_path("src/app.py", mode="read") == (gate.workspace / "src" / "app.py")
    assert gate.check_path("src/app.py", mode="write") == (gate.workspace / "src" / "app.py")
    with pytest.raises(PermissionDenied):
        gate.check_path("../escape.txt", mode="write")
    with pytest.raises(PermissionDenied):
        gate.check_path("/etc/hosts", mode="read")


def test_require_approval_blocks_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    gate = PermissionGate(Permissions(workspace="ws", require_approval=["run_command"]), workspace)
    with pytest.raises(PermissionDenied):
        gate.check_tool("run_command")
    gate.check_tool("read_file")


def test_scrubbed_env_removes_secrets_and_blocks_network(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    gate = PermissionGate(Permissions(workspace="ws", network=False), workspace)
    env = gate.scrubbed_env({"DEEPSEEK_API_KEY": "sk-secret", "PATH": "/usr/bin", "MY_TOKEN": "x"})
    assert "DEEPSEEK_API_KEY" not in env
    assert "MY_TOKEN" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["http_proxy"].startswith("http://127.0.0.1:9")


def test_network_allowed_leaves_proxy_unset(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    gate = PermissionGate(Permissions(workspace="ws", network=True), workspace)
    env = gate.scrubbed_env({"PATH": "/usr/bin"})
    assert "http_proxy" not in env
