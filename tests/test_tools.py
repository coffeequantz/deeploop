import asyncio
from pathlib import Path

import pytest

from deeploop.contract import Limits, Permissions
from deeploop.permissions import PermissionGate
from deeploop.tools import (
    CheckpointManager,
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    ToolContext,
    WriteFileTool,
    default_registry,
)


def make_ctx(tmp_path: Path, **overrides) -> ToolContext:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    gate = PermissionGate(Permissions(workspace="ws"), workspace)
    limits = Limits()
    return ToolContext(
        workspace=workspace,
        gate=gate,
        max_output_chars=overrides.get("max_output_chars", limits.max_output_chars),
        command_timeout_seconds=overrides.get("command_timeout_seconds", 5),
    )


def run(coro):
    return asyncio.run(coro)


def test_run_command_success_and_failure(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    tool = RunCommandTool()
    ok = run(tool.run({"command": "echo hello"}, ctx))
    assert ok.ok
    assert "hello" in ok.output
    assert "[exit 0]" in ok.output
    failed = run(tool.run({"command": "python3 -c 'import sys; sys.exit(3)'"}, ctx))
    assert not failed.ok
    assert "[exit 3]" in failed.error


def test_run_command_timeout(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, command_timeout_seconds=1)
    result = run(RunCommandTool().run({"command": "python3 -c 'import time; time.sleep(5)'"}, ctx))
    assert not result.ok
    assert "timed out" in result.error
    assert result.metadata.get("timeout") is True


def test_run_command_output_truncated(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, max_output_chars=200)
    result = run(
        RunCommandTool().run({"command": "python3 -c \"print('x' * 5000)\""}, ctx)
    )
    assert result.ok
    assert "chars omitted" in result.output


def test_run_command_denied(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    from deeploop.permissions import PermissionDenied

    with pytest.raises(PermissionDenied):
        run(RunCommandTool().run({"command": "curl https://example.com"}, ctx))


def test_registry_reports_permission_denial_as_tool_result(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    registry = default_registry()
    result = run(registry.execute("run_command", {"command": "curl https://example.com"}, ctx))
    assert not result.ok
    assert "permission denied" in result.error


def test_registry_unknown_tool(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    result = run(default_registry().execute("nope", {}, ctx))
    assert not result.ok
    assert "unknown tool" in result.error


def test_registry_schemas_are_openai_tools() -> None:
    schemas = default_registry().schemas()
    names = {schema["function"]["name"] for schema in schemas}
    assert {"run_command", "read_file", "write_file", "edit_file", "list_files"} <= names
    for schema in schemas:
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]


def test_file_tools_round_trip(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    write = run(WriteFileTool().run({"path": "src/a.txt", "content": "one\ntwo\n"}, ctx))
    assert write.ok
    read = run(ReadFileTool().run({"path": "src/a.txt"}, ctx))
    assert "one" in read.output and "2 lines total" in read.output
    edit = run(EditFileTool().run({"path": "src/a.txt", "old_string": "two", "new_string": "three"}, ctx))
    assert edit.ok
    assert "three" in (ctx.workspace / "src" / "a.txt").read_text()


def test_edit_requires_unique_match(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    run(WriteFileTool().run({"path": "a.txt", "content": "x\nx\n"}, ctx))
    ambiguous = run(EditFileTool().run({"path": "a.txt", "old_string": "x", "new_string": "y"}, ctx))
    assert not ambiguous.ok
    assert "matches 2 times" in ambiguous.error
    forced = run(
        EditFileTool().run({"path": "a.txt", "old_string": "x", "new_string": "y", "replace_all": True}, ctx)
    )
    assert forced.ok
    assert (ctx.workspace / "a.txt").read_text() == "y\ny\n"


def test_file_tools_block_escape(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    from deeploop.permissions import PermissionDenied

    with pytest.raises(PermissionDenied):
        run(WriteFileTool().run({"path": "../outside.txt", "content": "nope"}, ctx))


def test_list_files_skips_noise_dirs(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    (ctx.workspace / "pkg").mkdir()
    (ctx.workspace / "pkg" / "mod.py").write_text("x = 1")
    (ctx.workspace / "__pycache__").mkdir()
    (ctx.workspace / "__pycache__" / "junk.pyc").write_text("junk")
    result = run(ListFilesTool().run({}, ctx))
    assert "pkg/mod.py" in result.output
    assert "junk.pyc" not in result.output


def test_checkpoint_manager_commit_and_rollback(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "file.txt").write_text("v1")

    async def scenario():
        manager = CheckpointManager(workspace)
        assert await manager.ensure_repo() is True
        first = await manager.commit("first")
        assert first
        (workspace / "file.txt").write_text("v2")
        assert await manager.changed_files() == ["file.txt"]
        second = await manager.commit("second")
        assert second and second != first
        diff = await manager.diff_since(first)
        assert "v2" in diff
        assert await manager.rollback(first) is True
        assert (workspace / "file.txt").read_text() == "v1"
        return True

    assert run(scenario())


def test_checkpoint_manager_disabled(tmp_path: Path) -> None:
    from deeploop.contract import CheckpointConfig

    workspace = tmp_path / "ws"
    workspace.mkdir()

    async def scenario():
        manager = CheckpointManager(workspace, CheckpointConfig(enabled=False))
        assert await manager.ensure_repo() is False
        assert await manager.commit("nope") is None

    run(scenario())
    assert not (workspace / ".git").exists()
