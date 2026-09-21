from pathlib import Path

import pytest
from pydantic import ValidationError

from deeploop.contract import MissionPaths, TaskContract


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "mission.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_resolves_criterion_kinds(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
goal: "Fix it"
success_criteria:
  - id: a
    check: "pytest -q"
  - id: b
    description: "code is clean"
""",
    )
    contract = TaskContract.load(path)
    assert contract.command_criteria[0].id == "a"
    assert contract.command_criteria[0].kind == "command"
    assert contract.judge_criteria[0].kind == "judge"
    assert contract.command_criteria[0].description == "pytest -q"


def test_requires_criteria(tmp_path: Path) -> None:
    path = write(tmp_path, 'goal: "Fix it"\nsuccess_criteria: []\n')
    with pytest.raises(ValidationError):
        TaskContract.load(path)


def test_command_criterion_requires_check(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
goal: "Fix it"
success_criteria:
  - id: a
    kind: command
""",
    )
    with pytest.raises(ValidationError):
        TaskContract.load(path)


def test_judge_criterion_rejects_check(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
goal: "Fix it"
success_criteria:
  - id: a
    kind: judge
    check: "pytest -q"
""",
    )
    with pytest.raises(ValidationError):
        TaskContract.load(path)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
goal: "Fix it"
success_criteria:
  - id: a
    check: "true"
  - id: a
    check: "true"
""",
    )
    with pytest.raises(ValidationError):
        TaskContract.load(path)


def test_warnings_flag_prose_only_and_wide_permissions(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
goal: "Fix it"
success_criteria:
  - id: a
    description: "looks good"
permissions:
  allow_all_commands: true
  network: true
checkpoint:
  enabled: false
""",
    )
    warnings = TaskContract.load(path).warnings()
    joined = " ".join(warnings)
    assert "machine-checkable" in joined
    assert "allow_all_commands" in joined
    assert "network" in joined
    assert "checkpointing disabled" in joined


def test_defaults_are_conservative(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
goal: "Fix it"
success_criteria:
  - id: a
    check: "true"
""",
    )
    contract = TaskContract.load(path)
    assert contract.permissions.network is False
    assert contract.permissions.allow_all_commands is False
    assert "rm" in contract.permissions.allow_commands
    assert "sudo" in contract.permissions.deny_commands
    assert contract.escalation.on_budget_exhausted == "halt"


def test_mission_paths_resolve_workspace_relative_to_contract(tmp_path: Path) -> None:
    sub = tmp_path / "examples"
    sub.mkdir()
    path = write(
        sub,
        """
goal: "Fix it"
success_criteria:
  - id: a
    check: "true"
permissions:
  workspace: "project"
""",
    )
    contract = TaskContract.load(path)
    paths = MissionPaths.build(path, contract)
    assert paths.root == sub
    assert paths.workspace == (sub / "project").resolve()
    assert paths.ledger_path == sub / ".deeploop" / "ledger.jsonl"


def test_json_contract_supported(tmp_path: Path) -> None:
    path = tmp_path / "mission.json"
    path.write_text(
        '{"goal": "Fix it", "success_criteria": [{"id": "a", "check": "true"}]}',
        encoding="utf-8",
    )
    assert TaskContract.load(path).goal == "Fix it"
