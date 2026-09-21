import os
import shutil
import sys
from pathlib import Path

import yaml

from deeploop.cli import main

MOCK_PROJECT = "def add(a, b):\n    return a - b\n\n\ndef multiply(a, b):\n    return a * b\n"
MOCK_TEST = (
    "from mathlib import add, multiply\n"
    "\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "\n"
    "\n"
    "def test_multiply():\n"
    "    assert multiply(2, 3) == 6\n"
)

MOCK_CONTRACT = """
goal: "Make the demo project's test suite pass."
success_criteria:
  - id: tests-pass
    description: "pytest exits 0"
    check: "pytest -q"
budget:
  max_usd: 0.5
  max_iterations: 6
  max_wall_clock_minutes: 10
permissions:
  workspace: "project"
  network: false
provider:
  name: mock
checkpoint:
  enabled: true
  branch: deeploop/cli-test
"""


def test_init_writes_template_and_refuses_overwrite(tmp_path: Path) -> None:
    assert main(["init", str(tmp_path)]) == 0
    mission = tmp_path / "mission.yaml"
    assert mission.exists()
    assert "success_criteria" in mission.read_text()
    assert main(["init", str(tmp_path)]) == 1


def test_validate_reports_contract(tmp_path: Path) -> None:
    assert main(["init", str(tmp_path)]) == 0
    assert main(["validate", str(tmp_path / "mission.yaml")]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("goal: only a goal\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 1


def test_models_command() -> None:
    assert main(["models"]) == 0


def test_report_without_ledger(tmp_path: Path) -> None:
    assert main(["report", str(tmp_path)]) == 1


def test_headless_mock_run_end_to_end(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "mathlib.py").write_text(MOCK_PROJECT, encoding="utf-8")
    (project / "test_mathlib.py").write_text(MOCK_TEST, encoding="utf-8")
    contract = tmp_path / "mission.yaml"
    contract.write_text(MOCK_CONTRACT, encoding="utf-8")

    venv_bin = str(Path(sys.executable).parent)
    monkeypatch.setenv("PATH", venv_bin + os.pathsep + os.environ.get("PATH", ""))

    exit_code = main(["run", str(contract), "--headless"])
    assert exit_code == 0
    assert (project / "mathlib.py").read_text() == (
        "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n"
    )
    assert main(["report", str(tmp_path)]) == 0

    ledger = (tmp_path / ".deeploop" / "ledger.jsonl").read_text()
    assert '"kind": "mission_finished"' in ledger
    assert '"status": "done"' in ledger


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIEF_DEMO = REPO_ROOT / "examples" / "brief-demo"
FIXED_MATHLIB = "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n"


def _copy_brief_demo(tmp_path: Path) -> Path:
    project = tmp_path / "brief-demo"
    shutil.copytree(BRIEF_DEMO, project)
    return project


def _patch_path(monkeypatch) -> None:
    venv_bin = str(Path(sys.executable).parent)
    monkeypatch.setenv("PATH", venv_bin + os.pathsep + os.environ.get("PATH", ""))


def test_brief_command_derives_contract(tmp_path: Path, monkeypatch) -> None:
    project = _copy_brief_demo(tmp_path)
    _patch_path(monkeypatch)
    assert main(["brief", str(project), "--yes", "--provider", "mock"]) == 0
    contract = project / "mission.yaml"
    assert contract.exists()
    data = yaml.safe_load(contract.read_text())
    assert data["goal"]
    assert data["brief"]["dir"] == "."
    assert data["success_criteria"][0]["id"] == "tests-pass"
    assert data["success_criteria"][0]["check"] == "pytest -q"
    assert main(["brief", str(project), "--yes", "--provider", "mock"]) == 1


def test_run_derives_from_brief_then_completes(tmp_path: Path, monkeypatch) -> None:
    project = _copy_brief_demo(tmp_path)
    _patch_path(monkeypatch)
    assert main(["run", str(project), "--yes", "--headless", "--provider", "mock"]) == 0
    assert (project / "mathlib.py").read_text() == FIXED_MATHLIB
    ledger = (project / ".deeploop" / "ledger.jsonl").read_text()
    assert '"kind": "brief_loaded"' in ledger
    assert '"status": "done"' in ledger


def test_run_requires_contract_or_brief(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["run", str(empty)]) == 1
    assert main(["brief", str(empty)]) == 1
