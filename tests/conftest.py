"""Shared test fixtures: a tiny broken project plus a matching mission contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from deeploop.providers.base import ToolCall
from deeploop.providers.mock import MockTurn

BROKEN_CALC = "def add(a, b):\n    return a - b\n\n\ndef multiply(a, b):\n    return a * b\n"
FIXED_CALC = "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n"
TEST_CALC = (
    "from calc import add, multiply\n"
    "\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "\n"
    "\n"
    "def test_multiply():\n"
    "    assert multiply(2, 3) == 6\n"
)

CHECK_COMMAND = f"{sys.executable} -m pytest -q"


def call(name: str, **arguments):
    return ToolCall(id=f"call_{name}_{abs(hash(str(arguments))) % 100000}", name=name, arguments=arguments)


def fix_turns() -> list:
    """Two-iteration script that fixes the broken project."""
    return [
        MockTurn(tool_calls=[call("run_command", command=CHECK_COMMAND)], content=""),
        MockTurn(content="The suite fails: add() subtracts. I will fix it next iteration."),
        MockTurn(tool_calls=[call("write_file", path="calc.py", content=FIXED_CALC)], content=""),
        MockTurn(tool_calls=[call("run_command", command=CHECK_COMMAND)], content=""),
        MockTurn(content="calc.add now returns a + b and the suite passes."),
    ]


def idle_turns() -> list:
    """Turns that keep running the failing suite and change nothing."""
    return [
        MockTurn(tool_calls=[call("run_command", command=CHECK_COMMAND)], content=""),
        MockTurn(content="Still investigating."),
        MockTurn(tool_calls=[call("run_command", command=CHECK_COMMAND)], content=""),
        MockTurn(content="Still investigating."),
        MockTurn(tool_calls=[call("run_command", command=CHECK_COMMAND)], content=""),
        MockTurn(content="Still investigating."),
    ]


CONTRACT_TEMPLATE = """goal: "Make the test suite pass."
success_criteria:
  - id: tests-pass
    description: "pytest exits 0"
    check: "{check}"
budget:
  max_usd: 1.0
  max_iterations: {max_iterations}
  max_wall_clock_minutes: 10
permissions:
  workspace: "ws"
  network: false
  allow_all_commands: false
limits:
  max_tool_calls_per_iteration: 8
  stuck_after: {stuck_after}
  max_replans: {max_replans}
escalation:
  on_budget_exhausted: {on_budget_exhausted}
  on_stuck: {on_stuck}
  on_permission_denied: deny
provider:
  name: mock
checkpoint:
  enabled: true
  branch: deeploop/test
"""


@pytest.fixture
def mission(tmp_path: Path) -> SimpleNamespace:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "calc.py").write_text(BROKEN_CALC, encoding="utf-8")
    (workspace / "test_calc.py").write_text(TEST_CALC, encoding="utf-8")
    contract_path = tmp_path / "mission.yaml"
    contract_path.write_text(
        CONTRACT_TEMPLATE.format(
            check=CHECK_COMMAND,
            max_iterations=6,
            stuck_after=3,
            max_replans=2,
            on_budget_exhausted="halt",
            on_stuck="halt",
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        root=tmp_path,
        workspace=workspace,
        contract_path=contract_path,
        state_path=tmp_path / ".deeploop" / "state.json",
        ledger_path=tmp_path / ".deeploop" / "ledger.jsonl",
    )


def rewrite_contract(mission: SimpleNamespace, **overrides) -> None:
    values = dict(
        check=CHECK_COMMAND,
        max_iterations=6,
        stuck_after=3,
        max_replans=2,
        on_budget_exhausted="halt",
        on_stuck="halt",
    )
    values.update(overrides)
    mission.contract_path.write_text(CONTRACT_TEMPLATE.format(**values), encoding="utf-8")


def add_brief(mission: SimpleNamespace, rel_dir: str = "brief", describe_images: bool = False) -> Path:
    """Attach a brief folder to an existing mission fixture."""
    brief_dir = mission.root / rel_dir
    brief_dir.mkdir(parents=True, exist_ok=True)
    text = mission.contract_path.read_text(encoding="utf-8")
    text += f'\nbrief:\n  dir: "{rel_dir}"\n  describe_images: {str(describe_images).lower()}\n'
    mission.contract_path.write_text(text, encoding="utf-8")
    return brief_dir


def read_ledger(path: Path) -> list:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def ledger_kinds(path: Path) -> list:
    return [entry["kind"] for entry in read_ledger(path)]


@pytest.fixture(autouse=True)
def isolated_global_config(tmp_path_factory, monkeypatch):
    """Never read or write the developer's real ~/.config/deeploop/config.yaml."""
    path = tmp_path_factory.mktemp("deeploop-config") / "config.yaml"
    monkeypatch.setenv("DEEPLOOP_CONFIG", str(path))
    return path
