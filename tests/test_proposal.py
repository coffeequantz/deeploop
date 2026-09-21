import asyncio
import json
from pathlib import Path

import pytest
import yaml

from deeploop.brief import load_brief
from deeploop.budget import BudgetTracker
from deeploop.contract import TaskContract
from deeploop.events import EventBus
from deeploop.ledger import Ledger
from deeploop.llm import ModelRunner
from deeploop.proposal import (
    contract_from_yaml,
    contract_to_yaml,
    draft_from_payload,
    propose_contract,
)
from deeploop.providers.mock import MockProvider

BASE = {
    "goal": "placeholder",
    "success_criteria": [{"id": "x", "check": "true"}],
}

PAYLOAD = {
    "goal": "Ship a working widget page.",
    "workspace": "work",
    "criteria": [
        {"id": "tests-pass", "description": "tests pass", "check": "pytest -q"},
        {"id": "builds", "description": "build succeeds", "check": "npm run build"},
        {"id": "matches-mock", "description": "matches the reference mock", "check": None},
    ],
    "allow_commands": ["pytest", "npm", "rm -rf /", "bad; rm", ""],
    "budget": {"max_usd": 3.5, "max_iterations": 12, "max_wall_clock_minutes": 20},
    "notes": "Assumes Node 20.",
}


def base_contract() -> TaskContract:
    return TaskContract.model_validate(BASE)


def test_draft_from_payload_builds_criteria() -> None:
    draft = draft_from_payload(PAYLOAD, load_brief(Path("/nonexistent")))
    assert [c.kind for c in draft.criteria] == ["command", "command", "judge"]
    assert draft.workspace == "work"
    assert draft.budget.max_usd == 3.5
    assert draft.notes == "Assumes Node 20."


def test_allow_commands_are_sanitized() -> None:
    draft = draft_from_payload(PAYLOAD, load_brief(Path("/nonexistent")))
    assert "pytest" in draft.allow_commands
    assert "npm" in draft.allow_commands
    assert all(" " not in command and ";" not in command for command in draft.allow_commands)
    assert not any(command.startswith("bad") for command in draft.allow_commands)


def test_invalid_criteria_are_dropped_with_warnings() -> None:
    payload = dict(PAYLOAD)
    payload["criteria"] = [{"id": "ok", "check": "pytest -q"}, {"id": "bad", "check": ["nope"]}]
    draft = draft_from_payload(payload, load_brief(Path("/nonexistent")))
    assert [c.id for c in draft.criteria] == ["ok"]
    assert any("dropped invalid criterion" in warning for warning in draft.warnings)


def test_empty_criteria_falls_back_to_judge() -> None:
    payload = {"goal": "Do the thing", "criteria": []}
    draft = draft_from_payload(payload, load_brief(Path("/nonexistent")))
    assert len(draft.criteria) == 1
    assert draft.criteria[0].kind == "judge"
    assert any("fell back" in warning for warning in draft.warnings)


def test_budget_is_clamped() -> None:
    draft = draft_from_payload(
        {
            "goal": "g",
            "criteria": [],
            "budget": {"max_usd": 5000, "max_iterations": 0, "max_wall_clock_minutes": 1},
        },
        load_brief(Path("/nonexistent")),
    )
    assert draft.budget.max_usd == 50.0
    assert draft.budget.max_iterations == 1
    assert draft.budget.max_wall_clock_minutes == 1.0
    assert any("clamped" in warning for warning in draft.warnings)


def test_prose_only_proposal_warns() -> None:
    draft = draft_from_payload(
        {"goal": "g", "criteria": [{"id": "looks-good", "description": "looks good"}]},
        load_brief(Path("/nonexistent")),
    )
    assert any("no machine-checkable" in warning for warning in draft.warnings)


def test_draft_to_contract_keeps_permissions_conservative() -> None:
    draft = draft_from_payload(PAYLOAD, load_brief(Path("/nonexistent")))
    contract = draft.to_contract(base_contract(), "brief")
    assert contract.brief.dir == "brief"
    assert contract.permissions.workspace == "work"
    assert contract.permissions.network is False
    assert contract.permissions.allow_all_commands is False
    assert "sudo" in contract.permissions.deny_commands
    assert contract.budget.max_iterations == 12


def test_yaml_round_trip() -> None:
    draft = draft_from_payload(PAYLOAD, load_brief(Path("/nonexistent")))
    contract = draft.to_contract(base_contract(), ".")
    text = contract_to_yaml(contract)
    data = yaml.safe_load(text)
    assert data["budget"]["max_usd"] == 3.5
    assert data["brief"]["dir"] == "."
    restored = contract_from_yaml(text)
    assert restored.goal == contract.goal
    assert [c.id for c in restored.success_criteria] == [c.id for c in contract.success_criteria]
    assert restored.permissions.workspace == contract.permissions.workspace


def test_propose_contract_uses_planner_output(tmp_path: Path) -> None:
    brief_dir = tmp_path / "brief"
    brief_dir.mkdir()
    (brief_dir / "BRIEF.md").write_text("# Goal\n\nMake the widget work.\n")
    bundle = load_brief(brief_dir)
    provider = MockProvider(planner_text=json.dumps(PAYLOAD))
    base = base_contract()
    runner = ModelRunner(
        provider, base, BudgetTracker(base.budget), Ledger(tmp_path / "ledger.jsonl"), EventBus()
    )
    draft = asyncio.run(propose_contract(runner, bundle, base))
    assert draft.goal == PAYLOAD["goal"]
    assert len(draft.criteria) == 3


def test_propose_contract_rejects_non_json(tmp_path: Path) -> None:
    brief_dir = tmp_path / "brief"
    brief_dir.mkdir()
    (brief_dir / "BRIEF.md").write_text("# Goal\n\nSomething.\n")
    bundle = load_brief(brief_dir)
    provider = MockProvider(planner_text="1. do the thing\n2. verify")
    base = base_contract()
    runner = ModelRunner(
        provider, base, BudgetTracker(base.budget), Ledger(tmp_path / "ledger.jsonl"), EventBus()
    )
    with pytest.raises(ValueError):
        asyncio.run(propose_contract(runner, bundle, base))
