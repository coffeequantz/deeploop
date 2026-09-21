import asyncio
from pathlib import Path

import pytest

from deeploop.budget import BudgetTracker
from deeploop.contract import TaskContract
from deeploop.events import EventBus
from deeploop.ledger import Ledger
from deeploop.llm import ModelRunner
from deeploop.providers.mock import MockProvider
from deeploop.verifier import Critic, Evidence, Verifier

CONTRACT = """
goal: "Make the test suite pass."
success_criteria:
  - id: tests-pass
    check: "{check}"
  - id: not-weakened
    description: "The tests were not deleted or loosened."
budget:
  max_usd: 1.0
  max_iterations: 5
permissions:
  workspace: "ws"
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    return ws


def make_contract(tmp_path: Path, workspace: Path, check: str, judge_criteria: bool = True) -> TaskContract:
    text = CONTRACT.format(check=check)
    if not judge_criteria:
        text = text.replace(
            '  - id: not-weakened\n    description: "The tests were not deleted or loosened."\n', ""
        )
    path = tmp_path / "mission.yaml"
    path.write_text(text, encoding="utf-8")
    contract = TaskContract.load(path)
    contract.permissions.workspace = str(workspace)
    return contract


def make_verifier(contract: TaskContract, workspace: Path, provider: MockProvider):
    bus = EventBus()
    budget = BudgetTracker(contract.budget)
    ledger = Ledger(workspace / ".deeploop" / "ledger.jsonl")
    runner = ModelRunner(provider, contract, budget, ledger, bus)
    return Verifier(contract, runner, bus, workspace, contract.limits), runner


def evidence() -> Evidence:
    return Evidence(
        goal="Make the test suite pass.",
        iteration=1,
        diff="--- a/calc.py\n+++ b/calc.py\n-    return a - b\n+    return a + b",
        changed_files=["calc.py"],
        recent_outputs=[("run_command", "$ pytest -q\n1 passed\n[exit 0]")],
        final_text="fixed add()",
    )


def test_command_criterion_passes(tmp_path: Path, workspace: Path) -> None:
    contract = make_contract(tmp_path, workspace, "python3 -c 'pass'", judge_criteria=False)
    verifier, _ = make_verifier(contract, workspace, MockProvider())
    report = asyncio.run(verifier.verify(1, evidence()))
    assert report.all_passed
    assert report.results[0].kind == "command"
    assert report.judge_used is False


def test_command_criterion_fails_and_skips_judge(tmp_path: Path, workspace: Path) -> None:
    contract = make_contract(tmp_path, workspace, "python3 -c 'import sys; sys.exit(1)'")
    verifier, _ = make_verifier(contract, workspace, MockProvider())
    report = asyncio.run(verifier.verify(1, evidence()))
    assert not report.all_passed
    assert report.judge_used is False
    assert "judge skipped" in report.notes
    assert any(r.kind == "command" and not r.passed for r in report.results)
    assert report.feedback()


def test_judge_runs_when_command_passes(tmp_path: Path, workspace: Path) -> None:
    contract = make_contract(tmp_path, workspace, "python3 -c 'pass'")
    verifier, _ = make_verifier(contract, workspace, MockProvider(judge_pass=True))
    report = asyncio.run(verifier.verify(1, evidence()))
    assert report.judge_used is True
    assert report.all_passed
    judge_result = [r for r in report.results if r.kind == "judge"][0]
    assert "mock judge" in judge_result.reason


def test_judge_can_fail_the_mission(tmp_path: Path, workspace: Path) -> None:
    contract = make_contract(tmp_path, workspace, "python3 -c 'pass'")
    verifier, _ = make_verifier(contract, workspace, MockProvider(judge_pass=False))
    report = asyncio.run(verifier.verify(1, evidence()))
    assert not report.all_passed
    assert report.judge_used is True


def test_command_criterion_timeout(tmp_path: Path, workspace: Path) -> None:
    contract = make_contract(
        tmp_path, workspace, "python3 -c 'import time; time.sleep(30)'", judge_criteria=False
    )
    contract.success_criteria[0].timeout_seconds = 1
    verifier, _ = make_verifier(contract, workspace, MockProvider())
    report = asyncio.run(verifier.verify(1, evidence()))
    assert not report.all_passed
    assert "timed out" in report.results[0].reason


def test_verifier_records_llm_cost(tmp_path: Path, workspace: Path) -> None:
    contract = make_contract(tmp_path, workspace, "python3 -c 'pass'")
    verifier, runner = make_verifier(contract, workspace, MockProvider())
    asyncio.run(verifier.verify(1, evidence()))
    assert runner.budget.calls == 1  # the judge call


def test_critic_parses_progress(tmp_path: Path, workspace: Path) -> None:
    contract = make_contract(tmp_path, workspace, "python3 -c 'pass'", judge_criteria=False)
    _, runner = make_verifier(contract, workspace, MockProvider())
    critic = Critic(runner, EventBus())
    review = asyncio.run(critic.review(contract.goal, evidence()))
    assert review.progress is True

    _, runner2 = make_verifier(contract, workspace, MockProvider(critic_progress=False))
    critic2 = Critic(runner2, EventBus())
    review2 = asyncio.run(critic2.review(contract.goal, evidence()))
    assert review2.progress is False
    assert review2.hint


def test_critic_fails_open_on_bad_output(tmp_path: Path, workspace: Path) -> None:
    from deeploop.providers.base import ChatMessage, Completion, Usage

    class BadCriticProvider(MockProvider):
        async def complete(
            self, messages, *, model, tools=None, role="actor", temperature=None, max_tokens=None
        ):
            if role == "critic":
                return Completion(
                    message=ChatMessage.assistant("Looks good to me!"),
                    usage=Usage(),
                    model=model,
                    role=role,
                )
            return await super().complete(
                messages, model=model, tools=tools, role=role, temperature=temperature, max_tokens=max_tokens
            )

    contract = make_contract(tmp_path, workspace, "python3 -c 'pass'", judge_criteria=False)
    bus = EventBus()
    budget = BudgetTracker(contract.budget)
    ledger = Ledger(workspace / ".deeploop" / "ledger.jsonl")
    runner = ModelRunner(BadCriticProvider(), contract, budget, ledger, bus)
    critic = Critic(runner, bus)
    review = asyncio.run(critic.review(contract.goal, evidence()))
    assert review.progress is True
    assert "unparseable" in review.reason
