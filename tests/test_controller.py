import asyncio
import subprocess

from conftest import (
    CHECK_COMMAND,
    FIXED_CALC,
    call,
    fix_turns,
    idle_turns,
    ledger_kinds,
    read_ledger,
    rewrite_contract,
)

from deeploop.controller import MissionStatus
from deeploop.human import AutoHaltHuman
from deeploop.ledger import MissionState
from deeploop.providers.mock import MockProvider, MockTurn
from deeploop.runtime import build_runtime


def run_mission(mission, turns, resume: bool = False, human=None):
    runtime = build_runtime(
        mission.contract_path,
        resume=resume,
        human=human or AutoHaltHuman(),
        mock_turns=turns,
    )
    result = asyncio.run(runtime.controller.run())
    return result, runtime


def test_mission_completes_and_records_everything(mission) -> None:
    result, runtime = run_mission(mission, fix_turns())
    assert result.status == MissionStatus.DONE
    assert result.iterations == 2
    assert result.report is not None and result.report.all_passed
    assert (mission.workspace / "calc.py").read_text() == FIXED_CALC

    kinds = ledger_kinds(mission.ledger_path)
    for expected in (
        "mission_started",
        "iteration_started",
        "plan",
        "assistant_message",
        "tool_call",
        "checkpoint",
        "verification",
        "cost",
        "mission_finished",
    ):
        assert expected in kinds, expected

    state = MissionState.load(mission.state_path)
    assert state is not None
    assert state.status == "done"
    assert state.criteria == {"tests-pass": True}
    assert state.last_green_sha
    assert state.history

    branch = subprocess.run(
        ["git", "-C", str(mission.workspace), "branch", "--show-current"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "deeploop/test"


def test_iteration_budget_is_enforced_hard(mission) -> None:
    rewrite_contract(mission, max_iterations=1)
    result, runtime = run_mission(mission, fix_turns())
    assert result.status == MissionStatus.BUDGET_EXHAUSTED
    assert result.iterations == 1
    assert "iteration limit" in result.summary


def test_usd_budget_is_enforced_hard(mission) -> None:
    rewrite_contract(mission, max_iterations=10)
    runtime = build_runtime(mission.contract_path, human=AutoHaltHuman(), mock_turns=fix_turns())
    runtime.budget.budget.max_usd = 0.0
    runtime.budget.spent_usd = 0.01
    result = asyncio.run(runtime.controller.run())
    assert result.status == MissionStatus.BUDGET_EXHAUSTED
    assert "spend limit" in result.summary


def test_budget_ask_policy_can_continue(mission) -> None:
    rewrite_contract(mission, max_iterations=1, on_budget_exhausted="ask")
    result, runtime = run_mission(mission, fix_turns(), human=AutoHaltHuman(default="continue"))
    assert result.status == MissionStatus.DONE
    assert runtime.budget.budget.max_iterations > 1


def test_stuck_detection_halts(mission) -> None:
    rewrite_contract(mission, max_iterations=10, stuck_after=2, on_stuck="halt")
    result, _ = run_mission(mission, idle_turns())
    assert result.status == MissionStatus.HALTED
    assert "stuck" in result.summary


def test_stuck_policy_replan_forces_a_new_plan(mission) -> None:
    rewrite_contract(mission, max_iterations=4, stuck_after=1, max_replans=1, on_stuck="replan")
    result, runtime = run_mission(mission, idle_turns())
    assert result.status in (MissionStatus.HALTED, MissionStatus.BUDGET_EXHAUSTED)
    assert runtime.state.replans == 1
    assert "replan" in ledger_kinds(mission.ledger_path)


def test_permission_denial_is_recorded_and_fed_back(mission) -> None:
    rewrite_contract(mission, max_iterations=1)
    turns = [
        MockTurn(tool_calls=[call("run_command", command="curl https://example.com")], content=""),
        MockTurn(content="The command was denied; I need another approach."),
    ]
    result, _ = run_mission(mission, turns)
    assert result.status == MissionStatus.BUDGET_EXHAUSTED
    tool_calls = [e for e in read_ledger(mission.ledger_path) if e["kind"] == "tool_call"]
    assert tool_calls and tool_calls[0]["ok"] is False
    assert "permission denied" in tool_calls[0]["output"]


def test_path_escape_is_denied(mission) -> None:
    rewrite_contract(mission, max_iterations=1)
    turns = [
        MockTurn(tool_calls=[call("write_file", path="../escape.txt", content="nope")], content=""),
        MockTurn(content="Denied."),
    ]
    _, _ = run_mission(mission, turns)
    assert not (mission.root / "escape.txt").exists()


def test_resume_continues_from_ledger_state(mission) -> None:
    rewrite_contract(mission, max_iterations=1)
    first, _ = run_mission(mission, fix_turns())
    assert first.status == MissionStatus.BUDGET_EXHAUSTED

    rewrite_contract(mission, max_iterations=6)
    second, runtime = run_mission(mission, fix_turns(), resume=True)
    assert second.status == MissionStatus.DONE
    assert second.iterations == 3
    starts = [e for e in read_ledger(mission.ledger_path) if e["kind"] == "mission_started"]
    assert len(starts) == 2
    assert runtime.state.plan


def test_judge_criteria_are_used_when_commands_pass(mission) -> None:
    mission.contract_path.write_text(
        f"""
goal: "Make the test suite pass."
success_criteria:
  - id: tests-pass
    check: "{CHECK_COMMAND}"
  - id: tests-not-weakened
    description: "Tests were not deleted or loosened."
budget:
  max_usd: 1.0
  max_iterations: 6
permissions:
  workspace: "ws"
provider:
  name: mock
checkpoint:
  enabled: true
  branch: deeploop/test
""",
        encoding="utf-8",
    )
    result, runtime = run_mission(mission, fix_turns())
    assert result.status == MissionStatus.DONE
    assert result.report is not None and result.report.judge_used
    judge_calls = [c for c in runtime.provider.calls if c["role"] == "judge"]
    assert judge_calls


def test_actor_cannot_declare_done_without_verifier(mission) -> None:
    rewrite_contract(mission, max_iterations=2, stuck_after=99)
    turns = [MockTurn(content="I am completely done, mission accomplished!")]
    result, _ = run_mission(mission, turns)
    assert result.status == MissionStatus.BUDGET_EXHAUSTED
    assert result.report is None or not result.report.all_passed


def test_interjection_is_included_in_context(mission) -> None:
    rewrite_contract(mission, max_iterations=1)
    runtime = build_runtime(mission.contract_path, human=AutoHaltHuman(), mock_turns=fix_turns())
    runtime.controller.interject("please focus on mathlib")
    asyncio.run(runtime.controller.run())
    entries = read_ledger(mission.ledger_path)
    assert any(e["kind"] == "human_interjection" for e in entries)


def test_provider_error_ends_mission(mission) -> None:
    from deeploop.providers.base import ProviderError

    class BrokenProvider(MockProvider):
        async def complete(
            self, messages, *, model, tools=None, role="actor", temperature=None, max_tokens=None
        ):
            raise ProviderError("boom", status=500)

    rewrite_contract(mission, max_iterations=3)
    runtime = build_runtime(mission.contract_path, human=AutoHaltHuman())
    runtime.provider = BrokenProvider()
    runtime.runner.provider = runtime.provider
    result = asyncio.run(runtime.controller.run())
    assert result.status == MissionStatus.ERROR
    assert "provider unavailable" in result.summary


def test_brief_context_reaches_the_actor(mission) -> None:
    from conftest import add_brief

    brief_dir = add_brief(mission)
    (brief_dir / "BRIEF.md").write_text(
        "# Goal\n\nUse the flimflam module, never the flamflim one.\n", encoding="utf-8"
    )
    (brief_dir / "context").mkdir()
    (brief_dir / "context" / "notes.md").write_text("# Notes\n\nDetails.\n", encoding="utf-8")

    result, runtime = run_mission(mission, fix_turns())
    assert result.status == MissionStatus.DONE
    assert "brief_loaded" in ledger_kinds(mission.ledger_path)

    actor_calls = [call for call in runtime.provider.calls if call["role"] == "actor"]
    assert actor_calls
    joined = "\n".join(str(message.content) for message in actor_calls[0]["messages"])
    assert "flimflam" in joined
    assert "context/notes.md" in joined
    assert str(brief_dir) in runtime.gate.summary()["read_roots"]

    brief_records = [
        entry for entry in read_ledger(mission.ledger_path) if entry["kind"] == "brief_loaded"
    ]
    assert brief_records and brief_records[0]["hashes"]


def test_brief_images_without_vision_warn(mission) -> None:
    from conftest import add_brief

    from deeploop.events import E

    brief_dir = add_brief(mission, describe_images=True)
    (brief_dir / "BRIEF.md").write_text("# Goal\n\nFix it.\n", encoding="utf-8")
    (brief_dir / "assets").mkdir()
    (brief_dir / "assets" / "shot.png").write_bytes(b"not really a png")

    runtime = build_runtime(mission.contract_path, human=AutoHaltHuman(), mock_turns=fix_turns())
    events = []
    runtime.bus.subscribe(lambda event: events.append(event))
    asyncio.run(runtime.controller.run())
    assert any(
        event.kind == E.LOG and "not described" in str(event.data.get("message"))
        for event in events
    )
    assert any(event.kind == E.BRIEF for event in events)


def test_brief_change_is_detected_on_resume(mission) -> None:
    from conftest import add_brief

    rewrite_contract(mission, max_iterations=1)
    brief_dir = add_brief(mission)
    (brief_dir / "BRIEF.md").write_text("# Goal\n\nVersion one.\n", encoding="utf-8")
    run_mission(mission, fix_turns())
    (brief_dir / "BRIEF.md").write_text("# Goal\n\nVersion two.\n", encoding="utf-8")
    rewrite_contract(mission, max_iterations=2)
    add_brief(mission)
    _, runtime = run_mission(mission, fix_turns(), resume=True)
    records = [
        entry for entry in read_ledger(mission.ledger_path) if entry["kind"] == "brief_loaded"
    ]
    assert records[-1]["changed_since_last_run"] is True
