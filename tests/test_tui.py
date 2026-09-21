import asyncio

from conftest import fix_turns

from deeploop.runtime import build_runtime
from deeploop.tui.app import DeepLoopApp, TUIHuman
from deeploop.tui.widgets import CriteriaPanel, MissionHeader, StatusBar


def test_tui_renders_mission_and_finishes(mission) -> None:
    runtime = build_runtime(mission.contract_path, human=TUIHuman(), mock_turns=fix_turns())
    app = DeepLoopApp(runtime)

    async def scenario() -> None:
        async with app.run_test(size=(140, 45)) as pilot:
            for _ in range(400):
                if app._finished:
                    break
                await pilot.pause(0.05)
            assert app._finished, "mission did not finish inside the TUI"

            header = app.query_one(MissionHeader)
            assert "deeploop" in str(header.content)

            status = app.query_one(StatusBar)
            assert "done" in str(status.content)

            criteria = app.query_one(CriteriaPanel)
            assert "tests-pass" in str(criteria.content)

            log = app.query_one("#log")
            assert len(log.children) > 3

    asyncio.run(scenario())


def test_tui_escalation_prompt_accepts_answer(mission) -> None:
    from conftest import idle_turns, rewrite_contract

    rewrite_contract(mission, max_iterations=10, stuck_after=1, on_stuck="ask")
    runtime = build_runtime(mission.contract_path, human=TUIHuman(), mock_turns=idle_turns())
    app = DeepLoopApp(runtime)

    async def scenario() -> None:
        async with app.run_test(size=(140, 45)) as pilot:
            answered = False
            for _ in range(400):
                if app._escalation is not None:
                    app.query_one("#prompt").value = "halt"
                    await pilot.press("enter")
                    answered = True
                if app._finished:
                    break
                await pilot.pause(0.05)
            assert answered, "no escalation was raised"
            assert app._finished

    asyncio.run(scenario())


def test_proposal_screen_accepts_by_default(tmp_path) -> None:
    from deeploop.brief import load_brief
    from deeploop.proposal import draft_from_payload
    from deeploop.tui.proposal import ProposalApp

    bundle = load_brief(tmp_path)
    draft = draft_from_payload(
        {
            "goal": "Ship the widget.",
            "criteria": [{"id": "tests-pass", "check": "pytest -q"}],
            "budget": {"max_usd": 1.0, "max_iterations": 5, "max_wall_clock_minutes": 5},
        },
        bundle,
    )
    app = ProposalApp(draft, bundle)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
        return app.return_value

    assert asyncio.run(scenario()) == ("accept", None)


def test_proposal_screen_revises_with_a_note(tmp_path) -> None:
    from textual.widgets import Input

    from deeploop.brief import load_brief
    from deeploop.proposal import draft_from_payload
    from deeploop.tui.proposal import ProposalApp

    bundle = load_brief(tmp_path)
    draft = draft_from_payload(
        {"goal": "Ship the widget.", "criteria": [{"id": "a", "check": "true"}]},
        bundle,
    )
    app = ProposalApp(draft, bundle)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            field = app.query_one("#revision", Input)
            field.focus()
            field.value = "split the criteria into two"
            await pilot.press("enter")
        return app.return_value

    assert asyncio.run(scenario()) == ("revise", "split the criteria into two")
