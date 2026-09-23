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


def test_interview_screen_collects_answers() -> None:
    from textual.widgets import Input

    from deeploop.interview import Question
    from deeploop.tui.interview import InterviewApp

    questions = [
        Question(
            id="v",
            question="Verify how?",
            choices=["pytest -q"],
            default="pytest -q",
            affects="criteria",
        ),
        Question(id="d", question="Deps?", choices=["no", "yes"], default="no", affects="permissions"),
    ]
    app = InterviewApp(questions)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#q-0", Input).value = "1"
            app.query_one("#q-1", Input).value = "yes"
            await pilot.press("enter")
            await pilot.press("enter")
        return app.return_value

    assert asyncio.run(scenario()) == ["1", "yes"]


def test_interview_screen_can_skip_all() -> None:
    from deeploop.interview import Question
    from deeploop.tui.interview import InterviewApp

    app = InterviewApp([Question(id="v", question="Verify how?")])

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
        return app.return_value

    assert asyncio.run(scenario()) == []


def test_setup_screen_saves_provider(tmp_path) -> None:
    from deeploop.config import load_config
    from deeploop.tui.setup import SetupApp

    config_path = tmp_path / "config.yaml"
    app = SetupApp(default_provider="deepseek", path=config_path)

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await pilot.click("#p-mock")
            await pilot.click("#save")
        return app.return_value

    result = asyncio.run(scenario())
    assert result is not None
    assert result["provider"] == "mock"
    assert load_config(config_path).provider == "mock"
    assert load_config(config_path).onboarded is True


def test_setup_screen_requires_a_key(tmp_path) -> None:
    from textual.widgets import Static

    from deeploop.tui.setup import SetupApp

    config_path = tmp_path / "config.yaml"
    app = SetupApp(default_provider="deepseek", path=config_path)

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await pilot.click("#save")
            return str(app.query_one("#setup-status", Static).content)

    status = asyncio.run(scenario())
    assert "API key" in status
    assert not config_path.exists()
