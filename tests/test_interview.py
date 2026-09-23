import asyncio
import json
from pathlib import Path

from deeploop.brief import load_brief
from deeploop.budget import BudgetTracker
from deeploop.contract import TaskContract
from deeploop.events import EventBus
from deeploop.interview import (
    Answer,
    Question,
    collect_answers,
    load_clarifications_text,
    propose_questions,
    questions_from_payload,
    render_clarifications,
    resolve_answer,
    save_clarifications,
    skipped_summary,
)
from deeploop.ledger import Ledger
from deeploop.llm import ModelRunner
from deeploop.proposal import propose_contract
from deeploop.providers.mock import MockProvider

PAYLOAD = {
    "questions": [
        {
            "id": "verify-command",
            "question": "Which command proves the work is done?",
            "choices": ["pytest -q", "npm test"],
            "default": "pytest -q",
            "affects": "criteria",
            "why": "becomes the success criterion",
        },
        {
            "id": "deps",
            "question": "May the agent add dependencies?",
            "choices": ["no", "yes"],
            "default": "no",
            "affects": "permissions",
            "why": "changes the allowlist",
        },
    ]
}


def base_contract() -> TaskContract:
    return TaskContract.model_validate(
        {"goal": "g", "success_criteria": [{"id": "x", "check": "true"}]}
    )


def make_runner(tmp_path: Path, interviewer_text: str) -> ModelRunner:
    contract = base_contract()
    provider = MockProvider(interviewer_text=interviewer_text, planner_text=json.dumps({"goal": "g"}))
    return ModelRunner(
        provider, contract, BudgetTracker(contract.budget), Ledger(tmp_path / "ledger.jsonl"), EventBus()
    )


def test_questions_from_payload_parses_and_validates() -> None:
    questions = questions_from_payload(PAYLOAD, max_questions=5)
    assert [q.id for q in questions] == ["verify-command", "deps"]
    assert questions[0].affects == "criteria"
    assert questions[0].choices == ["pytest -q", "npm test"]
    assert questions[0].default == "pytest -q"


def test_questions_from_payload_drops_junk_and_caps() -> None:
    payload = {
        "questions": [
            {"question": "  "},
            "not a dict",
            {"id": "ok", "question": "Real question?", "affects": "nonsense"},
            {"id": "dup", "question": "Real question?"},
            {"id": "c", "question": "Third?"},
            {"id": "d", "question": "Fourth?"},
        ]
    }
    questions = questions_from_payload(payload, max_questions=2)
    assert [q.question for q in questions] == ["Real question?", "Third?"]
    assert questions[0].affects == "other"


def test_questions_from_payload_rejects_non_list() -> None:
    assert questions_from_payload({"questions": "nope"}) == []
    assert questions_from_payload({}) == []


def test_resolve_answer_accepts_number_prefix_and_text() -> None:
    question = Question(id="q", question="?", choices=["in place", "new directory"], default="in place")
    assert resolve_answer("2", question) == "new directory"
    assert resolve_answer("in p", question) == "in place"
    assert resolve_answer("something else", question) == "something else"
    assert resolve_answer("", question) == ""
    assert resolve_answer("9", question) == "9"


def test_render_and_skipped_summary() -> None:
    question = Question(id="deps", question="Dependencies?", default="no", affects="permissions")
    answers = [
        Answer(question=Question(id="v", question="Verify how?", affects="criteria"), answer="pytest -q"),
        Answer(question=question, answer=""),
    ]
    text = render_clarifications(answers)
    assert "HUMAN CLARIFICATIONS" in text
    assert "pytest -q" in text
    assert "skipped; agent assumes: no" in text
    assert skipped_summary(answers) == "unanswered clarifying question(s): deps"
    assert skipped_summary([answers[0]]) == ""


def test_collect_answers_console_flow() -> None:
    questions = [
        Question(id="a", question="First?", choices=["one", "two"], default="one"),
        Question(id="b", question="Second?"),
    ]
    replies = iter(["1", ""])
    answers = collect_answers(questions, lambda label: next(replies))
    assert [a.answer for a in answers] == ["one", ""]
    assert answers[0].skipped is False
    assert answers[1].skipped is True
    assert collect_answers(questions, lambda label: None) is None


def test_save_and_load_clarifications_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ".deeploop" / "artifacts" / "clarifications.json"
    answers = [
        Answer(question=Question(id="v", question="Verify how?", affects="criteria"), answer="pytest -q"),
        Answer(question=Question(id="d", question="Deps?", default="no", affects="permissions"), answer=""),
    ]
    save_clarifications(path, answers, "deepseek-reasoner")
    text = load_clarifications_text(path)
    assert "binding decisions" in text
    assert "pytest -q" in text
    assert "assumes: no" in text
    assert load_clarifications_text(tmp_path / "missing.json") == ""
    path.write_text("{broken", encoding="utf-8")
    assert load_clarifications_text(path) == ""


def test_propose_questions_uses_interviewer_role(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, json.dumps(PAYLOAD))
    questions = asyncio.run(propose_questions(runner, load_brief(tmp_path / "nope"), base_contract()))
    assert [q.id for q in questions] == ["verify-command", "deps"]
    assert runner.provider.calls[0]["role"] == "interviewer"


def test_propose_questions_tolerates_non_json(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, "I have no questions, you are doing great!")
    assert asyncio.run(propose_questions(runner, load_brief(tmp_path / "nope"), base_contract())) == []


def test_clarifications_reach_the_proposal_prompt(tmp_path: Path) -> None:
    base = base_contract()
    proposal_json = json.dumps({"goal": "g", "criteria": [{"id": "a", "check": "true"}]})
    provider = MockProvider(planner_text=proposal_json)
    runner = ModelRunner(
        provider, base, BudgetTracker(base.budget), Ledger(tmp_path / "ledger.jsonl"), EventBus()
    )
    clarifications = render_clarifications(
        [Answer(question=Question(id="v", question="Verify how?"), answer="pytest -q")]
    )
    asyncio.run(
        propose_contract(runner, load_brief(tmp_path / "nope"), base, "", clarifications)
    )
    prompt = "\n".join(str(m.content) for m in provider.calls[0]["messages"])
    assert "pytest -q" in prompt
    assert "binding" in prompt.lower()


def test_mock_provider_ships_demo_questions(tmp_path: Path) -> None:
    provider = MockProvider()
    parsed = json.loads(provider.interviewer_text)
    assert len(parsed["questions"]) >= 2
