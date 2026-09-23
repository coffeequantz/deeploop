"""Clarifying interview ("grill me"): questions the human answers before a brief
becomes a contract.

Every question must name the contract field it changes, which keeps the
interview from drifting into open-ended conversation. Answers are binding input
to the proposal, are recorded in the ledger and in
`.deeploop/artifacts/clarifications.json`, and are shown to the planner, the
executor and the judge.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .brief import BriefBundle
from .contract import TaskContract
from .llm import ModelRunner
from .prompts import parse_json_response
from .providers.base import ChatMessage

AFFECTS = ("criteria", "workspace", "permissions", "budget", "scope", "other")
MAX_CHOICES = 6
DEFAULT_MAX_QUESTIONS = 5

INTERVIEW_SYSTEM = """You interrogate the human before an autonomous coding agent starts work.

Your questions remove ambiguity that would change the mission contract. This is not a
conversation: every question must change a contract field, or it is dropped.

Respond with strict JSON only:

{"questions": [
  {"id": "kebab-case-id",
   "question": "one direct question",
   "choices": ["short option", "short option"],
   "default": "what the agent should assume if the human does not answer",
   "affects": "criteria|workspace|permissions|budget|scope",
   "why": "one sentence: what changes in the contract because of this answer"}
]}

Rules:
- 2 to {max_questions} questions, ordered by impact on whether the mission can succeed.
- Never ask what the brief already answers. Re-asking stated facts wastes the human's time.
- Prioritise, in this order: how completion is verified (an exact command), scope
  boundaries (edit in place or a new directory; may dependencies be added), permissions
  (which commands and whether network access is needed), and budget expectations.
- Offer "choices" when the answer space is small, and always provide a default.
- Never ask the human to restate the goal, and never ask about your own capabilities."""


@dataclass
class Question:
    id: str
    question: str
    choices: List[str] = field(default_factory=list)
    default: str = ""
    affects: str = "scope"
    why: str = ""

    def hint(self) -> str:
        bits: List[str] = []
        if self.choices:
            bits.append("options: " + " / ".join(self.choices))
        if self.default:
            bits.append(f"default: {self.default}")
        if self.why:
            bits.append(self.why)
        return " · ".join(bits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "choices": list(self.choices),
            "default": self.default,
            "affects": self.affects,
            "why": self.why,
        }


@dataclass
class Answer:
    question: Question
    answer: str = ""

    @property
    def skipped(self) -> bool:
        return not self.answer.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.question.id,
            "question": self.question.question,
            "affects": self.question.affects,
            "answer": self.answer,
            "skipped": self.skipped,
            "assumed_default": self.question.default if self.skipped else "",
        }


async def propose_questions(
    runner: ModelRunner,
    bundle: BriefBundle,
    base: TaskContract,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
) -> List[Question]:
    messages = interview_messages(bundle, base, max_questions)
    completion = await runner.call("interviewer", messages)
    parsed = parse_json_response(completion.message.content)
    if parsed is None:
        return []
    return questions_from_payload(parsed, max_questions=max_questions)


def interview_messages(
    bundle: BriefBundle, base: TaskContract, max_questions: int = DEFAULT_MAX_QUESTIONS
) -> List[ChatMessage]:
    contract_skeleton = [
        "CONTRACT SKELETON (what your answers will change):",
        f"- criteria: currently {len(base.success_criteria)} placeholder criterion",
        f"- budget: ${base.budget.max_usd:.2f} / {base.budget.max_iterations} iterations",
        f"- workspace: {base.permissions.workspace}",
        f"- network: {'allowed' if base.permissions.network else 'blocked'}",
    ]
    user = "\n\n".join(
        [
            f"BRIEF:\n{bundle.manifest(max_chars=10000)}",
            "\n".join(contract_skeleton),
            f"Ask up to {max_questions} questions as JSON.",
        ]
    )
    system = INTERVIEW_SYSTEM.replace("{max_questions}", str(max_questions))
    return [ChatMessage.system(system), ChatMessage.user(user)]


def questions_from_payload(
    payload: Dict[str, Any], max_questions: int = DEFAULT_MAX_QUESTIONS
) -> List[Question]:
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        return []
    questions: List[Question] = []
    seen_text: set = set()
    for index, raw in enumerate(raw_questions):
        if len(questions) >= max_questions:
            break
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("question") or "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.lower())
        if key in seen_text:
            continue
        seen_text.add(key)
        affects = str(raw.get("affects") or "scope").strip().lower()
        if affects not in AFFECTS:
            affects = "other"
        choices = [str(choice).strip() for choice in (raw.get("choices") or []) if str(choice).strip()]
        questions.append(
            Question(
                id=_slug(str(raw.get("id") or f"question-{index + 1}")) or f"question-{index + 1}",
                question=text[:400],
                choices=choices[:MAX_CHOICES],
                default=str(raw.get("default") or "").strip()[:200],
                affects=affects,
                why=str(raw.get("why") or "").strip()[:300],
            )
        )
    return questions


def resolve_answer(value: str, question: Question) -> str:
    """Accept free text, a 1-based option number, or a choice prefix."""
    value = (value or "").strip()
    if not value:
        return ""
    if value.isdigit() and question.choices:
        index = int(value)
        if 1 <= index <= len(question.choices):
            return question.choices[index - 1]
    lowered = value.lower()
    for choice in question.choices:
        if choice.lower() == lowered:
            return choice
    for choice in question.choices:
        if choice.lower().startswith(lowered):
            return choice
    return value


def collect_answers(questions: List[Question], prompts) -> Optional[List[Answer]]:
    """Console interview. `prompts` is a callable taking the display string and
    returning the raw input; returning None cancels the interview."""
    answers: List[Answer] = []
    for question in questions:
        label = f"[{question.affects}] {question.question}"
        if question.hint():
            label += f"\n    {question.hint()}"
        raw = prompts(label)
        if raw is None:
            return None
        answers.append(Answer(question=question, answer=resolve_answer(raw, question)))
    return answers


def render_clarifications(answers: List[Answer]) -> str:
    if not answers:
        return ""
    lines = ["HUMAN CLARIFICATIONS (binding decisions from the brief interview):"]
    for item in answers:
        prefix = f"- [{item.question.affects}] {item.question.question}"
        if item.skipped:
            assumed = item.question.default or "no default given"
            lines.append(f"{prefix}\n  answer: (skipped; agent assumes: {assumed})")
        else:
            lines.append(f"{prefix}\n  answer: {item.answer}")
    return "\n".join(lines)


def skipped_summary(answers: List[Answer]) -> str:
    skipped = [item.question.id for item in answers if item.skipped]
    if not skipped:
        return ""
    return "unanswered clarifying question(s): " + ", ".join(skipped)


def save_clarifications(path: Path, answers: List[Answer], model: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "model": model,
        "questions": [item.question.to_dict() for item in answers],
        "answers": [item.to_dict() for item in answers],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_clarifications_text(path: Path) -> str:
    """Render stored clarifications for prompts. Empty string when absent."""
    path = Path(path)
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    records = payload.get("answers")
    if not isinstance(records, list):
        return ""
    lines = ["HUMAN CLARIFICATIONS (binding decisions from the brief interview):"]
    for record in records:
        if not isinstance(record, dict):
            continue
        affects = record.get("affects", "scope")
        question = str(record.get("question", "")).strip()
        if record.get("skipped"):
            assumed = record.get("assumed_default") or "no default given"
            lines.append(f"- [{affects}] {question}\n  answer: (skipped; agent assumes: {assumed})")
        else:
            lines.append(f"- [{affects}] {question}\n  answer: {record.get('answer', '')}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48]
