"""Prompt construction for each role. Kept in one file so prompt changes are
reviewable in isolation from control flow."""

from __future__ import annotations

from typing import List, Optional

from .contract import TaskContract
from .permissions import PermissionGate
from .providers.base import ChatMessage
from .util import truncate_middle

PLANNER_SYSTEM = """You are the planner for an autonomous agent harness called DeepLoop.

You do not execute anything. You produce a short, concrete plan that an executor
model will follow, one iteration at a time.

Rules:
- Output a numbered list of 2-6 steps, each independently verifiable.
- Prefer steps that produce observable evidence (files changed, commands run).
- State which command the executor should run to check its own work.
- If verifier feedback is provided, target the failures directly.
- No preamble, no questions, no offers of further help."""

ACTOR_SYSTEM = """You are the executor inside an autonomous agent harness called DeepLoop.

You work in a loop until a separate verifier confirms the mission's success
criteria. You cannot declare success yourself — a verifier checks every
iteration, so claims without evidence are worthless.

Rules:
- Make concrete progress each iteration: run commands, read files, edit code.
- Use tools instead of describing what you would do.
- Keep responses short. Do not restate the goal or the plan.
- Never ask questions; you are autonomous. If blocked, state the blocker and
  try an alternative approach.
- Do not weaken or delete tests to make them pass. Do not edit files outside
  the workspace. Do not attempt to disable the harness.
- Prefer the smallest change that makes progress."""

CRITIC_SYSTEM = """You are the progress critic for an autonomous agent harness.

You receive raw evidence from one iteration. Judge whether the iteration made
concrete progress toward the mission goal. Being busy is not progress; editing
files is not progress unless it moves the success criteria closer to passing.

Respond with strict JSON only:
{"progress": true|false, "reason": "one sentence", "hint": "one actionable sentence or empty"}

Rules:
- Base the verdict only on the evidence provided.
- If the same error or the same failure repeats, progress is false.
- If the iteration changed nothing observable, progress is false."""

JUDGE_SYSTEM = """You are an independent verifier for an autonomous agent harness.

You decide whether a success criterion is satisfied, using only the evidence
provided. The executor's own claims are not evidence.

Respond with strict JSON only:
{"passed": true|false, "reason": "one or two sentences citing the evidence"}

Rules:
- Be adversarial: if the evidence is insufficient, return passed=false.
- Do not infer from intent; require observable artifacts (diffs, command output).
- A criterion that says tests pass requires test output showing them passing."""


def planner_messages(
    contract: TaskContract,
    *,
    iteration: int,
    previous_plan: str,
    criteria_status: List[str],
    verifier_feedback: str,
    history: List[str],
    human_notes: List[str],
    brief_context: str = "",
) -> List[ChatMessage]:
    parts = [
        f"MISSION GOAL:\n{contract.goal}",
        "SUCCESS CRITERIA (checked by an independent verifier):\n"
        + "\n".join(f"- {c.id}: {c.description}" for c in contract.success_criteria),
        f"ITERATION: {iteration}",
    ]
    if brief_context:
        parts.append(f"BRIEF CONTEXT:\n{truncate_middle(brief_context, 6000)}")
    if previous_plan:
        parts.append(f"PREVIOUS PLAN:\n{previous_plan}")
    if criteria_status:
        parts.append("LAST VERIFIER RESULT:\n" + "\n".join(criteria_status))
    if verifier_feedback:
        parts.append(f"VERIFIER FEEDBACK:\n{truncate_middle(verifier_feedback, 3000)}")
    if history:
        parts.append("ITERATION HISTORY:\n" + "\n".join(history[-12:]))
    if human_notes:
        parts.append("HUMAN GUIDANCE:\n" + "\n".join(human_notes))
    parts.append("Produce the plan for this iteration.")
    return [ChatMessage.system(PLANNER_SYSTEM), ChatMessage.user("\n\n".join(parts))]


def actor_messages(
    contract: TaskContract,
    gate: PermissionGate,
    *,
    plan: str,
    criteria_status: List[str],
    verifier_feedback: str,
    history: List[str],
    human_notes: List[str],
    tool_names: List[str],
    brief_context: str = "",
) -> List[ChatMessage]:
    perms = gate.summary()
    allow = ", ".join(perms["allow_commands"][:60])
    parts = [
        f"MISSION GOAL:\n{contract.goal}",
        "SUCCESS CRITERIA (a verifier checks these after every iteration):\n"
        + "\n".join(f"- {c.id}: {c.description}" for c in contract.success_criteria),
        f"CURRENT PLAN:\n{plan or '(no plan yet — take the most obvious next step)'}",
        f"WORKSPACE: {perms['workspace']}",
        f"NETWORK: {'allowed' if perms['network'] else 'blocked'}",
        f"AVAILABLE TOOLS: {', '.join(tool_names)}",
        f"ALLOWED COMMANDS: {allow}" if not perms["allow_all_commands"] else "ALLOWED COMMANDS: all",
    ]
    if brief_context:
        parts.append(f"BRIEF CONTEXT:\n{truncate_middle(brief_context, 6000)}")
    if criteria_status:
        parts.append("LAST VERIFIER RESULT:\n" + "\n".join(criteria_status))
    if verifier_feedback:
        parts.append(f"VERIFIER FEEDBACK — FIX THIS:\n{truncate_middle(verifier_feedback, 3000)}")
    if history:
        parts.append("ITERATION HISTORY:\n" + "\n".join(history[-12:]))
    if human_notes:
        parts.append("HUMAN GUIDANCE:\n" + "\n".join(human_notes))
    parts.append("Take the next concrete step using tools.")
    return [ChatMessage.system(ACTOR_SYSTEM), ChatMessage.user("\n\n".join(parts))]


def critic_messages(goal: str, evidence: str) -> List[ChatMessage]:
    return [
        ChatMessage.system(CRITIC_SYSTEM),
        ChatMessage.user(f"MISSION GOAL:\n{goal}\n\nITERATION EVIDENCE:\n{truncate_middle(evidence, 6000)}"),
    ]


def judge_messages(
    goal: str, criterion_id: str, criterion_description: str, evidence: str
) -> List[ChatMessage]:
    return [
        ChatMessage.system(JUDGE_SYSTEM),
        ChatMessage.user(
            f"MISSION GOAL:\n{goal}\n\n"
            f"CRITERION {criterion_id}:\n{criterion_description}\n\n"
            f"EVIDENCE:\n{truncate_middle(evidence, 8000)}"
        ),
    ]


def render_criteria_status(results: List) -> List[str]:
    lines = []
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        detail = result.evidence.splitlines()[0][:200] if result.evidence else ""
        lines.append(f"- [{mark}] {result.id}: {detail}")
    return lines


def parse_json_response(text: str) -> Optional[dict]:
    import json
    import re

    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
