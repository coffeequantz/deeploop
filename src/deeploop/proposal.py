"""Turn a prose brief into a verifiable mission contract.

The proposal is never auto-trusted: it is shown to a human (TUI or console),
who can accept, revise in natural language, or edit the YAML directly. Budget
and permission fields are clamped and merged with conservative defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from .brief import BriefBundle
from .contract import DEFAULT_ALLOW_COMMANDS, Budget, Criterion, TaskContract
from .llm import ModelRunner
from .prompts import parse_json_response
from .providers.base import ChatMessage

PROPOSAL_SYSTEM = """You turn a written brief into a verifiable mission contract for an autonomous agent.

Respond with strict JSON only:

{
  "goal": "one sentence describing the finished outcome",
  "workspace": ".",
  "criteria": [
    {"id": "kebab-case-id", "description": "what is being checked",
     "check": "shell command run from the workspace root, or null",
     "expect_exit": 0}
  ],
  "allow_commands": ["pytest", "npm", "make"],
  "budget": {"max_usd": 2.0, "max_iterations": 30, "max_wall_clock_minutes": 30},
  "notes": "one short paragraph on assumptions or risks"
}

Rules:
- Prefer machine-checkable criteria: every criterion you can express as a command
  MUST include a `check` command. Use null only for qualities no command can test
  (design fidelity, tone, visual match).
- 1-5 criteria. Each must be independently checkable and non-overlapping.
- Check commands must be runnable from the workspace root with the stack described
  in the brief. Never invent tooling the brief does not mention.
- workspace: use "." if the brief describes changing existing files in this folder;
  use a subdirectory such as "work" if it describes creating a new artifact.
- allow_commands: list only the executables needed for the work and the checks.
- Keep the budget modest: the loop stops at max_usd, max_iterations or the wall clock.
- Do not include the brief text in your answer; only the JSON object."""


@dataclass
class ContractDraft:
    goal: str
    criteria: List[Criterion] = field(default_factory=list)
    allow_commands: List[str] = field(default_factory=list)
    workspace: str = "."
    budget: Budget = field(default_factory=Budget)
    notes: str = ""
    warnings: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_contract(self, base: TaskContract, brief_dir_value: Optional[str]) -> TaskContract:
        permissions = base.permissions.model_dump()
        permissions["workspace"] = self.workspace
        merged = sorted(set(permissions.get("allow_commands", [])) | set(self.allow_commands))
        permissions["allow_commands"] = merged
        data: Dict[str, Any] = {
            "goal": self.goal,
            "success_criteria": [c.model_dump(exclude_none=True) for c in self.criteria],
            "budget": self.budget.model_dump(),
            "permissions": permissions,
            "limits": base.limits.model_dump(),
            "escalation": base.escalation.model_dump(),
            "provider": base.provider.model_dump(),
            "model": base.model.model_dump(),
            "checkpoint": base.checkpoint.model_dump(),
            "brief": {**base.brief.model_dump(), "dir": brief_dir_value},
        }
        return TaskContract.model_validate(data)

    def summary(self) -> str:
        lines = [f"goal: {self.goal}", f"workspace: {self.workspace}", "criteria:"]
        for criterion in self.criteria:
            if criterion.kind == "command":
                lines.append(f"  - [command] {criterion.id}: {criterion.check}")
            else:
                lines.append(f"  - [judge]   {criterion.id}: {criterion.description}")
        lines.append(
            f"budget: ${self.budget.max_usd:.2f} · {self.budget.max_iterations} iterations · "
            f"{self.budget.max_wall_clock_minutes:.0f} minutes"
        )
        extra = sorted(set(self.allow_commands) - set(DEFAULT_ALLOW_COMMANDS))
        if extra:
            lines.append(f"extra allowed commands: {', '.join(extra)}")
        if self.notes:
            lines.append(f"notes: {self.notes}")
        for warning in self.warnings:
            lines.append(f"warning: {warning}")
        return "\n".join(lines)


async def propose_contract(
    runner: ModelRunner,
    bundle: BriefBundle,
    base: TaskContract,
    revision: str = "",
    max_criteria: int = 5,
) -> ContractDraft:
    messages = proposal_messages(bundle, revision)
    completion = await runner.call("planner", messages)
    parsed = parse_json_response(completion.message.content)
    if parsed is None:
        raise ValueError("planner did not return a JSON contract proposal")
    return draft_from_payload(parsed, bundle, max_criteria=max_criteria)


def proposal_messages(bundle: BriefBundle, revision: str = "") -> List[ChatMessage]:
    context = bundle.manifest(max_chars=12000)
    parts = [context]
    if revision:
        parts.append(f"HUMAN REVISION REQUEST (apply it):\n{revision}")
    parts.append("Produce the mission contract JSON.")
    return [ChatMessage.system(PROPOSAL_SYSTEM), ChatMessage.user("\n\n".join(parts))]


def draft_from_payload(payload: Dict[str, Any], bundle: BriefBundle, max_criteria: int = 5) -> ContractDraft:
    warnings: List[str] = []
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        goal = _fallback_goal(bundle)
        warnings.append("proposal had no goal; used the brief's first line")

    criteria: List[Criterion] = []
    seen_ids: set = set()
    for index, raw in enumerate(payload.get("criteria") or []):
        if len(criteria) >= max_criteria:
            warnings.append(f"dropped criteria beyond the first {max_criteria}")
            break
        if not isinstance(raw, dict):
            continue
        candidate = dict(raw)
        candidate.setdefault("id", f"criterion-{index + 1}")
        candidate["id"] = _slug(str(candidate["id"])) or f"criterion-{index + 1}"
        if candidate["id"] in seen_ids:
            candidate["id"] = f"{candidate['id']}-{index + 1}"
        seen_ids.add(candidate["id"])
        if not candidate.get("check"):
            candidate.pop("check", None)
            candidate.pop("expect_exit", None)
            candidate["kind"] = "judge"
        try:
            criteria.append(Criterion.model_validate(candidate))
        except ValidationError as exc:
            warnings.append(f"dropped invalid criterion {candidate['id']!r}: {exc.errors()[0]['msg']}")
    if not criteria:
        criteria = [
            Criterion(
                id="goal-achieved",
                description=goal,
                kind="judge",
            )
        ]
        warnings.append("no valid criteria in the proposal; fell back to a single judged criterion")
    if all(c.kind == "judge" for c in criteria):
        warnings.append("proposal has no machine-checkable criteria: completion rests on an LLM judge")

    allow_commands = _sanitize_commands(payload.get("allow_commands"))
    workspace = str(payload.get("workspace") or ".").strip() or "."
    budget = _clamp_budget(payload.get("budget"), warnings)
    notes = str(payload.get("notes") or "").strip()[:1000]
    return ContractDraft(
        goal=goal,
        criteria=criteria,
        allow_commands=allow_commands,
        workspace=workspace,
        budget=budget,
        notes=notes,
        warnings=warnings,
        raw=payload,
    )


def contract_to_yaml(contract: TaskContract) -> str:
    data = contract.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    data.setdefault("budget", contract.budget.model_dump())
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def contract_from_yaml(text: str) -> TaskContract:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("contract YAML must be a mapping")
    return TaskContract.model_validate(data)


def _fallback_goal(bundle: BriefBundle) -> str:
    for line in (bundle.brief_text or "").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:300]
    return f"Complete the work described in {bundle.dir}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48]


def _sanitize_commands(raw: Any) -> List[str]:
    commands: List[str] = []
    if not isinstance(raw, list):
        return commands
    for item in raw:
        token = str(item).strip().split()[0] if str(item).strip() else ""
        if not token:
            continue
        if any(char in token for char in ";|&$`()<>"):
            continue
        commands.append(token)
    return sorted(set(commands))


def _clamp_budget(raw: Any, warnings: List[str]) -> Budget:
    if not isinstance(raw, dict):
        return Budget()
    try:
        usd = float(raw.get("max_usd", 2.0))
    except (TypeError, ValueError):
        usd = 2.0
    try:
        iterations = int(raw.get("max_iterations", 30))
    except (TypeError, ValueError):
        iterations = 30
    try:
        minutes = float(raw.get("max_wall_clock_minutes", 30.0))
    except (TypeError, ValueError):
        minutes = 30.0
    usd = min(max(usd, 0.05), 50.0)
    iterations = min(max(iterations, 1), 200)
    minutes = min(max(minutes, 1.0), 480.0)
    if (usd, iterations, minutes) != (
        float(raw.get("max_usd", 2.0) or 0),
        int(raw.get("max_iterations", 30) or 0),
        float(raw.get("max_wall_clock_minutes", 30.0) or 0),
    ):
        warnings.append("budget values were clamped to safe ranges")
    return Budget(max_usd=usd, max_iterations=iterations, max_wall_clock_minutes=minutes)
