"""Verification ladder: deterministic checks first, LLM judge only for prose
criteria, plus a separate critic that rates iteration progress."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .contract import Criterion, Limits, TaskContract
from .events import E, EventBus
from .llm import ModelRunner
from .prompts import critic_messages, judge_messages, parse_json_response
from .util import first_error_line, truncate_middle


@dataclass
class CriterionResult:
    id: str
    kind: str
    passed: bool
    evidence: str = ""
    duration_ms: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    iteration: int
    results: List[CriterionResult] = field(default_factory=list)
    judge_used: bool = False
    notes: str = ""

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def failed(self) -> List[CriterionResult]:
        return [r for r in self.results if not r.passed]

    def feedback(self) -> str:
        lines: List[str] = []
        for result in self.failed:
            lines.append(f"[{result.id}] FAILED — {result.reason or 'criterion not satisfied'}")
            if result.evidence:
                lines.append(truncate_middle(result.evidence, 1500))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "all_passed": self.all_passed,
            "judge_used": self.judge_used,
            "notes": self.notes,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class ProgressReview:
    progress: bool
    reason: str = ""
    hint: str = ""


@dataclass
class Evidence:
    goal: str
    iteration: int
    diff: str = ""
    changed_files: List[str] = field(default_factory=list)
    recent_outputs: List[Tuple[str, str]] = field(default_factory=list)
    final_text: str = ""
    criteria_status: List[str] = field(default_factory=list)

    def render(self, max_chars: int = 8000) -> str:
        parts = [f"ITERATION {self.iteration}"]
        if self.changed_files:
            parts.append("FILES CHANGED:\n" + "\n".join(f"- {f}" for f in self.changed_files[:40]))
        else:
            parts.append("FILES CHANGED: none")
        if self.diff:
            parts.append("DIFF:\n" + truncate_middle(self.diff, max_chars // 2))
        if self.recent_outputs:
            blocks = []
            for name, output in self.recent_outputs[-6:]:
                blocks.append(f"--- {name} ---\n{truncate_middle(output, max_chars // 3)}")
            parts.append("TOOL OUTPUT:\n" + "\n\n".join(blocks))
        if self.final_text:
            parts.append("EXECUTOR FINAL MESSAGE:\n" + truncate_middle(self.final_text, 1500))
        if self.criteria_status:
            parts.append("CRITERIA STATUS BEFORE THIS VERIFICATION:\n" + "\n".join(self.criteria_status))
        return "\n\n".join(parts)


class Verifier:
    def __init__(
        self,
        contract: TaskContract,
        runner: ModelRunner,
        bus: EventBus,
        workspace: Path,
        limits: Limits,
    ) -> None:
        self.contract = contract
        self.runner = runner
        self.bus = bus
        self.workspace = Path(workspace)
        self.limits = limits

    async def verify(self, iteration: int, evidence: Evidence) -> VerificationReport:
        report = VerificationReport(iteration=iteration)
        for criterion in self.contract.command_criteria:
            report.results.append(await self._run_command_criterion(criterion))
        command_failed = any(not r.passed for r in report.results)
        judge_criteria = self.contract.judge_criteria
        if judge_criteria:
            if command_failed:
                report.notes = "judge skipped: deterministic criteria failed this iteration"
            else:
                report.judge_used = True
                for criterion in judge_criteria:
                    report.results.append(await self._run_judge_criterion(criterion, evidence))
        await self.bus.emit(E.VERIFICATION, **report.to_dict())
        return report

    async def _run_command_criterion(self, criterion: Criterion) -> CriterionResult:
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                criterion.check or "",
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=criterion.timeout_seconds)
            exit_code = proc.returncode or 0
            output = (stdout or b"").decode("utf-8", errors="replace")
            passed = exit_code == criterion.expect_exit
            reason = "" if passed else (first_error_line(output) or f"exit code {exit_code}")
            return CriterionResult(
                id=criterion.id,
                kind="command",
                passed=passed,
                evidence=f"$ {criterion.check}\n[exit {exit_code}]\n{truncate_middle(output, 4000)}",
                duration_ms=int((time.monotonic() - started) * 1000),
                reason=reason,
            )
        except asyncio.TimeoutError:
            return CriterionResult(
                id=criterion.id,
                kind="command",
                passed=False,
                evidence=f"$ {criterion.check}\n[timeout after {criterion.timeout_seconds}s]",
                duration_ms=int((time.monotonic() - started) * 1000),
                reason=f"check timed out after {criterion.timeout_seconds}s",
            )

    async def _run_judge_criterion(self, criterion: Criterion, evidence: Evidence) -> CriterionResult:
        started = time.monotonic()
        messages = judge_messages(
            self.contract.goal,
            criterion.id,
            criterion.description,
            evidence.render(),
        )
        try:
            completion = await self.runner.call("judge", messages)
        except Exception as exc:  # noqa: BLE001 - a failed judge must not pass
            return CriterionResult(
                id=criterion.id,
                kind="judge",
                passed=False,
                reason=f"judge call failed: {exc}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        parsed = parse_json_response(completion.message.content)
        if parsed is None:
            return CriterionResult(
                id=criterion.id,
                kind="judge",
                passed=False,
                reason="judge returned unparseable output; treated as not passed",
                evidence=truncate_middle(completion.message.content, 1000),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        passed = bool(parsed.get("passed"))
        return CriterionResult(
            id=criterion.id,
            kind="judge",
            passed=passed,
            reason=str(parsed.get("reason", ""))[:1000],
            evidence=evidence.render(3000),
            duration_ms=int((time.monotonic() - started) * 1000),
        )


class Critic:
    def __init__(self, runner: ModelRunner, bus: EventBus) -> None:
        self.runner = runner
        self.bus = bus

    async def review(self, goal: str, evidence: Evidence) -> ProgressReview:
        try:
            completion = await self.runner.call("critic", critic_messages(goal, evidence.render()))
        except Exception as exc:  # noqa: BLE001 - fail open: absence of a critic is not evidence of stuck
            review = ProgressReview(progress=True, reason=f"critic unavailable: {exc}")
            await self.bus.emit(E.CRITIC, progress=review.progress, reason=review.reason, hint="")
            return review
        parsed = parse_json_response(completion.message.content)
        if parsed is None:
            review = ProgressReview(progress=True, reason="critic output unparseable; assuming progress")
        else:
            review = ProgressReview(
                progress=bool(parsed.get("progress", True)),
                reason=str(parsed.get("reason", ""))[:1000],
                hint=str(parsed.get("hint", ""))[:1000],
            )
        await self.bus.emit(E.CRITIC, progress=review.progress, reason=review.reason, hint=review.hint)
        return review
