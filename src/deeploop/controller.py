"""The mission controller: plan -> act -> observe -> verify, with hard budget
enforcement, stuck detection, escalation and resumable state."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .brief import BriefBundle, load_brief
from .budget import BudgetExceeded, BudgetTracker
from .contract import MissionPaths, TaskContract
from .events import E, EventBus
from .human import HumanInterface
from .interview import load_clarifications_text
from .ledger import Ledger, MissionState
from .llm import ModelRunner
from .permissions import PermissionGate
from .prompts import actor_messages, planner_messages, render_criteria_status
from .providers.base import ChatMessage, Provider, ProviderError
from .tools import CheckpointManager, ToolContext, ToolRegistry
from .util import first_error_line, one_line, stable_hash, truncate_middle
from .verifier import Critic, Evidence, ProgressReview, VerificationReport, Verifier
from .vision import describe_images, image_cache_path


class MissionStatus(str, Enum):
    DONE = "done"
    HALTED = "halted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class MissionResult:
    status: MissionStatus
    iterations: int
    spent_usd: float
    elapsed_seconds: float
    summary: str = ""
    report: Optional[VerificationReport] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["report"] = self.report.to_dict() if self.report else None
        return data


@dataclass
class IterationOutcome:
    iteration: int
    tool_outputs: List[Tuple[str, str]] = field(default_factory=list)
    final_text: str = ""
    tool_calls: int = 0
    changed_files: List[str] = field(default_factory=list)
    changes_known: bool = True
    permission_denials: int = 0
    provider_errors: int = 0
    halt_reason: Optional[str] = None
    sha: Optional[str] = None


class Controller:
    def __init__(
        self,
        contract: TaskContract,
        paths: MissionPaths,
        provider: Provider,
        bus: EventBus,
        ledger: Ledger,
        budget: BudgetTracker,
        gate: PermissionGate,
        registry: ToolRegistry,
        runner: ModelRunner,
        verifier: Verifier,
        critic: Critic,
        checkpoints: CheckpointManager,
        human: HumanInterface,
        state: Optional[MissionState] = None,
    ) -> None:
        self.contract = contract
        self.paths = paths
        self.provider = provider
        self.bus = bus
        self.ledger = ledger
        self.budget = budget
        self.gate = gate
        self.registry = registry
        self.runner = runner
        self.verifier = verifier
        self.critic = critic
        self.checkpoints = checkpoints
        self.human = human
        self.state = state or MissionState()
        self.tool_ctx = ToolContext(
            workspace=paths.workspace,
            gate=gate,
            max_output_chars=contract.limits.max_output_chars,
            command_timeout_seconds=contract.limits.command_timeout_seconds,
            pycache_prefix=paths.pycache_dir,
            checkpoints=checkpoints,
            bus=bus,
        )
        self._stop_requested = False
        self._human_notes: List[str] = []
        self.brief: Optional[BriefBundle] = None
        self.clarifications_text: str = ""
        self._force_replan = False
        self._last_report: Optional[VerificationReport] = None
        self._consecutive_provider_errors = 0
        self._finished = False

    # ------------------------------------------------------------------ public

    def request_stop(self) -> None:
        self._stop_requested = True

    def interject(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._human_notes.append(text)
        self.ledger.append("human_interjection", text=text)
        try:
            asyncio.get_running_loop().create_task(self.bus.emit(E.INTERJECTION, text=text))
        except RuntimeError:
            pass

    async def run(self) -> MissionResult:
        await self.bus.emit(
            E.MISSION_STARTED,
            goal=self.contract.goal,
            criteria=[c.model_dump() for c in self.contract.success_criteria],
            permissions=self.gate.summary(),
            budget=self.budget.snapshot(),
            workspace=str(self.paths.workspace),
            resumed=bool(self.state.iteration),
        )
        self.ledger.append(
            "mission_started",
            goal=self.contract.goal,
            workspace=str(self.paths.workspace),
            provider=self.contract.provider.name,
            models=self.contract.model.model_dump(),
            max_usd=self.contract.budget.max_usd,
            max_iterations=self.contract.budget.max_iterations,
        )
        try:
            await self.checkpoints.ensure_repo()
            if self.checkpoints.ready and not self.state.last_green_sha:
                sha = await self.checkpoints.commit("deeploop: mission start")
                if sha:
                    self.state.last_green_sha = sha
                    self.ledger.append("checkpoint", sha=sha, message="mission start", green=True)
            await self._load_brief()
            await self._load_clarifications()
            result = await self._loop()
            return result
        except Exception as exc:  # noqa: BLE001 - surface as mission error
            self.ledger.append("error", message=f"{type(exc).__name__}: {exc}")
            await self.bus.emit(E.LOG, level="error", message=f"{type(exc).__name__}: {exc}")
            return await self._finish(MissionStatus.ERROR, f"{type(exc).__name__}: {exc}")
        finally:
            await self.human.close()
            try:
                await self.provider.aclose()
            except Exception:  # noqa: BLE001
                pass

    # -------------------------------------------------------------------- loop

    async def _loop(self) -> MissionResult:
        while not self._stop_requested:
            try:
                self.budget.start_iteration()
            except BudgetExceeded as exc:
                result = await self._handle_budget_exceeded(exc)
                if result is not None:
                    return result
                continue
            iteration = self.budget.iterations
            self.state.iteration = iteration
            self.runner.current_iteration = iteration
            await self.bus.emit(
                E.ITERATION_STARTED,
                iteration=iteration,
                max_iterations=self.budget.budget.max_iterations,
            )
            self.ledger.append(
                "iteration_started",
                iteration=iteration,
                spent_usd=round(self.budget.spent_usd, 6),
            )
            try:
                if self._should_plan():
                    await self._replan(iteration)
                outcome = await self._run_actor(iteration)
                if outcome.halt_reason:
                    return await self._finish(MissionStatus.HALTED, outcome.halt_reason)
                outcome.sha = await self._checkpoint(iteration)
                if outcome.sha:
                    self.state.last_sha = outcome.sha
                evidence = await self._build_evidence(outcome)
                report = await self.verifier.verify(iteration, evidence)
                self._last_report = report
                self.state.criteria = {r.id: r.passed for r in report.results}
                self.ledger.append(
                    "verification",
                    iteration=iteration,
                    all_passed=report.all_passed,
                    judge_used=report.judge_used,
                    results=[r.to_dict() for r in report.results],
                )
                if report.all_passed:
                    if outcome.sha:
                        self.state.last_green_sha = outcome.sha
                        self.ledger.append(
                            "checkpoint", sha=outcome.sha, message="verified green", green=True
                        )
                    return await self._finish(
                        MissionStatus.DONE,
                        f"all {len(report.results)} criteria passed at iteration {iteration}",
                        report=report,
                    )
                review = await self.critic.review(self.contract.goal, evidence)
                self._update_streaks(outcome, review)
                self._append_history(iteration, outcome, report, review)
                if self._is_stuck():
                    stuck_result = await self._handle_stuck(iteration, report, review)
                    if stuck_result is not None:
                        return stuck_result
                self._save_state()
            except BudgetExceeded as exc:
                result = await self._handle_budget_exceeded(exc)
                if result is not None:
                    return result
            except ProviderError as exc:
                self._consecutive_provider_errors += 1
                self.ledger.append("provider_error", message=str(exc), status=exc.status)
                await self.bus.emit(E.LOG, level="error", message=f"provider error: {exc}")
                self._append_history(
                    self.budget.iterations,
                    IterationOutcome(iteration=self.budget.iterations, provider_errors=1),
                    None,
                    None,
                )
                self._save_state()
                if self._consecutive_provider_errors > self.contract.limits.max_llm_retries:
                    return await self._finish(
                        MissionStatus.ERROR, f"provider unavailable: {exc}"
                    )
            else:
                self._consecutive_provider_errors = 0
        report = self._last_report
        return await self._finish(MissionStatus.HALTED, "stop requested", report=report)

    # ------------------------------------------------------------------- steps

    async def _load_brief(self) -> None:
        config = self.contract.brief
        brief_dir = self.paths.brief_dir
        if not config.dir or brief_dir is None:
            return
        if not brief_dir.exists():
            await self.bus.emit(
                E.LOG, level="warning", message=f"brief directory not found: {brief_dir}"
            )
            return
        exclude = [path for path in (self.paths.dir, self.paths.contract_path) if path]
        bundle = load_brief(brief_dir, config, exclude=exclude)
        self.brief = bundle
        previous_hashes: Dict[str, str] = {}
        for record in self.ledger.entries():
            if record.get("kind") == "brief_loaded" and record.get("dir") == str(brief_dir):
                previous_hashes = dict(record.get("hashes") or {})
        changed = bool(previous_hashes) and previous_hashes != bundle.hashes()
        self.ledger.append(
            "brief_loaded",
            dir=str(brief_dir),
            brief_file=bundle.brief_file,
            files=[item.to_dict() for item in bundle.files],
            hashes=bundle.hashes(),
            changed_since_last_run=changed,
        )
        manifest_path = self.paths.artifacts_dir / "brief_manifest.json"
        try:
            import json

            manifest_path.write_text(
                json.dumps(
                    {
                        "dir": str(brief_dir),
                        "brief_file": bundle.brief_file,
                        "files": [item.to_dict() for item in bundle.files],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        await self.bus.emit(
            E.BRIEF,
            dir=str(brief_dir),
            brief_file=bundle.brief_file,
            docs=len(bundle.docs()),
            images=len(bundle.images()),
            files=[item.to_dict() for item in bundle.files],
            changed=changed,
        )
        if changed:
            await self.bus.emit(
                E.LOG, level="warning", message="brief files changed since the last run"
            )
        images = bundle.images()
        if images and config.describe_images:
            vision_model = self.contract.model.vision
            if vision_model:
                await describe_images(
                    self.runner,
                    bundle,
                    image_cache_path(self.paths.artifacts_dir),
                    self.bus,
                    model=vision_model,
                )
            else:
                await self.bus.emit(
                    E.LOG,
                    level="warning",
                    message=(
                        f"{len(images)} brief image(s) not described: set model.vision to a "
                        f"vision-capable model to include them as text"
                    ),
                )
        elif images:
            await self.bus.emit(
                E.LOG,
                level="info",
                message=f"{len(images)} brief image(s) ignored (brief.describe_images=false)",
            )

    async def _load_clarifications(self) -> None:
        text = load_clarifications_text(self.paths.artifacts_dir / "clarifications.json")
        if not text:
            return
        self.clarifications_text = text
        self.ledger.append("clarifications_loaded", chars=len(text))
        await self.bus.emit(
            E.LOG, level="info", message="brief interview answers loaded and treated as binding"
        )

    def _brief_context(self) -> str:
        parts: List[str] = []
        if self.brief is not None:
            parts.append(self.brief.manifest(max_chars=self.contract.brief.max_context_chars))
        if self.clarifications_text:
            parts.append(self.clarifications_text)
        return "\n\n".join(parts)

    def _should_plan(self) -> bool:
        if self._force_replan:
            self._force_replan = False
            return True
        return not self.state.plan

    async def _replan(self, iteration: int) -> None:
        previous = self._last_report
        feedback = previous.feedback() if previous else ""
        messages = planner_messages(
            self.contract,
            iteration=iteration,
            previous_plan=self.state.plan,
            criteria_status=render_criteria_status(previous.results) if previous else [],
            verifier_feedback=feedback,
            history=self.state.history,
            human_notes=self._human_notes[-5:],
            brief_context=self._brief_context(),
            clarifications=self.clarifications_text,
        )
        completion = await self.runner.call("planner", messages)
        self.state.plan = completion.message.content.strip()
        self.ledger.append("plan", iteration=iteration, plan=self.state.plan, model=completion.model)
        await self.bus.emit(E.PLAN, iteration=iteration, plan=self.state.plan, model=completion.model)

    async def _run_actor(self, iteration: int) -> IterationOutcome:
        outcome = IterationOutcome(iteration=iteration)
        messages = actor_messages(
            self.contract,
            self.gate,
            plan=self.state.plan,
            criteria_status=render_criteria_status(self._last_report.results) if self._last_report else [],
            verifier_feedback=self._last_report.feedback() if self._last_report else "",
            history=self.state.history,
            human_notes=self._human_notes[-5:],
            tool_names=self.registry.names(),
            brief_context=self._brief_context(),
            clarifications=self.clarifications_text,
        )
        limit = self.contract.limits.max_tool_calls_per_iteration
        while outcome.tool_calls < limit and not self._stop_requested:
            try:
                completion = await self.runner.call("actor", messages, tools=self.registry.schemas())
            except ProviderError:
                outcome.provider_errors += 1
                raise
            content = (completion.message.content or "").strip()
            calls = completion.message.tool_calls
            self.ledger.append(
                "assistant_message",
                iteration=iteration,
                role="actor",
                model=completion.model,
                content=truncate_middle(content, 4000),
                tool_calls=[c.name for c in calls],
                cost_usd=completion.usage.cost_usd,
            )
            if content:
                await self.bus.emit(
                    E.ASSISTANT_MESSAGE, iteration=iteration, text=content, model=completion.model
                )
            if not calls:
                outcome.final_text = content
                break
            messages.append(completion.message)
            for call in calls:
                if self._stop_requested:
                    outcome.halt_reason = "stop requested"
                    return outcome
                outcome.tool_calls += 1
                await self.bus.emit(
                    E.TOOL_STARTED,
                    iteration=iteration,
                    name=call.name,
                    args=call.arguments,
                    call_id=call.id,
                )
                result = await self.registry.execute(call.name, call.arguments, self.tool_ctx)
                rendered = result.render(self.contract.limits.max_output_chars)
                await self.bus.emit(
                    E.TOOL_FINISHED,
                    iteration=iteration,
                    name=call.name,
                    ok=result.ok,
                    output=rendered if result.ok else "",
                    error="" if result.ok else rendered,
                    duration_ms=result.duration_ms,
                    call_id=call.id,
                )
                self.ledger.append(
                    "tool_call",
                    iteration=iteration,
                    name=call.name,
                    args=truncate_middle(str(call.arguments), 800),
                    ok=result.ok,
                    duration_ms=result.duration_ms,
                    output=truncate_middle(rendered, 2000),
                )
                if not result.ok and result.error.startswith("permission denied"):
                    outcome.permission_denials += 1
                    halt = await self._handle_permission_denied(result.error)
                    if halt:
                        outcome.halt_reason = halt
                        return outcome
                messages.append(
                    ChatMessage.tool_result(call.id, call.name, result.as_message_content())
                )
                outcome.tool_outputs.append((call.name, rendered))
                if outcome.tool_calls >= limit:
                    break
        outcome.changed_files = await self.checkpoints.changed_files() if self.checkpoints.ready else []
        outcome.changes_known = bool(self.checkpoints.ready)
        return outcome

    async def _checkpoint(self, iteration: int) -> Optional[str]:
        sha = await self.checkpoints.commit(f"deeploop: iteration {iteration}")
        if sha:
            self.ledger.append("checkpoint", sha=sha, message=f"iteration {iteration}")
        return sha

    async def _build_evidence(self, outcome: IterationOutcome) -> Evidence:
        diff = ""
        if self.checkpoints.ready:
            diff = await self.checkpoints.diff_since(self.state.last_green_sha, max_chars=8000)
        return Evidence(
            goal=self.contract.goal,
            iteration=outcome.iteration,
            diff=diff,
            changed_files=outcome.changed_files,
            recent_outputs=outcome.tool_outputs,
            final_text=outcome.final_text,
            criteria_status=(
                render_criteria_status(self._last_report.results) if self._last_report else []
            ),
            clarifications=self.clarifications_text,
        )

    # ------------------------------------------------------------------ verdicts

    def _update_streaks(self, outcome: IterationOutcome, review: ProgressReview) -> None:
        if outcome.changes_known:
            self.state.no_change_streak = (
                0 if outcome.changed_files else self.state.no_change_streak + 1
            )
        signature = ""
        for name, output in outcome.tool_outputs:
            found = first_error_line(output)
            if found:
                signature = stable_hash(f"{name}:{found}")
                break
        if signature and signature == self.state.last_error_signature:
            self.state.error_streak += 1
        elif signature:
            self.state.error_streak = 1
            self.state.last_error_signature = signature
        else:
            self.state.error_streak = 0
            self.state.last_error_signature = ""
        self.state.no_progress_streak = (
            self.state.no_progress_streak + 1 if not review.progress else 0
        )

    def _append_history(
        self,
        iteration: int,
        outcome: IterationOutcome,
        report: Optional[VerificationReport],
        review: Optional[ProgressReview],
    ) -> None:
        bits = [f"iter {iteration}: {len(outcome.changed_files)} file(s) changed"]
        if not outcome.changes_known:
            bits = [f"iter {iteration}: file changes unknown (checkpointing disabled)"]
        if report is not None:
            failed = [r.id for r in report.failed]
            bits.append("verify: " + ("ALL PASS" if report.all_passed else f"FAIL {', '.join(failed)}"))
            for result in report.failed[:2]:
                if result.reason:
                    bits.append(one_line(result.reason, 120))
        if review is not None and not review.progress:
            bits.append(f"critic: no progress ({one_line(review.reason, 100)})")
        if outcome.permission_denials:
            bits.append(f"{outcome.permission_denials} permission denial(s)")
        self.state.history.append(" | ".join(bits))

    def _is_stuck(self) -> bool:
        threshold = self.contract.limits.stuck_after
        return (
            self.state.no_change_streak >= threshold
            or self.state.error_streak >= threshold
            or self.state.no_progress_streak >= threshold
        )

    def _stuck_reason(self) -> str:
        reasons = []
        if self.state.no_change_streak >= self.contract.limits.stuck_after:
            reasons.append(f"{self.state.no_change_streak} iterations with no file changes")
        if self.state.error_streak >= self.contract.limits.stuck_after:
            reasons.append(f"the same error repeated {self.state.error_streak} times")
        if self.state.no_progress_streak >= self.contract.limits.stuck_after:
            reasons.append(f"critic saw no progress for {self.state.no_progress_streak} iterations")
        return "; ".join(reasons) or "no measurable progress"

    async def _handle_stuck(
        self, iteration: int, report: VerificationReport, review: ProgressReview
    ) -> Optional[MissionResult]:
        reason = self._stuck_reason()
        policy = self.contract.escalation.on_stuck
        await self.bus.emit(E.STUCK, iteration=iteration, reason=reason, policy=policy)
        self.ledger.append("stuck", iteration=iteration, reason=reason, policy=policy)
        action = policy
        if policy == "ask":
            action = await self.human.ask(f"stuck: {reason}", ["replan", "halt"])
        if action == "replan" and self.state.replans < self.contract.limits.max_replans:
            self.state.replans += 1
            self._force_replan = True
            self.state.no_change_streak = 0
            self.state.error_streak = 0
            self.state.no_progress_streak = 0
            self.ledger.append("replan", iteration=iteration, replans=self.state.replans)
            await self.bus.emit(
                E.STUCK, iteration=iteration, reason=reason, policy="replan",
                replans=self.state.replans,
            )
            return None
        summary = f"stuck after {iteration} iterations: {reason}"
        if review.hint:
            summary += f" (critic hint: {review.hint})"
        return await self._finish(MissionStatus.HALTED, summary, report=report)

    async def _handle_permission_denied(self, error: str) -> Optional[str]:
        policy = self.contract.escalation.on_permission_denied
        if policy == "deny":
            return None
        if policy == "halt":
            return f"permission denied under halt policy: {one_line(error, 200)}"
        answer = await self.human.ask(f"permission denied: {one_line(error, 200)}", ["continue", "halt"])
        if answer == "halt":
            return f"permission denied: {one_line(error, 200)}"
        return None

    async def _handle_budget_exceeded(self, exc: BudgetExceeded) -> Optional[MissionResult]:
        policy = self.contract.escalation.on_budget_exhausted
        await self.bus.emit(
            E.ESCALATION,
            kind="budget",
            reason=str(exc),
            limit_kind=exc.kind,
            policy=policy,
            budget=self.budget.snapshot(),
        )
        self.ledger.append("escalation", kind="budget", reason=str(exc), policy=policy)
        action = policy
        if policy == "ask":
            action = await self.human.ask(f"budget exhausted ({exc.kind}): {exc}", ["halt", "continue"])
            await self.bus.emit(E.ESCALATION_RESOLVED, kind="budget", action=action)
        if action == "continue":
            record = self.budget.extend(
                usd=self.contract.budget.max_usd * 0.5,
                iterations=5,
                minutes=10,
                reason="human approved continuation",
            )
            self.ledger.append("budget_extension", **record)
            await self.bus.emit(E.BUDGET, **self.budget.snapshot())
            return None
        if action == "degrade" and self.contract.model.fallback_actor:
            self.runner.actor_model_override = self.contract.model.fallback_actor
            record = self.budget.extend(
                usd=self.contract.budget.max_usd * 0.5,
                iterations=5,
                minutes=10,
                reason=f"degrade to {self.contract.model.fallback_actor}",
            )
            self.ledger.append("budget_extension", **record)
            await self.bus.emit(
                E.LOG,
                level="warning",
                message=f"degrading actor model to {self.contract.model.fallback_actor}",
            )
            return None
        report = self._last_report
        return await self._finish(
            MissionStatus.BUDGET_EXHAUSTED,
            f"budget exhausted ({exc.kind}): {exc}",
            report=report,
        )

    async def _finish(
        self,
        status: MissionStatus,
        summary: str,
        report: Optional[VerificationReport] = None,
    ) -> MissionResult:
        self.state.status = status.value
        self.state.spent_usd = round(self.budget.spent_usd, 6)
        self._save_state()
        self.ledger.append(
            "mission_finished",
            status=status.value,
            summary=summary,
            iterations=self.budget.iterations,
            spent_usd=round(self.budget.spent_usd, 6),
            elapsed_seconds=round(self.budget.elapsed_seconds, 1),
        )
        result = MissionResult(
            status=status,
            iterations=self.budget.iterations,
            spent_usd=round(self.budget.spent_usd, 6),
            elapsed_seconds=round(self.budget.elapsed_seconds, 1),
            summary=summary,
            report=report,
        )
        self._finished = True
        await self.bus.emit(E.MISSION_FINISHED, **result.to_dict())
        return result

    def _save_state(self) -> None:
        self.state.spent_usd = round(self.budget.spent_usd, 6)
        self.state.save(self.paths.state_path)
