"""Textual TUI: mission log, budget sidebar, criteria, escalation prompt.

Layout is inspired by Kilo/OpenCode-style coding agent TUIs: a scrollback of
plan/tool/verification blocks, a live sidebar of enforced limits, and a single
input line for human interjection or escalation answers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input

from ..controller import MissionResult, MissionStatus
from ..events import Event
from ..human import HumanInterface
from ..runtime import Runtime
from .widgets import (
    AssistantBlock,
    BriefPanel,
    BudgetPanel,
    CriteriaPanel,
    CriticBlock,
    IterationDivider,
    MissionHeader,
    NoticeBlock,
    PermissionsPanel,
    PlanBlock,
    StatusBar,
    ToolBlock,
    VerifyBlock,
)


class TUIHuman(HumanInterface):
    def __init__(self) -> None:
        self._handler: Optional[Callable[[str, List[str]], Awaitable[str]]] = None

    def bind(self, handler: Callable[[str, List[str]], Awaitable[str]]) -> None:
        self._handler = handler

    async def ask(self, reason: str, options: List[str]) -> str:
        if self._handler is None:
            return options[0]
        return await self._handler(reason, options)


class EventMessage(Message):
    def __init__(self, event: Event) -> None:
        super().__init__()
        self.event = event


class MissionDone(Message):
    def __init__(self, result: MissionResult) -> None:
        super().__init__()
        self.result = result


class DeepLoopApp(App[None]):
    CSS_PATH = "theme.tcss"
    TITLE = "deeploop"
    SUB_TITLE = "budget-first agent harness"
    BINDINGS = [
        Binding("q", "quit_app", "quit"),
        Binding("s", "stop_mission", "stop"),
        Binding("i", "focus_prompt", "interject"),
        Binding("escape", "blur_prompt", "leave input", show=False),
        Binding("ctrl+c", "stop_mission", "stop", show=False),
    ]

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self.runtime = runtime
        self._tool_blocks: Dict[str, ToolBlock] = {}
        self._assistant_block: Optional[AssistantBlock] = None
        self._assistant_iteration = -1
        self._escalation: Optional[Tuple[str, List[str], asyncio.Future]] = None
        self._finished = False

    def compose(self) -> ComposeResult:
        yield MissionHeader()
        with Horizontal(id="body"):
            with VerticalScroll(id="log"):
                yield NoticeBlock("mission starting…", level="info")
            with Vertical(id="sidebar"):
                yield BudgetPanel(self.runtime.budget)
                yield CriteriaPanel(self.runtime.contract)
                yield PermissionsPanel(self.runtime.gate)
                yield BriefPanel(self.runtime.paths.brief_dir)
        yield StatusBar()
        yield Input(
            placeholder="interject guidance, or answer an escalation: continue / halt / replan",
            id="prompt",
        )

    def on_mount(self) -> None:
        contract = self.runtime.contract
        self.query_one(MissionHeader).set_goal(contract.goal, str(self.runtime.paths.workspace))
        self.query_one(StatusBar).set_state(
            status="running",
            max_iterations=self.runtime.budget.budget.max_iterations,
            max_usd=self.runtime.budget.budget.max_usd,
        )
        self.runtime.bus.subscribe(self._on_event)
        if isinstance(self.runtime.human, TUIHuman):
            self.runtime.human.bind(self._ask_human)
        self.set_interval(1.0, self._tick)
        self._run_mission()

    def on_unmount(self) -> None:
        self.runtime.bus.unsubscribe(self._on_event)

    @work(exclusive=True)
    async def _run_mission(self) -> None:
        try:
            result = await self.runtime.controller.run()
        except Exception as exc:  # noqa: BLE001 - never let the UI hang on a crash
            result = MissionResult(
                status=MissionStatus.ERROR,
                iterations=self.runtime.budget.iterations,
                spent_usd=round(self.runtime.budget.spent_usd, 6),
                elapsed_seconds=round(self.runtime.budget.elapsed_seconds, 1),
                summary=f"{type(exc).__name__}: {exc}",
            )
        self.post_message(MissionDone(result))

    # ------------------------------------------------------------------ events

    def _on_event(self, event: Event) -> None:
        self.post_message(EventMessage(event))

    async def on_event_message(self, message: EventMessage) -> None:
        event = message.event
        handler = getattr(self, f"_handle_{event.kind}", None)
        if handler is not None:
            await handler(event.data)

    async def _append(self, widget: Any) -> None:
        log = self.query_one("#log", VerticalScroll)
        await log.mount(widget)
        log.scroll_end(animate=False)

    async def _handle_mission_started(self, data: Dict[str, Any]) -> None:
        self.query_one(StatusBar).set_state(status="running")
        perms = data.get("permissions", {})
        await self._append(
            NoticeBlock(
                f"goal: {data.get('goal')}\n"
                f"workspace: {perms.get('workspace')}  network: "
                f"{'allowed' if perms.get('network') else 'blocked'}  "
                f"budget: ${data.get('budget', {}).get('max_usd')} / "
                f"{data.get('budget', {}).get('max_iterations')} iterations",
                level="info",
            )
        )

    async def _handle_iteration_started(self, data: Dict[str, Any]) -> None:
        self.query_one(StatusBar).set_state(
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 0),
            spent=self.runtime.budget.spent_usd,
        )
        await self._append(
            IterationDivider(
                data.get("iteration", 0),
                data.get("max_iterations", 0),
                self.runtime.budget.spent_usd,
            )
        )

    async def _handle_plan(self, data: Dict[str, Any]) -> None:
        await self._append(PlanBlock(data.get("plan", ""), data.get("model", "")))

    async def _handle_assistant_message(self, data: Dict[str, Any]) -> None:
        iteration = data.get("iteration", -1)
        text = (data.get("text") or "").strip()
        if not text:
            return
        if self._assistant_block is not None and iteration == self._assistant_iteration:
            self._assistant_block.append("\n\n" + text)
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)
        else:
            self._assistant_block = AssistantBlock(text, data.get("model", ""))
            self._assistant_iteration = iteration
            await self._append(self._assistant_block)

    async def _handle_tool_started(self, data: Dict[str, Any]) -> None:
        block = ToolBlock(data.get("name", "?"), data.get("args", {}), data.get("call_id", ""))
        self._tool_blocks[data.get("call_id", "")] = block
        await self._append(block)

    async def _handle_tool_finished(self, data: Dict[str, Any]) -> None:
        block = self._tool_blocks.pop(data.get("call_id", ""), None)
        if block is None:
            return
        block.finish(
            bool(data.get("ok")),
            data.get("output") or "",
            data.get("error") or "",
            int(data.get("duration_ms", 0)),
        )
        self.query_one("#log", VerticalScroll).scroll_end(animate=False)

    async def _handle_verification(self, data: Dict[str, Any]) -> None:
        await self._append(VerifyBlock(data.get("results", []), data.get("notes", "")))
        results = {r.get("id", "?"): bool(r.get("passed")) for r in data.get("results", [])}
        self.query_one(CriteriaPanel).set_results(results)

    async def _handle_critic(self, data: Dict[str, Any]) -> None:
        await self._append(
            CriticBlock(bool(data.get("progress")), data.get("reason", ""), data.get("hint", ""))
        )

    async def _handle_budget(self, data: Dict[str, Any]) -> None:
        self.query_one(BudgetPanel).refresh_panel(data)
        self.query_one(StatusBar).set_state(
            spent=data.get("spent_usd", 0.0),
            elapsed=data.get("elapsed_seconds", 0.0),
        )

    async def _handle_brief_loaded(self, data: Dict[str, Any]) -> None:
        self.query_one(BriefPanel).set_data(data)

    async def _handle_stuck(self, data: Dict[str, Any]) -> None:
        await self._append(
            NoticeBlock(
                f"stuck: {data.get('reason')} (policy: {data.get('policy')})", level="warning"
            )
        )

    async def _handle_escalation(self, data: Dict[str, Any]) -> None:
        await self._append(
            NoticeBlock(
                f"escalation [{data.get('kind')}]: {data.get('reason')}\n"
                f"policy: {data.get('policy')}",
                level="error",
            )
        )

    async def _handle_interjection(self, data: Dict[str, Any]) -> None:
        await self._append(NoticeBlock(f"human ▸ {data.get('text')}", level="info"))

    async def _handle_log(self, data: Dict[str, Any]) -> None:
        level = data.get("level", "info")
        await self._append(NoticeBlock(str(data.get("message", "")), level=level))

    async def _handle_mission_finished(self, data: Dict[str, Any]) -> None:
        status = data.get("status", "unknown")
        level = "success" if status == "done" else "error"
        self._finished = True
        self.query_one(StatusBar).set_state(status=status)
        await self._append(
            NoticeBlock(
                f"mission {status}: {data.get('summary')}\n"
                f"iterations: {data.get('iterations')}  "
                f"spent: ${data.get('spent_usd', 0):.4f}  "
                f"elapsed: {data.get('elapsed_seconds', 0):.0f}s\n"
                f"ledger: {self.runtime.paths.ledger_path}",
                level=level,
            )
        )

    async def on_mission_done(self, message: MissionDone) -> None:
        if not self._finished:
            self.query_one(StatusBar).set_state(status=message.result.status.value)
            await self._append(
                NoticeBlock(f"mission ended: {message.result.summary}", level="warning")
            )

    # ------------------------------------------------------------- interaction

    def _tick(self) -> None:
        if not self._finished:
            self.query_one(StatusBar).set_state(elapsed=self.runtime.budget.elapsed_seconds)

    async def _ask_human(self, reason: str, options: List[str]) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._escalation = (reason, options, future)
        await self._append(
            NoticeBlock(
                f"human input required: {reason}\n"
                f"reply with: {' / '.join(options)}",
                level="warning",
            )
        )
        self.query_one("#prompt", Input).focus()
        return await future

    def _match_option(self, value: str, options: List[str]) -> str:
        value = value.strip().lower()
        if value in options:
            return value
        for option in options:
            if value and option.startswith(value):
                return option
        return options[0]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if self._escalation is not None:
            reason, options, future = self._escalation
            self._escalation = None
            if future.done():
                return
            action = self._match_option(value, options)
            future.set_result(action)
            return
        if value:
            self.runtime.controller.interject(value)

    def action_stop_mission(self) -> None:
        if self._finished:
            return
        self.runtime.controller.request_stop()
        self.query_one(StatusBar).set_state(status="stopping")

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_blur_prompt(self) -> None:
        self.query_one("#log", VerticalScroll).focus()

    def action_quit_app(self) -> None:
        if not self._finished:
            self.runtime.controller.request_stop()
        self.exit()


def run_tui(runtime: Runtime) -> None:
    DeepLoopApp(runtime).run()
