"""TUI widgets. Blocks are appended to the mission log as the loop runs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ProgressBar, Static

from ..budget import BudgetTracker
from ..contract import TaskContract
from ..permissions import PermissionGate


class MissionHeader(Static):
    def __init__(self) -> None:
        super().__init__("deeploop", id="mission-header")

    def set_goal(self, goal: str, workspace: str) -> None:
        self.update(
            f"[bold magenta]deeploop[/]  [bold]{escape(goal)}[/]  [dim]{escape(workspace)}[/]"
        )


class Block(Static):
    """Base class for log blocks. Content is plain text unless markup=True."""

    def __init__(self, content: str = "", *, classes: str = "block", markup: bool = False) -> None:
        super().__init__(content, classes=classes, markup=markup)


class IterationDivider(Block):
    def __init__(self, iteration: int, max_iterations: int, spent_usd: float) -> None:
        super().__init__(
            f"── iteration {iteration}/{max_iterations}  [dim]spent ${spent_usd:.4f}[/] ──",
            classes="block divider",
            markup=True,
        )


class PlanBlock(Block):
    def __init__(self, plan: str, model: str) -> None:
        super().__init__("", classes="block plan", markup=True)
        self.plan = plan
        self.model = model

    def on_mount(self) -> None:
        body = escape(self.plan).replace("\n", "\n   ")
        self.update(f"[bold cyan]◇ plan[/] [dim]({escape(self.model)})[/]\n   {body}")


class AssistantBlock(Block):
    def __init__(self, text: str, model: str = "") -> None:
        super().__init__(f"▸ {text}", classes="block assistant")
        self._assistant_text = text

    def append(self, chunk: str) -> None:
        self._assistant_text += chunk
        self.update(f"▸ {self._assistant_text}")


class ToolBlock(Block):
    def __init__(self, name: str, args: Dict[str, Any], call_id: str) -> None:
        self.tool_name = name
        self.tool_args = args
        self.call_id = call_id
        self.finished = False
        preview = _format_args(args)
        super().__init__(f"⚒ {name} {preview}", classes="block tool pending")

    def finish(self, ok: bool, output: str, error: str, duration_ms: int, max_lines: int = 6) -> None:
        self.finished = True
        glyph = "✔" if ok else "✘"
        self.remove_class("pending")
        self.add_class("ok" if ok else "failed")
        body = output if ok else error
        lines = body.splitlines()
        shown = lines[:max_lines]
        extra = len(lines) - len(shown)
        rendered = "\n".join(f"    {line}" for line in shown)
        if extra > 0:
            rendered += f"\n    [dim]… {extra} more line(s)[/dim]"
        self.update(
            f"{glyph} {self.tool_name} {_format_args(self.tool_args)}  [dim]{duration_ms}ms[/]\n{rendered}"
        )


class VerifyBlock(Block):
    def __init__(self, results: List[Dict[str, Any]], notes: str = "") -> None:
        lines: List[str] = []
        for result in results:
            ok = result.get("passed")
            glyph = "✓" if ok else "✗"
            color = "green" if ok else "red"
            reason = (result.get("reason") or "").splitlines()
            reason_text = reason[0][:140] if reason else ""
            lines.append(
                f"[{color}]{glyph}[/] {escape(str(result.get('id')))} "
                f"[dim]({result.get('kind')}, {result.get('duration_ms')}ms)[/] {escape(reason_text)}"
            )
        if notes:
            lines.append(f"[dim]note: {escape(notes)}[/]")
        super().__init__("verify\n" + "\n".join(lines), classes="block verify", markup=True)


class CriticBlock(Block):
    def __init__(self, progress: bool, reason: str, hint: str = "") -> None:
        color = "dim" if progress else "yellow"
        text = f"[{color}]◌ critic: {'progress' if progress else 'no progress'}[/] {escape(reason)}"
        if hint:
            text += f"\n   [dim]hint: {escape(hint)}[/]"
        super().__init__(text, classes="block critic", markup=True)


class NoticeBlock(Block):
    def __init__(self, text: str, level: str = "info", markup: bool = False) -> None:
        super().__init__(text, classes=f"block notice {level}", markup=markup)


class BudgetPanel(Vertical):
    def __init__(self, budget: BudgetTracker) -> None:
        super().__init__(id="budget-panel", classes="panel")
        self.budget = budget

    def compose(self) -> ComposeResult:
        yield Static("BUDGET", classes="panel-title")
        yield ProgressBar(total=self.budget.budget.max_usd, show_eta=False, id="budget-bar")
        yield Static("", id="budget-text", markup=True)

    def refresh_panel(self, data: Optional[Dict[str, Any]] = None) -> None:
        data = data or self.budget.snapshot()
        bar = self.query_one("#budget-bar", ProgressBar)
        bar.update(total=data["max_usd"], progress=min(data["spent_usd"], data["max_usd"]))
        elapsed = data["elapsed_seconds"]
        minutes, seconds = divmod(int(elapsed), 60)
        self.query_one("#budget-text", Static).update(
            f"${data['spent_usd']:.4f} / ${data['max_usd']:.2f}\n"
            f"iters {data['iterations']}/{data['max_iterations']}  "
            f"time {minutes}m{seconds:02d}s / {data['max_wall_clock_minutes']:.0f}m\n"
            f"[dim]{data['prompt_tokens']} in / {data['completion_tokens']} out tokens"
            f" ({data['calls']} calls)[/]"
        )


class CriteriaPanel(Static):
    def __init__(self, contract: TaskContract) -> None:
        self.contract = contract
        super().__init__("", id="criteria-panel", classes="panel", markup=True)

    def on_mount(self) -> None:
        self.set_results({})

    def set_results(self, results: Dict[str, bool]) -> None:
        lines = ["[bold]SUCCESS CRITERIA[/]"]
        for criterion in self.contract.success_criteria:
            if criterion.id in results:
                ok = results[criterion.id]
                glyph = "[green]✓[/]" if ok else "[red]✗[/]"
            else:
                glyph = "[dim]○[/]"
            kind = f"[dim]{criterion.kind}[/]"
            lines.append(f"{glyph} {escape(criterion.id)} {kind}")
        self.update("\n".join(lines))


class PermissionsPanel(Static):
    def __init__(self, gate: PermissionGate) -> None:
        summary = gate.summary()
        network = "[red]blocked[/]" if not summary["network"] else "[yellow]allowed[/]"
        allow_count = (
            "all" if summary["allow_all_commands"] else str(len(summary["allow_commands"]))
        )
        lines = [
            "[bold]PERMISSIONS[/]",
            f"network: {network}",
            f"allowed commands: {allow_count}",
            f"[dim]{escape(summary['workspace'])}[/]",
        ]
        if summary["require_approval"]:
            lines.append(f"[yellow]approval required: {escape(', '.join(summary['require_approval']))}[/]")
        super().__init__("\n".join(lines), id="permissions-panel", classes="panel", markup=True)


class BriefPanel(Static):
    def __init__(self, brief_dir: Optional[Any] = None) -> None:
        self.brief_dir = brief_dir
        super().__init__("", id="brief-panel", classes="panel", markup=True)

    def on_mount(self) -> None:
        self.set_data({})

    def set_data(self, data: Dict[str, Any]) -> None:
        if not self.brief_dir and not data:
            self.update("[bold]BRIEF[/]\n[dim]none configured[/]")
            return
        directory = data.get("dir") or str(self.brief_dir)
        lines = ["[bold]BRIEF[/]", f"[dim]{escape(directory)}[/]"]
        docs = data.get("docs")
        if docs is None:
            lines.append("[dim]loading…[/]")
        else:
            lines.append(f"{docs} doc(s), {data.get('images', 0)} image(s)")
            if data.get("brief_file"):
                lines.append(f"[dim]{escape(str(data['brief_file']))}[/]")
            if data.get("changed"):
                lines.append("[yellow]changed since last run[/]")
        self.update("\n".join(lines))


class StatusBar(Static):
    def __init__(self) -> None:
        self._status = "starting"
        self._iteration = 0
        self._max_iterations = 0
        self._spent = 0.0
        self._max_usd = 0.0
        self._elapsed = 0
        super().__init__("", id="status-bar", markup=True)

    def set_state(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
        self._refresh_status()

    def _refresh_status(self) -> None:
        color = {
            "done": "green",
            "error": "red",
            "budget_exhausted": "red",
            "halted": "yellow",
            "running": "cyan",
        }.get(self._status, "cyan")
        minutes, seconds = divmod(int(self._elapsed), 60)
        self.update(
            f"[{color}]● {self._status}[/]  "
            f"iter {self._iteration}/{self._max_iterations}  "
            f"${self._spent:.4f}/${self._max_usd:.2f}  "
            f"{minutes}m{seconds:02d}s   "
            f"[dim]s stop · i interject · esc leave input · q quit[/]"
        )


def _format_args(args: Dict[str, Any], limit: int = 140) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(args)
    return text[:limit] + ("…" if len(text) > limit else "")
