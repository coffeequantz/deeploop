"""Headless renderer: the same event stream the TUI consumes, printed as text."""

from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich.text import Text

from ..events import Event
from ..util import one_line

GLYPHS = {
    "iteration": "──",
    "plan": "◇",
    "tool": "⚒",
    "ok": "✔",
    "fail": "✘",
    "verify_ok": "✓",
    "verify_fail": "✗",
    "critic": "◌",
    "stuck": "⚠",
    "escalation": "⛔",
    "done": "■",
}


class ConsoleRenderer:
    """Subscribe to a mission's event bus and print a compact transcript."""

    def __init__(self, console: Console = None, verbose: bool = False) -> None:
        self.console = console or Console()
        self.verbose = verbose
        self._last_budget_iteration = -1
        self._last_budget_spent = -1.0

    def __call__(self, event: Event) -> None:
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler is not None:
            handler(event.data)

    def _print(self, text: Text) -> None:
        self.console.print(text, highlight=False, soft_wrap=True)

    def _on_mission_started(self, data: Dict[str, Any]) -> None:
        goal = one_line(data.get("goal", ""), 100)
        self._print(Text.assemble(("deeploop", "bold magenta"), "  ", (goal, "bold")))
        perms = data.get("permissions", {})
        budget = data.get("budget", {})
        self._print(
            Text(
                f"  workspace={perms.get('workspace')} network={perms.get('network')} "
                f"budget=${budget.get('max_usd')} iters={budget.get('max_iterations')}",
                style="dim",
            )
        )

    def _on_iteration_started(self, data: Dict[str, Any]) -> None:
        self._print(
            Text(
                f"{GLYPHS['iteration']} iteration {data.get('iteration')}/{data.get('max_iterations')} "
                f"{GLYPHS['iteration'] * 3}",
                style="bold blue",
            )
        )

    def _on_plan(self, data: Dict[str, Any]) -> None:
        plan = data.get("plan", "")
        self._print(Text(f"{GLYPHS['plan']} plan", style="bold cyan"))
        for line in plan.splitlines():
            self._print(Text(f"   {line}", style="cyan"))

    def _on_assistant_message(self, data: Dict[str, Any]) -> None:
        text = (data.get("text") or "").strip()
        if text:
            self._print(Text(f"▸ {text}", style="white"))

    def _on_tool_started(self, data: Dict[str, Any]) -> None:
        args = one_line(str(data.get("args", {})), 160)
        self._print(Text(f"{GLYPHS['tool']} {data.get('name')} {args}", style="yellow"))

    def _on_tool_finished(self, data: Dict[str, Any]) -> None:
        ok = data.get("ok")
        glyph = GLYPHS["ok"] if ok else GLYPHS["fail"]
        style = "green" if ok else "red"
        header = f"  {glyph} {data.get('name')} ({data.get('duration_ms')}ms)"
        self._print(Text(header, style=style))
        if self.verbose or not ok:
            body = data.get("output") or data.get("error") or ""
            for line in body.splitlines()[: (None if self.verbose else 6)]:
                self._print(Text(f"    {line}", style="dim"))

    def _on_verification(self, data: Dict[str, Any]) -> None:
        for result in data.get("results", []):
            ok = result.get("passed")
            glyph = GLYPHS["verify_ok"] if ok else GLYPHS["verify_fail"]
            style = "green" if ok else "red"
            reason = one_line(result.get("reason", ""), 120)
            self._print(Text(f"{glyph} {result.get('id')} ({result.get('kind')}) {reason}", style=style))
        if data.get("notes"):
            self._print(Text(f"  note: {data['notes']}", style="dim"))

    def _on_critic(self, data: Dict[str, Any]) -> None:
        style = "dim" if data.get("progress") else "yellow"
        self._print(
            Text(
                f"{GLYPHS['critic']} critic: progress={data.get('progress')} "
                f"{one_line(data.get('reason', ''), 120)}",
                style=style,
            )
        )

    def _on_budget(self, data: Dict[str, Any]) -> None:
        spent = float(data.get("spent_usd", 0.0) or 0.0)
        iterations = int(data.get("iterations", 0) or 0)
        changed_iteration = iterations != self._last_budget_iteration
        grew = abs(spent - self._last_budget_spent) >= 0.01
        if not (self.verbose or changed_iteration or grew):
            return
        self._last_budget_iteration = iterations
        self._last_budget_spent = spent
        self._print(
            Text(
                f"  budget ${spent:.4f}/${data.get('max_usd', 0):.2f} "
                f"iters {iterations}/{data.get('max_iterations', 0)} "
                f"tokens {data.get('prompt_tokens', 0)}in/{data.get('completion_tokens', 0)}out",
                style="dim",
            )
        )

    def _on_checkpoint(self, data: Dict[str, Any]) -> None:
        if self.verbose:
            self._print(Text(f"  checkpoint {str(data.get('sha'))[:10]} {data.get('message')}", style="dim"))

    def _on_rollback(self, data: Dict[str, Any]) -> None:
        self._print(Text(f"  rollback to {str(data.get('sha'))[:10]}", style="magenta"))

    def _on_stuck(self, data: Dict[str, Any]) -> None:
        text = f"{GLYPHS['stuck']} stuck: {data.get('reason')} (policy={data.get('policy')})"
        self._print(Text(text, style="bold yellow"))

    def _on_escalation(self, data: Dict[str, Any]) -> None:
        self._print(Text(f"{GLYPHS['escalation']} escalation: {data.get('reason')}", style="bold red"))

    def _on_interjection(self, data: Dict[str, Any]) -> None:
        self._print(Text(f"human ▸ {data.get('text')}", style="bold magenta"))

    def _on_log(self, data: Dict[str, Any]) -> None:
        level = data.get("level", "info")
        style = {"info": "dim", "warning": "yellow", "error": "bold red"}.get(level, "dim")
        self._print(Text(f"[{level}] {data.get('message')}", style=style))

    def _on_mission_finished(self, data: Dict[str, Any]) -> None:
        status = data.get("status", "unknown")
        style = "bold green" if status == "done" else "bold red"
        self._print(
            Text(
                f"{GLYPHS['done']} mission {status}: {data.get('summary')} "
                f"({data.get('iterations')} iterations, ${data.get('spent_usd', 0):.4f}, "
                f"{data.get('elapsed_seconds', 0):.0f}s)",
                style=style,
            )
        )


def render_report_table(rows: List[Dict[str, Any]], console: Console = None) -> None:
    console = console or Console()
    from rich.table import Table

    table = Table(title="deeploop missions")
    table.add_column("mission")
    table.add_column("status")
    table.add_column("iters", justify="right")
    table.add_column("spend", justify="right")
    table.add_column("criteria")
    for row in rows:
        table.add_row(
            row.get("mission", ""),
            row.get("status", ""),
            str(row.get("iterations", 0)),
            f"${row.get('spent_usd', 0):.4f}",
            row.get("criteria", ""),
        )
    console.print(table)
