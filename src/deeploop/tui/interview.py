"""Brief interview screen: the human answers the clarifying questions before a
contract is proposed."""

from __future__ import annotations

from typing import List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Static

from ..interview import Question

InterviewResult = Optional[List[str]]


class InterviewApp(App[InterviewResult]):
    """Returns the answers in question order, [] to skip all, or None to cancel."""

    TITLE = "deeploop · brief interview"
    BINDINGS = [
        Binding("escape", "skip_all", "skip"),
        Binding("ctrl+c", "cancel", "cancel", show=False),
    ]

    def __init__(self, questions: List[Question], cost_usd: float = 0.0) -> None:
        super().__init__()
        self.questions = questions
        self.cost_usd = cost_usd

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold magenta]deeploop[/]  {len(self.questions)} clarifying question(s)  "
            "[dim](answers become binding contract input; enter moves on, esc skips)[/]",
            id="interview-title",
        )
        with VerticalScroll(id="interview-body"):
            for index, question in enumerate(self.questions):
                label = f"[{question.affects}] {question.question}"
                if question.hint():
                    label += f"\n[dim]{question.hint()}[/]"
                yield Static(label, classes="interview-question", markup=True)
                yield Input(
                    placeholder=question.default or "your answer",
                    id=f"q-{index}",
                )
        yield Static(self._footer(), id="interview-footer")
        with Horizontal(id="interview-actions"):
            yield Button("Submit answers", id="submit", variant="success")
            yield Button("Skip all", id="skip", variant="warning")
            yield Button("Cancel", id="cancel", variant="error")

    def _footer(self) -> str:
        if self.cost_usd:
            cost = f"questions cost ${self.cost_usd:.4f}"
        else:
            cost = "questions cost: free (mock provider)"
        return f"[dim]{cost} · blank answers are recorded as skipped, with your default assumed[/]"

    def on_mount(self) -> None:
        first = self.query_one("#q-0", Input)
        first.focus()

    def _answers(self) -> List[str]:
        values: List[str] = []
        for index in range(len(self.questions)):
            values.append(self.query_one(f"#q-{index}", Input).value)
        return values

    def _submit(self) -> None:
        self.exit(self._answers())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        index = int(event.input.id.split("-")[-1]) if event.input.id else 0
        if index + 1 < len(self.questions):
            self.query_one(f"#q-{index + 1}", Input).focus()
        else:
            self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self._submit()
        elif event.button.id == "skip":
            self.exit([])
        else:
            self.exit(None)

    def action_skip_all(self) -> None:
        self.exit([])

    def action_cancel(self) -> None:
        self.exit(None)
