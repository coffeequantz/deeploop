"""Interactive review of a proposed mission contract, before anything is spent."""

from __future__ import annotations

from typing import Optional, Tuple

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from ..brief import BriefBundle
from ..proposal import ContractDraft

ProposalResult = Optional[Tuple[str, Optional[str]]]


class ProposalApp(App[ProposalResult]):
    """Shows the derived contract and returns ("accept"|"revise"|"cancel", note)."""

    TITLE = "deeploop · contract proposal"
    BINDINGS = [
        Binding("escape", "cancel", "cancel"),
        Binding("ctrl+c", "cancel", "cancel", show=False),
    ]

    def __init__(self, draft: ContractDraft, bundle: BriefBundle, cost_usd: float = 0.0) -> None:
        super().__init__()
        self.draft = draft
        self.bundle = bundle
        self.cost_usd = cost_usd

    def on_mount(self) -> None:
        self.query_one("#accept", Button).focus()

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold magenta]deeploop[/]  proposed mission contract  "
            "[dim](nothing runs until you accept)[/]",
            id="proposal-title",
        )
        with Horizontal(id="proposal-body"):
            with VerticalScroll(id="proposal-main"):
                yield Static(self.draft.summary(), id="proposal-summary", markup=False)
                for warning in self.draft.warnings:
                    yield Static(f"warning: {warning}", classes="proposal-warning", markup=False)
            with Vertical(id="proposal-side"):
                yield Static("BRIEF", classes="proposal-heading")
                yield Static(self._brief_digest(), id="proposal-brief", markup=False)
        yield Static(self._footer(), id="proposal-footer")
        yield Input(placeholder="revision note (enter = regenerate)", id="revision")
        with Horizontal(id="proposal-actions"):
            yield Button("Accept", id="accept", variant="success")
            yield Button("Regenerate", id="regenerate", variant="warning")
            yield Button("Cancel", id="cancel", variant="error")

    def _brief_digest(self) -> str:
        lines = [str(self.bundle.dir)]
        if self.bundle.brief_file:
            lines.append(f"brief: {self.bundle.brief_file}")
        lines.append(f"docs: {len(self.bundle.docs())}")
        images = self.bundle.images()
        lines.append(f"images: {len(images)}")
        for image in images[:6]:
            lines.append(f"  - {image.rel} {image.summary}")
        return "\n".join(lines)

    def _footer(self) -> str:
        if self.cost_usd:
            cost = f"proposal cost ${self.cost_usd:.4f}"
        else:
            cost = "proposal cost: free (mock provider)"
        return f"[dim]{cost} · enter on a button to choose · type a note to regenerate · esc cancels[/]"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "accept":
            self.exit(("accept", None))
        elif event.button.id == "regenerate":
            note = self.query_one("#revision", Input).value.strip()
            self.exit(("revise", note or "tighten the criteria and make them independently checkable"))
        else:
            self.exit(("cancel", None))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        note = event.value.strip()
        if note:
            self.exit(("revise", note))
        else:
            self.query_one("#accept", Button).focus()

    def action_cancel(self) -> None:
        self.exit(("cancel", None))
