"""First-run setup screen: pick a provider, paste a key, verify it, save."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, RadioButton, RadioSet, Static

from ..catalog import provider_options
from ..config import load_config
from ..setup import apply_setup, test_connection, validate_inputs


class SetupApp(App[Optional[Dict[str, Any]]]):
    """Returns the saved settings, or None if the human cancelled."""

    TITLE = "deeploop · setup"
    BINDINGS = [
        Binding("escape", "cancel", "cancel"),
        Binding("ctrl+c", "cancel", "cancel", show=False),
    ]

    def __init__(self, default_provider: str = "deepseek", path: Optional[Path] = None) -> None:
        super().__init__()
        self.options = provider_options()
        self.default_provider = default_provider
        self.config_path = path

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold magenta]deeploop[/]  provider setup  "
            "[dim](stored in your user config with 0600 permissions)[/]",
            id="setup-title",
        )
        with VerticalScroll(id="setup-body"):
            yield Static("1 · Choose a provider", classes="setup-heading")
            yield RadioSet(
                *[
                    RadioButton(
                        option.label,
                        value=option.name == self.default_provider,
                        id=f"p-{option.name}",
                    )
                    for option in self.options
                ],
                id="provider-set",
            )
            yield Static("", id="provider-notes", markup=True)
            yield Static("2 · API key", classes="setup-heading")
            yield Input(
                password=True,
                placeholder="paste the key (not needed for ollama or mock)",
                id="key",
            )
            yield Static("3 · Base URL (optional)", classes="setup-heading")
            yield Input(placeholder="leave blank for the provider default", id="base-url")
            yield Static("", id="setup-status", markup=True)
        yield Static(
            "[dim]enter on Save stores the key; Test connection makes one tiny call "
            "(fractions of a cent)[/]",
            id="setup-footer",
        )
        with Horizontal(id="setup-actions"):
            yield Button("Test connection", id="test")
            yield Button("Save", id="save", variant="success")
            yield Button("Cancel", id="cancel", variant="error")

    def on_mount(self) -> None:
        self._refresh_notes()
        self.query_one("#key", Input).focus()

    def _selected_provider(self) -> str:
        index = self.query_one("#provider-set", RadioSet).pressed_index
        if index is None or index < 0 or index >= len(self.options):
            return self.default_provider
        return self.options[index].name

    def _refresh_notes(self) -> None:
        option = next(
            (item for item in self.options if item.name == self._selected_provider()), None
        )
        if option is None:
            return
        notes: List[str] = []
        if option.notes:
            notes.append(option.notes)
        if option.key_url:
            notes.append(f"key: {option.key_url}")
        if option.models:
            notes.append(f"models: {option.models.get('actor', '')}")
        self.query_one("#provider-notes", Static).update("[dim]" + " · ".join(notes) + "[/]")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._refresh_notes()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "test":
            self._run_test()
        elif event.button.id == "save":
            self._save()
        else:
            self.exit(None)

    def _inputs(self) -> Dict[str, str]:
        return {
            "provider": self._selected_provider(),
            "api_key": self.query_one("#key", Input).value.strip(),
            "base_url": self.query_one("#base-url", Input).value.strip(),
        }

    def _status(self, message: str, level: str = "info") -> None:
        color = {"info": "dim", "ok": "green", "error": "red"}.get(level, "dim")
        self.query_one("#setup-status", Static).update(f"[{color}]{message}[/]")

    def _save(self) -> None:
        values = self._inputs()
        errors = validate_inputs(values["provider"], values["api_key"], values["base_url"])
        if errors:
            self._status("; ".join(errors), "error")
            return
        saved = apply_setup(
            values["provider"], values["api_key"], values["base_url"], path=self.config_path
        )
        self.exit({**values, "saved_to": str(saved)})

    @work(exclusive=True)
    async def _run_test(self) -> None:
        values = self._inputs()
        errors = validate_inputs(values["provider"], values["api_key"], values["base_url"])
        if errors:
            self._status("; ".join(errors), "error")
            return
        self._status(f"testing {values['provider']} …")
        ok, message = await test_connection(
            values["provider"], values["api_key"], values["base_url"]
        )
        self._status(("✓ " if ok else "✗ ") + message, "ok" if ok else "error")

    def action_cancel(self) -> None:
        self.exit(None)


def run_setup(
    default_provider: Optional[str] = None, path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    provider = default_provider or load_config(path).provider
    return SetupApp(default_provider=provider, path=path).run()
