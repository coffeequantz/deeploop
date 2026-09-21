"""Command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .brief import load_brief
from .budget import BudgetTracker
from .contract import TaskContract
from .controller import MissionStatus
from .events import EventBus
from .ledger import Ledger
from .llm import ModelRunner
from .proposal import contract_from_yaml, contract_to_yaml, propose_contract
from .providers import build_provider
from .providers.mock import MockProvider, demo_proposal
from .providers.pricing import as_table_rows
from .runtime import build_runtime, resolve_brief_target, resolve_contract_path
from .tui import TUIHuman, run_tui

CONSOLE = Console()

TEMPLATE = """# DeepLoop mission contract
goal: "Describe the finished outcome in one sentence."

success_criteria:
  - id: tests-pass
    description: "The project test suite passes."
    check: "pytest -q"

budget:
  max_usd: 2.00
  max_iterations: 30
  max_wall_clock_minutes: 30

permissions:
  workspace: "."
  network: false
  allow_all_commands: false
  # allow_commands defaults to a conservative developer allowlist; extend it here.

escalation:
  on_budget_exhausted: halt     # halt | ask | degrade
  on_stuck: ask                 # halt | ask | replan
  on_permission_denied: deny    # halt | ask | deny

provider:
  name: deepseek                # deepseek | openrouter | ollama | mock
  # api_key_env: DEEPSEEK_API_KEY

model:
  planner: deepseek-reasoner
  actor: deepseek-chat
  critic: deepseek-chat
  judge: deepseek-reasoner
  # vision: qwen/qwen-2.5-vl-72b-instruct   # optional: describe brief images
  # fallback_actor: deepseek-chat           # used when on_budget_exhausted=degrade

checkpoint:
  enabled: true
  branch: deeploop/mission

# Optional brief folder: BRIEF.md plus context docs and images.
# `deeploop brief <dir>` derives this contract from the folder automatically.
# brief:
#   dir: "."
#   describe_images: true
"""


def _base_contract(provider_name: Optional[str] = None) -> TaskContract:
    data = yaml.safe_load(TEMPLATE)
    if provider_name:
        data["provider"]["name"] = provider_name
    return TaskContract.model_validate(data)


def _print_contract(contract: TaskContract, contract_path: Path) -> None:
    CONSOLE.print(f"[bold magenta]deeploop[/] contract [dim]{contract_path}[/]")
    CONSOLE.print(f"[bold]goal[/] {contract.goal}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("criterion")
    table.add_column("kind")
    table.add_column("check / description")
    for criterion in contract.success_criteria:
        table.add_row(criterion.id, criterion.kind or "?", criterion.check or criterion.description)
    CONSOLE.print(table)
    CONSOLE.print(
        f"[bold]budget[/] ${contract.budget.max_usd} · {contract.budget.max_iterations} iterations · "
        f"{contract.budget.max_wall_clock_minutes} minutes"
    )
    CONSOLE.print(
        f"[bold]permissions[/] workspace={contract.permissions.workspace} "
        f"network={contract.permissions.network} "
        f"allow_all_commands={contract.permissions.allow_all_commands}"
    )
    if contract.brief.dir:
        CONSOLE.print(f"[bold]brief[/] {contract.brief.dir}")
    CONSOLE.print(
        f"[bold]provider[/] {contract.provider.name} · actor={contract.model.actor} "
        f"planner={contract.model.planner} judge={contract.model.judge}"
    )
    warnings = contract.warnings()
    for warning in warnings:
        CONSOLE.print(f"[yellow]warning[/] {warning}")
    if not warnings:
        CONSOLE.print("[green]no warnings[/]")


def _report(directory: Path) -> int:
    ledger_path = directory / ".deeploop" / "ledger.jsonl"
    if not ledger_path.exists():
        CONSOLE.print(f"[red]no ledger found at {ledger_path}[/]")
        return 1
    ledger = Ledger(ledger_path)
    summary = ledger.summarize()
    status_color = "green" if summary.status == "done" else "yellow"
    CONSOLE.print(
        f"[bold]mission[/] [{status_color}]{summary.status}[/]  "
        f"iterations={summary.iterations}  spend=${summary.spent_usd:.4f}  "
        f"llm_calls={summary.calls}"
    )
    if summary.criteria:
        for criterion_id, passed in summary.criteria.items():
            glyph = "[green]✓[/]" if passed else "[red]✗[/]"
            CONSOLE.print(f"  {glyph} {criterion_id}")
    per_iteration = {}
    for entry in ledger.entries():
        if entry.get("kind") == "cost":
            iteration = entry.get("iteration")
            key = f"iteration {iteration}" if iteration is not None else "setup"
            per_iteration[key] = per_iteration.get(key, 0.0) + float(entry.get("cost_usd") or 0.0)
    if per_iteration:
        table = Table(title="spend by iteration", show_header=True, header_style="bold")
        table.add_column("bucket")
        table.add_column("usd", justify="right")
        for key, value in sorted(per_iteration.items(), key=_iteration_sort_key):
            table.add_row(str(key), f"${value:.4f}")
        CONSOLE.print(table)
    return 0


def _iteration_sort_key(item):
    key = str(item[0])
    if key.startswith("iteration "):
        try:
            return (0, int(key.split(" ", 1)[1]))
        except ValueError:
            return (0, 0)
    return (1, 0)


def _proposal_context(
    base: TaskContract, contract_dir: Path
) -> Tuple[ModelRunner, BudgetTracker, Ledger]:
    if base.provider.name == "mock":
        provider = MockProvider(planner_text=json.dumps(demo_proposal()))
    else:
        provider = build_provider(base.provider)
    budget = BudgetTracker(base.budget)
    ledger = Ledger(contract_dir / ".deeploop" / "ledger.jsonl")
    bus = EventBus()
    runner = ModelRunner(provider, base, budget, ledger, bus)
    return runner, budget, ledger


def _review_console(draft, budget: BudgetTracker) -> Tuple[str, Optional[str]]:
    CONSOLE.print()
    CONSOLE.print(Panel(draft.summary(), title="proposed mission contract", border_style="magenta"))
    for warning in draft.warnings:
        CONSOLE.print(f"[yellow]warning[/] {warning}")
    if not sys.stdin.isatty():
        CONSOLE.print(
            "[yellow]not an interactive terminal: pass --yes to accept the proposal as-is[/]"
        )
        return ("cancel", None)
    while True:
        try:
            answer = input("[a]ccept · [r]evise · [e]dit yaml · [c]ancel > ").strip().lower()
        except EOFError:
            return ("cancel", None)
        if answer in ("a", "accept"):
            return ("accept", None)
        if answer in ("c", "cancel", ""):
            return ("cancel", None)
        if answer in ("e", "edit"):
            return ("edit", None)
        if answer in ("r", "revise"):
            try:
                note = input("revision note: ").strip()
            except EOFError:
                note = ""
            return ("revise", note)


def _edit_in_editor(contract: TaskContract) -> Optional[TaskContract]:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(contract_to_yaml(contract))
        path = Path(handle.name)
    if editor:
        subprocess.call(f"{editor} {path}", shell=True)
    else:
        CONSOLE.print(f"[yellow]no $EDITOR set: edit {path} and press enter[/]")
        try:
            input()
        except EOFError:
            pass
    try:
        return contract_from_yaml(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - show the validation error and re-prompt
        CONSOLE.print(f"[red]edited contract is invalid: {type(exc).__name__}: {exc}[/]")
        return None
    finally:
        path.unlink(missing_ok=True)


def _derive_contract(
    contract_dir: Path,
    brief_dir: Path,
    *,
    provider_name: Optional[str],
    auto_yes: bool,
    use_tui: bool,
) -> Optional[Path]:
    base = _base_contract(provider_name)
    exclude = [brief_dir / ".deeploop"]
    bundle = load_brief(brief_dir, base.brief, exclude=exclude)
    if not bundle.brief_text and not bundle.files:
        CONSOLE.print(f"[red]brief folder has no readable files: {brief_dir}[/]")
        return None
    runner, budget, ledger = _proposal_context(base, contract_dir)
    brief_rel = os.path.relpath(brief_dir, contract_dir)
    revision = ""
    contract: Optional[TaskContract] = None
    try:
        while True:
            CONSOLE.print(f"[dim]deriving a contract from {brief_dir} …[/]")
            try:
                draft = asyncio.run(propose_contract(runner, bundle, base, revision))
            except Exception as exc:  # noqa: BLE001 - report and stop
                CONSOLE.print(f"[red]could not derive a contract: {type(exc).__name__}: {exc}[/]")
                return None
            contract = draft.to_contract(base, brief_rel)
            if auto_yes:
                break
            if use_tui:
                from .tui.proposal import ProposalApp

                result = ProposalApp(draft, bundle, cost_usd=budget.spent_usd).run()
                action, note = result or ("cancel", None)
            else:
                action, note = _review_console(draft, budget)
            if action == "cancel":
                CONSOLE.print("[yellow]proposal discarded; nothing was written[/]")
                return None
            if action == "revise":
                revision = note or "tighten the criteria and make them independently checkable"
                continue
            if action == "edit":
                edited = _edit_in_editor(contract)
                if edited is None:
                    continue
                contract = edited
                break
            break
    finally:
        asyncio.run(runner.provider.aclose())

    target = contract_dir / "mission.yaml"
    if target.exists():
        CONSOLE.print(f"[red]refusing to overwrite existing contract: {target}[/]")
        return None
    target.write_text(contract_to_yaml(contract), encoding="utf-8")
    CONSOLE.print(f"[green]wrote[/] {target} [dim](proposal cost ${budget.spent_usd:.4f})[/]")
    ledger.append(
        "contract_written",
        path=str(target),
        goal=contract.goal,
        provider=base.provider.name,
        cost_usd=round(budget.spent_usd, 6),
    )
    _print_contract(contract, target)
    return target


def _derive_only(args: argparse.Namespace) -> int:
    target = Path(args.path)
    resolved = resolve_contract_path(target)
    if resolved is not None:
        CONSOLE.print(f"[yellow]contract already exists: {resolved}[/]")
        return 1
    brief_target = resolve_brief_target(target)
    if brief_target is None:
        CONSOLE.print(f"[red]no BRIEF.md (or brief/) found under {target}[/]")
        return 1
    contract_dir, brief_dir = brief_target
    use_tui = not args.headless and sys.stdout.isatty() and sys.stdin.isatty()
    contract_path = _derive_contract(
        contract_dir,
        brief_dir,
        provider_name=args.provider,
        auto_yes=args.yes,
        use_tui=use_tui,
    )
    return 0 if contract_path else 1


def _run(args: argparse.Namespace) -> int:
    target = Path(args.contract)
    contract_path = resolve_contract_path(target)
    use_tui = not args.headless and sys.stdout.isatty() and sys.stdin.isatty()
    if contract_path is None:
        brief_target = resolve_brief_target(target)
        if brief_target is None:
            CONSOLE.print(
                f"[red]no contract or brief found at {target} "
                f"(expected mission.yaml or BRIEF.md)[/]"
            )
            return 1
        contract_dir, brief_dir = brief_target
        contract_path = _derive_contract(
            contract_dir,
            brief_dir,
            provider_name=args.provider,
            auto_yes=args.yes,
            use_tui=use_tui,
        )
        if contract_path is None:
            return 1
    human = TUIHuman() if use_tui else None
    try:
        runtime = build_runtime(
            contract_path,
            resume=args.command == "resume",
            provider_name=args.provider,
            human=human,
        )
    except Exception as exc:  # noqa: BLE001 - config errors should be readable
        CONSOLE.print(f"[red]failed to load mission: {type(exc).__name__}: {exc}[/]")
        return 1

    if runtime.contract.provider.name == "mock":
        CONSOLE.print("[yellow]mock provider: scripted turns, no network calls[/]")

    if use_tui:
        run_tui(runtime)
        return 0

    from .render.console import ConsoleRenderer

    renderer = ConsoleRenderer(verbose=args.verbose)
    runtime.bus.subscribe(renderer)
    result = _run_sync(runtime)
    if result.status == MissionStatus.DONE:
        return 0
    if result.status == MissionStatus.ERROR:
        return 1
    return 2


def _run_sync(runtime):
    return asyncio.run(runtime.controller.run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deeploop", description="Budget-first autonomous agent harness")
    parser.add_argument("--version", action="version", version=f"deeploop {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write a starter mission.yaml")
    init.add_argument("directory", nargs="?", default=".", help="directory to write mission.yaml into")

    validate = sub.add_parser("validate", help="validate a contract and print warnings")
    validate.add_argument("contract", help="path to mission.yaml")

    brief = sub.add_parser("brief", help="derive a mission contract from a brief folder")
    brief.add_argument("path", nargs="?", default=".", help="folder with BRIEF.md (or a brief/ subfolder)")
    brief.add_argument("--provider", default=None, help="override provider (deepseek|openrouter|ollama|mock)")
    brief.add_argument("--yes", action="store_true", help="accept the derived contract without review")
    brief.add_argument("--headless", action="store_true", help="review in the console instead of the TUI")

    for name, help_text in (("run", "run a mission"), ("resume", "resume a mission from its ledger")):
        run = sub.add_parser(name, help=help_text)
        run.add_argument(
            "contract",
            help="path to mission.yaml, or a folder containing a brief",
        )
        run.add_argument("--headless", action="store_true", help="force plain console output")
        run.add_argument(
            "--provider", default=None, help="override provider (deepseek|openrouter|ollama|mock)"
        )
        run.add_argument("--verbose", action="store_true", help="print full tool output")
        run.add_argument(
            "--yes", action="store_true", help="accept a derived contract without review"
        )

    report = sub.add_parser("report", help="summarize a mission ledger")
    report.add_argument("directory", nargs="?", default=".", help="mission directory containing .deeploop/")

    sub.add_parser("models", help="show the pricing table used for budget accounting")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        target = Path(args.directory)
        target.mkdir(parents=True, exist_ok=True)
        mission = target / "mission.yaml"
        if mission.exists():
            CONSOLE.print(f"[yellow]mission.yaml already exists in {target}[/]")
            return 1
        mission.write_text(TEMPLATE, encoding="utf-8")
        CONSOLE.print(f"[green]wrote[/] {mission}")
        return 0
    if args.command == "validate":
        try:
            contract = TaskContract.load(Path(args.contract))
        except Exception as exc:  # noqa: BLE001
            CONSOLE.print(f"[red]invalid contract: {type(exc).__name__}: {exc}[/]")
            return 1
        _print_contract(contract, Path(args.contract))
        return 0
    if args.command == "brief":
        return _derive_only(args)
    if args.command in ("run", "resume"):
        return _run(args)
    if args.command == "report":
        return _report(Path(args.directory))
    if args.command == "models":
        table = Table(title="deeploop pricing (USD per 1M tokens)", show_header=True, header_style="bold")
        table.add_column("model")
        table.add_column("input", justify="right")
        table.add_column("cached input", justify="right")
        table.add_column("output", justify="right")
        for row in as_table_rows()["models"]:
            table.add_row(
                row["model"],
                f"${row['input_per_m']:.2f}",
                f"${row['cached_input_per_m']:.2f}",
                f"${row['output_per_m']:.2f}",
            )
        CONSOLE.print(table)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
