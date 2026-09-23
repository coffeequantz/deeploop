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
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from . import config as global_config
from . import setup as provider_setup
from .brief import load_brief
from .budget import BudgetTracker
from .catalog import get_option, provider_names
from .contract import TaskContract
from .controller import MissionStatus
from .events import EventBus
from .interview import (
    Answer,
    collect_answers,
    propose_questions,
    render_clarifications,
    resolve_answer,
    save_clarifications,
    skipped_summary,
)
from .ledger import Ledger
from .llm import ModelRunner
from .proposal import contract_from_yaml, contract_to_yaml, propose_contract
from .providers import build_provider
from .providers.mock import MockProvider, demo_proposal
from .providers.pricing import as_table_rows
from .runtime import build_runtime, resolve_brief_target, resolve_contract_path
from .tui import TUIHuman, run_tui

CONSOLE = Console()

STARTER_BRIEF = """# Brief: <name this project>

Describe the finished outcome in plain language. deeploop turns this into a
contract with machine-checkable criteria and loops until they pass.

## Goal

One or two sentences describing what should exist when this is done.

## What done looks like

- A command that proves it, e.g. `pytest -q` exits 0
- Any other observable, checkable outcome

## Constraints

- What must not change
- Whether new dependencies are allowed

## Context

- Files, docs, or links the agent should read (drop them in context/ or assets/)

## Non-goals

- What this work explicitly is not
"""

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
    contract = TaskContract.model_validate(data)
    provider_setup.apply_to_contract(contract, provider_override=provider_name)
    return contract


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


def _collect_answers(
    questions, use_tui: bool, budget: BudgetTracker
) -> Optional[List[Answer]]:
    """Returns answers (possibly all skipped), or None if the human cancelled."""
    if use_tui:
        from .tui.interview import InterviewApp

        result = InterviewApp(questions, cost_usd=budget.spent_usd).run()
        if result is None:
            return None
        return [
            Answer(question=question, answer=resolve_answer(value, question))
            for question, value in zip(questions, result)
        ]
    if not sys.stdin.isatty():
        CONSOLE.print(
            "[yellow]not an interactive terminal: clarifying questions skipped "
            "(run without --headless to answer them)[/]"
        )
        return []
    CONSOLE.print()
    CONSOLE.print(
        Panel(
            "\n".join(f"[{q.affects}] {q.question}" for q in questions),
            title="clarifying questions (blank = skip, default assumed)",
            border_style="cyan",
        )
    )

    def ask(label: str) -> str:
        try:
            return input(f"{label}\n> ")
        except EOFError:
            return ""

    try:
        return collect_answers(questions, ask)
    except KeyboardInterrupt:
        return None


def _derive_contract(
    contract_dir: Path,
    brief_dir: Path,
    *,
    provider_name: Optional[str],
    auto_yes: bool,
    use_tui: bool,
    ask_questions: bool = True,
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
    answers: List[Answer] = []
    clarifications = ""
    try:
        if ask_questions and not auto_yes:
            try:
                questions = asyncio.run(propose_questions(runner, bundle, base))
            except Exception as exc:  # noqa: BLE001 - the interview is optional
                CONSOLE.print(f"[yellow]interview skipped: {type(exc).__name__}: {exc}[/]")
                questions = []
            if questions:
                CONSOLE.print(f"[dim]{len(questions)} clarifying question(s)[/]")
                answers = _collect_answers(questions, use_tui, budget)
                if answers is None:
                    CONSOLE.print("[yellow]interview cancelled; nothing was written[/]")
                    return None
                clarifications = render_clarifications(answers)
                save_clarifications(
                    contract_dir / ".deeploop" / "artifacts" / "clarifications.json",
                    answers,
                    base.model.planner,
                )
                ledger.append(
                    "clarifications",
                    answered=sum(1 for item in answers if not item.skipped),
                    skipped=sum(1 for item in answers if item.skipped),
                    questions=[item.question.id for item in answers],
                )
        while True:
            CONSOLE.print(f"[dim]deriving a contract from {brief_dir} …[/]")
            try:
                draft = asyncio.run(
                    propose_contract(runner, bundle, base, revision, clarifications)
                )
            except Exception as exc:  # noqa: BLE001 - report and stop
                CONSOLE.print(f"[red]could not derive a contract: {type(exc).__name__}: {exc}[/]")
                return None
            unanswered = skipped_summary(answers)
            if unanswered and unanswered not in draft.warnings:
                draft.warnings.append(unanswered)
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
        ask_questions=not args.no_questions,
    )
    return 0 if contract_path else 1


def _prompt(text: str) -> Optional[str]:
    try:
        return input(f"{text}\n> ")
    except (EOFError, KeyboardInterrupt):
        return None


def _confirm(text: str) -> bool:
    answer = _prompt(f"{text} [Y/n]")
    if answer is None:
        return False
    return answer.strip().lower() in ("", "y", "yes")


def _interactive_setup(use_tui: bool) -> Optional[Dict[str, Any]]:
    if use_tui:
        from .tui.setup import run_setup

        return run_setup()
    if not sys.stdin.isatty():
        return None
    import getpass

    def secret(text: str) -> Optional[str]:
        try:
            return getpass.getpass(f"{text}: ")
        except (EOFError, KeyboardInterrupt):
            return None

    result = provider_setup.console_setup(
        _prompt, secret, default_provider=global_config.load_config().provider
    )
    if result is None:
        return None
    if result.get("error"):
        CONSOLE.print(f"[red]{result['error']}[/]")
        return None
    if "test_message" in result:
        level = "green" if result.get("test_ok") else "red"
        CONSOLE.print(f"[{level}]{result['test_message']}[/]")
    if result.get("saved_to"):
        CONSOLE.print(f"[green]saved[/] {result['saved_to']}")
    return result


def _ensure_provider(use_tui: bool, provider: Optional[str] = None) -> bool:
    target = provider or global_config.load_config().provider
    if global_config.is_configured(target):
        return True
    option = get_option(target)
    CONSOLE.print(f"[yellow]no API key configured for {target}[/]")
    if option is not None and option.key_url:
        CONSOLE.print(f"[dim]get one at {option.key_url}[/]")
    _interactive_setup(use_tui)
    return global_config.is_configured(global_config.load_config().provider)


def _setup_command(args: argparse.Namespace) -> int:
    if args.show:
        CONSOLE.print(global_config.describe())
        return 0
    if args.provider:
        key = args.key or global_config.resolve_api_key(args.provider) or ""
        base_url = args.base_url or ""
        errors = provider_setup.validate_inputs(args.provider, key, base_url)
        if errors:
            CONSOLE.print(f"[red]{'; '.join(errors)}[/]")
            return 1
        if not args.no_test:
            ok, message = asyncio.run(
                provider_setup.test_connection(args.provider, key, base_url)
            )
            CONSOLE.print(("[green]✓ [/]" if ok else "[red]✗ [/]") + message)
            if not ok:
                return 1
        saved = provider_setup.apply_setup(args.provider, key, base_url)
        CONSOLE.print(f"[green]saved[/] {saved}")
        return 0
    use_tui = not args.headless and sys.stdout.isatty() and sys.stdin.isatty()
    result = _interactive_setup(use_tui)
    if not result:
        CONSOLE.print("[yellow]setup cancelled; nothing was saved[/]")
        return 1
    return 0


def _brief_candidates(root: Path) -> List[Path]:
    candidates: List[Path] = []
    if resolve_contract_path(root) or resolve_brief_target(root):
        candidates.append(root.resolve())
    try:
        children = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError:
        return candidates
    for child in children:
        if child.name.startswith(".") or child.name in ("node_modules", ".venv"):
            continue
        if resolve_contract_path(child) or resolve_brief_target(child):
            candidates.append(child.resolve())
        if len(candidates) >= 8:
            break
    return candidates


def _write_starter_brief(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    brief = folder / "BRIEF.md"
    if not brief.exists():
        brief.write_text(STARTER_BRIEF, encoding="utf-8")
    return brief


def _choose_brief_folder() -> Optional[Path]:
    cwd = Path.cwd()
    candidates = _brief_candidates(cwd)
    if candidates:
        CONSOLE.print("found:")
        for index, path in enumerate(candidates, start=1):
            CONSOLE.print(f"  {index}. {path}")
    default = str(candidates[0]) if candidates else str(cwd / "brief")
    for _ in range(3):
        raw = _prompt(
            "Which folder should I work from? (one with BRIEF.md or mission.yaml)"
            f"\n[{default}]"
        )
        if raw is None:
            return None
        answer = raw.strip() or default
        if answer.isdigit() and candidates and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1]
        path = Path(answer).expanduser()
        if path.is_dir():
            if resolve_contract_path(path) or resolve_brief_target(path):
                return path
            CONSOLE.print(f"[yellow]{path} has no mission.yaml or BRIEF.md[/]")
            continue
        if _confirm(f"{path} does not exist. Create a starter brief there?"):
            brief = _write_starter_brief(path)
            CONSOLE.print(f"[green]wrote[/] {brief}")
            return path
    return None


def _interactive_start() -> int:
    """Bare `deeploop`: configure a provider if needed, pick a folder, run it."""
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        build_parser().print_help()
        return 0
    CONSOLE.print("[bold magenta]deeploop[/]  first run")
    if not _ensure_provider(use_tui=True):
        CONSOLE.print("[red]no provider configured; run `deeploop setup`[/]")
        return 1
    folder = _choose_brief_folder()
    if folder is None:
        CONSOLE.print(
            "[yellow]nothing to run.[/] Write a BRIEF.md, then: deeploop run <folder>"
        )
        return 0
    return _run(
        argparse.Namespace(
            command="run",
            contract=str(folder),
            headless=False,
            provider=None,
            verbose=False,
            yes=False,
            no_questions=False,
        )
    )


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
            ask_questions=not args.no_questions,
        )
        if contract_path is None:
            return 1
    try:
        contract = TaskContract.load(contract_path)
    except Exception as exc:  # noqa: BLE001 - config errors should be readable
        CONSOLE.print(f"[red]invalid contract: {type(exc).__name__}: {exc}[/]")
        return 1
    provider_name, provider_note = provider_setup.resolve_provider(contract, args.provider)
    if provider_note:
        CONSOLE.print(f"[yellow]{provider_note}[/]")
    if not global_config.is_configured(provider_name):
        if sys.stdin.isatty() and sys.stdout.isatty():
            _ensure_provider(use_tui, provider_name)
            provider_name, provider_note = provider_setup.resolve_provider(contract, args.provider)
    if not global_config.is_configured(provider_name):
        CONSOLE.print(
            f"[red]no API key for provider {provider_name!r}: run `deeploop setup`[/]"
        )
        return 1

    human = TUIHuman() if use_tui else None
    try:
        runtime = build_runtime(
            contract_path,
            resume=args.command == "resume",
            provider_name=provider_name,
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
    brief.add_argument("--no-questions", action="store_true", help="skip the clarifying interview")

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
        run.add_argument(
            "--no-questions", action="store_true", help="skip the clarifying interview"
        )

    setup = sub.add_parser("setup", help="configure a provider and store its API key")
    setup.add_argument("--provider", choices=provider_names(), default=None)
    setup.add_argument("--key", default=None, help="API key (otherwise prompted)")
    setup.add_argument("--base-url", default=None, help="override the provider base URL")
    setup.add_argument("--show", action="store_true", help="print the current configuration")
    setup.add_argument("--no-test", action="store_true", help="skip the connection test")
    setup.add_argument("--headless", action="store_true", help="prompt in the console, not the TUI")

    report = sub.add_parser("report", help="summarize a mission ledger")
    report.add_argument("directory", nargs="?", default=".", help="mission directory containing .deeploop/")

    sub.add_parser("models", help="show the pricing table used for budget accounting")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        return _interactive_start()
    parser = build_parser()
    args = parser.parse_args(raw)
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
    if args.command == "setup":
        return _setup_command(args)
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
