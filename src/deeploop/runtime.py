"""Runtime assembly: wire a contract into a runnable mission."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .budget import BudgetTracker
from .contract import MissionPaths, TaskContract
from .controller import Controller
from .events import EventBus
from .human import HumanInterface
from .ledger import Ledger, MissionState
from .llm import ModelRunner
from .permissions import PermissionGate
from .providers import MockTurn, Provider, build_provider
from .tools import CheckpointManager, ToolRegistry, default_registry
from .verifier import Critic, Verifier

CONTRACT_FILENAMES = ("mission.yaml", "mission.yml", "brief.yaml", "contract.yaml", "deeploop.yaml")
BRIEF_FILENAMES = ("BRIEF.md", "brief.md", "Brief.md", "goal.md", "README.md")


def resolve_contract_path(path: Path) -> Optional[Path]:
    """Accept either a contract file or a directory containing one."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for name in CONTRACT_FILENAMES:
            candidate = path / name
            if candidate.exists():
                return candidate
    return None


def resolve_brief_target(path: Path) -> Optional[Tuple[Path, Path]]:
    """Find a brief folder, returning (contract_dir, brief_dir).

    Accepts either a folder containing BRIEF.md directly, or a project folder
    with a `brief/` subfolder.
    """
    path = Path(path).resolve()
    if not path.is_dir():
        return None
    nested = path / "brief"
    if nested.is_dir() and any((nested / name).exists() for name in BRIEF_FILENAMES):
        return path, nested
    if any((path / name).exists() for name in BRIEF_FILENAMES):
        return path, path
    return None


@dataclass
class Runtime:
    contract: TaskContract
    paths: MissionPaths
    bus: EventBus
    ledger: Ledger
    budget: BudgetTracker
    gate: PermissionGate
    registry: ToolRegistry
    provider: Provider
    runner: ModelRunner
    verifier: Verifier
    critic: Critic
    checkpoints: CheckpointManager
    human: HumanInterface
    controller: Controller
    state: MissionState


def build_runtime(
    contract_path: Path,
    *,
    bus: Optional[EventBus] = None,
    resume: bool = False,
    provider_name: Optional[str] = None,
    human: Optional[HumanInterface] = None,
    mock_turns=None,
) -> Runtime:
    contract = TaskContract.load(Path(contract_path))
    if provider_name:
        contract.provider.name = provider_name  # type: ignore[assignment]
    paths = MissionPaths.build(Path(contract_path), contract).ensure()
    if paths.brief_dir and paths.brief_dir.exists():
        brief_entry = str(paths.brief_dir)
        if brief_entry not in contract.permissions.read_paths:
            contract.permissions.read_paths.append(brief_entry)
    bus = bus or EventBus()
    ledger = Ledger(paths.ledger_path)
    budget = BudgetTracker(contract.budget)

    state: Optional[MissionState] = None
    if resume:
        summary = ledger.summarize()
        budget.spent_usd = summary.spent_usd
        budget.iterations = summary.iterations
        budget.calls = summary.calls
        budget.prompt_tokens = summary.prompt_tokens
        budget.completion_tokens = summary.completion_tokens
        budget.started_at = time.time()
        state = MissionState.load(paths.state_path)
        if state is None:
            state = MissionState(
                iteration=summary.iterations,
                spent_usd=summary.spent_usd,
                status="resumed",
                last_green_sha=summary.last_green_sha,
            )
    if state is None:
        state = MissionState()

    gate = PermissionGate(contract.permissions, paths.workspace)
    registry = default_registry()
    provider = build_provider(contract.provider, mock_turns=mock_turns)
    runner = ModelRunner(provider, contract, budget, ledger, bus)
    checkpoints = CheckpointManager(paths.workspace, contract.checkpoint, bus)
    verifier = Verifier(
        contract,
        runner,
        bus,
        paths.workspace,
        contract.limits,
        pycache_prefix=paths.pycache_dir,
    )
    critic = Critic(runner, bus)
    from .human import ConsoleHuman

    controller = Controller(
        contract=contract,
        paths=paths,
        provider=provider,
        bus=bus,
        ledger=ledger,
        budget=budget,
        gate=gate,
        registry=registry,
        runner=runner,
        verifier=verifier,
        critic=critic,
        checkpoints=checkpoints,
        human=human or ConsoleHuman(),
        state=state,
    )
    return Runtime(
        contract=contract,
        paths=paths,
        bus=bus,
        ledger=ledger,
        budget=budget,
        gate=gate,
        registry=registry,
        provider=provider,
        runner=runner,
        verifier=verifier,
        critic=critic,
        checkpoints=checkpoints,
        human=controller.human,
        controller=controller,
        state=state,
    )


__all__ = [
    "Runtime",
    "build_runtime",
    "MockTurn",
    "resolve_contract_path",
    "resolve_brief_target",
]
