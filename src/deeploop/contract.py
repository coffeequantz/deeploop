"""Task contract: the structured input a mission is defined by.

Everything the harness enforces (budget, permissions, escalation policy) lives
here, in data, not in prompts. The contract is validated before a mission runs
and re-read on resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_ALLOW_COMMANDS = [
    "cd",
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "rg",
    "find",
    "file",
    "stat",
    "diff",
    "echo",
    "printf",
    "sort",
    "uniq",
    "cut",
    "sed",
    "awk",
    "tr",
    "tee",
    "mkdir",
    "touch",
    "cp",
    "mv",
    "rm",
    "pytest",
    "python",
    "python3",
    "pip",
    "pip3",
    "node",
    "npm",
    "npx",
    "pnpm",
    "yarn",
    "tsc",
    "make",
    "cmake",
    "go",
    "cargo",
    "rustc",
    "gcc",
    "clang",
    "javac",
    "java",
    "ruff",
    "black",
    "mypy",
    "flake8",
    "eslint",
    "prettier",
    "jest",
    "vitest",
    "tox",
    "uv",
    "poetry",
    "git",
    "jq",
    "true",
    "false",
    "test",
]

DEFAULT_DENY_COMMANDS = [
    "sudo",
    "su ",
    "doas",
    "shutdown",
    "reboot",
    "halt",
    "mkfs",
    "diskutil eraseDisk",
    "dd if=",
    "chmod 777 /",
    "chown -R",
    ":(){",
    "rm -rf /",
    "rm -rf ~",
    "git push",
    "git reset --hard HEAD~",
    "git filter-branch",
    "npm publish",
    "pip install --user",
    "crontab",
    "launchctl",
    "systemctl",
    "killall",
]


class Criterion(BaseModel):
    """A single success criterion. Command criteria are machine-checkable."""

    id: str
    description: str = ""
    check: Optional[str] = None
    expect_exit: int = 0
    kind: Optional[Literal["command", "judge"]] = None
    timeout_seconds: int = 600

    @model_validator(mode="after")
    def _resolve_kind(self) -> Criterion:
        if self.kind is None:
            self.kind = "command" if self.check else "judge"
        if self.kind == "command" and not self.check:
            raise ValueError(f"criterion {self.id!r}: kind=command requires a `check` command")
        if self.kind == "judge" and self.check:
            raise ValueError(f"criterion {self.id!r}: kind=judge must not define `check`")
        if not self.description and self.check:
            self.description = self.check
        return self


class Budget(BaseModel):
    max_usd: float = Field(default=2.0, gt=0)
    max_iterations: int = Field(default=30, gt=0)
    max_wall_clock_minutes: float = Field(default=30.0, gt=0)


class Permissions(BaseModel):
    workspace: str = "."
    read_paths: List[str] = Field(default_factory=lambda: ["."])
    write_paths: List[str] = Field(default_factory=lambda: ["."])
    allow_commands: List[str] = Field(default_factory=lambda: list(DEFAULT_ALLOW_COMMANDS))
    deny_commands: List[str] = Field(default_factory=lambda: list(DEFAULT_DENY_COMMANDS))
    allow_all_commands: bool = False
    network: bool = False
    require_approval: List[str] = Field(default_factory=list)


class Limits(BaseModel):
    max_tool_calls_per_iteration: int = Field(default=8, gt=0)
    command_timeout_seconds: int = Field(default=300, gt=0)
    max_output_chars: int = Field(default=6000, gt=0)
    stuck_after: int = Field(default=3, gt=0)
    max_replans: int = Field(default=2, ge=0)
    context_keep_iterations: int = Field(default=4, ge=1)
    max_llm_retries: int = Field(default=2, ge=0)


class Escalation(BaseModel):
    on_budget_exhausted: Literal["halt", "ask", "degrade"] = "halt"
    on_stuck: Literal["halt", "ask", "replan"] = "ask"
    on_permission_denied: Literal["halt", "ask", "deny"] = "deny"


class ModelConfig(BaseModel):
    planner: str = "deepseek-reasoner"
    actor: str = "deepseek-chat"
    critic: str = "deepseek-chat"
    judge: str = "deepseek-reasoner"
    vision: Optional[str] = None
    fallback_actor: Optional[str] = None

    def for_role(self, role: str) -> str:
        return {
            "planner": self.planner,
            "actor": self.actor,
            "critic": self.critic,
            "judge": self.judge,
            "vision": self.vision or "",
        }.get(role, self.actor)


class ProviderConfig(BaseModel):
    name: Literal["deepseek", "openrouter", "ollama", "mock"] = "deepseek"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    timeout_seconds: float = 600.0
    pricing: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class CheckpointConfig(BaseModel):
    enabled: bool = True
    branch: Optional[str] = "deeploop/mission"
    init_repo: bool = True


class BriefConfig(BaseModel):
    """A folder of human-authored input: BRIEF.md plus context docs and images."""

    dir: Optional[str] = None
    include: List[str] = Field(
        default_factory=lambda: [
            "**/*.md",
            "**/*.txt",
            "**/*.rst",
            "**/*.csv",
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
        ]
    )
    images: List[str] = Field(
        default_factory=lambda: ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.webp"]
    )
    max_file_bytes: int = Field(default=200_000, gt=0)
    describe_images: bool = True
    max_context_chars: int = Field(default=6000, gt=0)


class TaskContract(BaseModel):
    goal: str
    success_criteria: List[Criterion]
    budget: Budget = Field(default_factory=Budget)
    permissions: Permissions = Field(default_factory=Permissions)
    limits: Limits = Field(default_factory=Limits)
    escalation: Escalation = Field(default_factory=Escalation)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    brief: BriefConfig = Field(default_factory=BriefConfig)
    mission_id: Optional[str] = None

    @model_validator(mode="after")
    def _require_criteria(self) -> TaskContract:
        if not self.success_criteria:
            raise ValueError("contract requires at least one success criterion")
        ids = [c.id for c in self.success_criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("success criterion ids must be unique")
        return self

    @property
    def command_criteria(self) -> List[Criterion]:
        return [c for c in self.success_criteria if c.kind == "command"]

    @property
    def judge_criteria(self) -> List[Criterion]:
        return [c for c in self.success_criteria if c.kind == "judge"]

    def warnings(self) -> List[str]:
        out: List[str] = []
        if not self.command_criteria:
            out.append(
                "no machine-checkable criteria: completion is decided by an LLM judge "
                "(false-done risk is higher)"
            )
        if self.permissions.allow_all_commands:
            out.append("allow_all_commands=true: command allowlist is disabled")
        if self.permissions.network:
            out.append("network=true: the agent may reach the network")
        if self.budget.max_usd > 25:
            out.append(f"large budget: ${self.budget.max_usd:.2f}")
        if self.budget.max_iterations > 100:
            out.append(f"large iteration budget: {self.budget.max_iterations}")
        if not self.checkpoint.enabled:
            out.append("checkpointing disabled: no rollback or crash-safe iteration history")
        return out

    @classmethod
    def load(cls, path: Path) -> TaskContract:
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith(".json"):
            data: Any = json.loads(text)
        else:
            data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"contract at {path} must be a mapping")
        return cls.model_validate(data)


@dataclass
class MissionPaths:
    """Filesystem layout for a mission: `.deeploop/` lives next to the contract."""

    root: Path
    workspace: Path
    contract_path: Optional[Path] = None
    brief_dir: Optional[Path] = None

    @classmethod
    def build(cls, contract_path: Path, contract: TaskContract) -> MissionPaths:
        contract_path = Path(contract_path).resolve()
        root = contract_path.parent
        ws = Path(contract.permissions.workspace)
        if not ws.is_absolute():
            ws = root / ws
        brief_dir: Optional[Path] = None
        if contract.brief.dir:
            candidate = Path(contract.brief.dir)
            if not candidate.is_absolute():
                candidate = root / candidate
            brief_dir = candidate.resolve()
        return cls(
            root=root,
            workspace=ws.resolve(),
            contract_path=contract_path,
            brief_dir=brief_dir,
        )

    @property
    def dir(self) -> Path:
        return self.root / ".deeploop"

    @property
    def ledger_path(self) -> Path:
        return self.dir / "ledger.jsonl"

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.dir / "artifacts"

    def ensure(self) -> MissionPaths:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "pycache").mkdir(parents=True, exist_ok=True)
        return self

    @property
    def pycache_dir(self) -> Path:
        return self.dir / "pycache"
