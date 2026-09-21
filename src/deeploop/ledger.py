"""Append-only mission ledger and resumable state snapshot."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LedgerSummary:
    entries: int = 0
    iterations: int = 0
    spent_usd: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_green_sha: Optional[str] = None
    status: str = "unknown"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    criteria: Dict[str, bool] = field(default_factory=dict)


class Ledger:
    """JSONL event log. The source of truth for cost and mission history."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_kind: str, **data: Any) -> Dict[str, Any]:
        record = {"ts": time.time(), "kind": event_kind, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
        return record

    def entries(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def summarize(self) -> LedgerSummary:
        summary = LedgerSummary()
        for rec in self.entries():
            summary.entries += 1
            kind = rec.get("kind")
            if kind == "mission_started":
                summary.started_at = rec.get("ts")
            elif kind == "iteration_started":
                summary.iterations = max(summary.iterations, int(rec.get("iteration", 0)))
            elif kind == "llm_call":
                summary.calls += 1
                summary.prompt_tokens += int(rec.get("prompt_tokens", 0) or 0)
                summary.completion_tokens += int(rec.get("completion_tokens", 0) or 0)
            elif kind == "cost":
                summary.spent_usd += float(rec.get("cost_usd", 0.0) or 0.0)
            elif kind == "checkpoint":
                if rec.get("green"):
                    summary.last_green_sha = rec.get("sha")
            elif kind == "verification":
                for result in rec.get("results", []):
                    summary.criteria[result.get("id", "?")] = bool(result.get("passed"))
            elif kind == "mission_finished":
                summary.status = rec.get("status", "unknown")
                summary.finished_at = rec.get("ts")
        return summary


@dataclass
class MissionState:
    """Fast-resume snapshot written after every iteration."""

    iteration: int = 0
    spent_usd: float = 0.0
    status: str = "pending"
    last_green_sha: Optional[str] = None
    last_sha: Optional[str] = None
    plan: str = ""
    replans: int = 0
    no_change_streak: int = 0
    error_streak: int = 0
    no_progress_streak: int = 0
    last_error_signature: str = ""
    history: List[str] = field(default_factory=list)
    criteria: Dict[str, bool] = field(default_factory=dict)
    actor_model: Optional[str] = None
    started_at: float = field(default_factory=time.time)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Optional[MissionState]:
        if not Path(path).exists():
            return None
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
