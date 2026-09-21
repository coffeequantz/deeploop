"""Budget enforcement. Hard limits live here, outside the model's control."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contract import Budget


class BudgetExceeded(Exception):
    def __init__(self, kind: str, message: str, spent_usd: float, limit: Any) -> None:
        super().__init__(message)
        self.kind = kind
        self.spent_usd = spent_usd
        self.limit = limit


@dataclass
class Usage:
    role: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class BudgetTracker:
    budget: Budget
    started_at: float = field(default_factory=time.time)
    spent_usd: float = 0.0
    iterations: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    extensions: List[Dict[str, Any]] = field(default_factory=list)
    _warned: List[str] = field(default_factory=list, repr=False)

    def start_iteration(self) -> None:
        if self.iterations >= self.budget.max_iterations:
            raise BudgetExceeded(
                "iterations",
                f"iteration limit reached: {self.iterations} of {self.budget.max_iterations} used",
                self.spent_usd,
                self.budget.max_iterations,
            )
        self.iterations += 1
        self.check(kind="iteration")

    def check(self, kind: str = "general") -> None:
        elapsed_minutes = self.elapsed_seconds / 60.0
        if elapsed_minutes > self.budget.max_wall_clock_minutes:
            raise BudgetExceeded(
                "wall_clock",
                f"wall clock limit reached: {elapsed_minutes:.1f}min > "
                f"{self.budget.max_wall_clock_minutes:.1f}min",
                self.spent_usd,
                self.budget.max_wall_clock_minutes,
            )
        if self.spent_usd > self.budget.max_usd:
            raise BudgetExceeded(
                "usd",
                f"spend limit reached: ${self.spent_usd:.4f} > ${self.budget.max_usd:.2f}",
                self.spent_usd,
                self.budget.max_usd,
            )
        if self.iterations > self.budget.max_iterations:
            raise BudgetExceeded(
                "iterations",
                f"iteration limit reached: {self.iterations} > {self.budget.max_iterations}",
                self.spent_usd,
                self.budget.max_iterations,
            )

    def add_usage(self, usage: Usage) -> float:
        self.spent_usd += usage.cost_usd
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.cached_tokens += usage.cached_tokens
        return self.spent_usd

    def extend(
        self,
        usd: Optional[float] = None,
        iterations: Optional[int] = None,
        minutes: Optional[float] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {"reason": reason, "at": time.time()}
        if usd:
            self.budget.max_usd += float(usd)
            record["usd"] = float(usd)
        if iterations:
            self.budget.max_iterations += int(iterations)
            record["iterations"] = int(iterations)
        if minutes:
            self.budget.max_wall_clock_minutes += float(minutes)
            record["minutes"] = float(minutes)
        self.extensions.append(record)
        return record

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget.max_usd - self.spent_usd)

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.budget.max_iterations - self.iterations)

    def near_limit(self, threshold: float = 0.8) -> bool:
        return self.spent_usd >= self.budget.max_usd * threshold

    def snapshot(self) -> Dict[str, Any]:
        return {
            "spent_usd": round(self.spent_usd, 6),
            "max_usd": self.budget.max_usd,
            "iterations": self.iterations,
            "max_iterations": self.budget.max_iterations,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "max_wall_clock_minutes": self.budget.max_wall_clock_minutes,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "remaining_usd": round(self.remaining_usd, 6),
            "remaining_iterations": self.remaining_iterations,
        }
