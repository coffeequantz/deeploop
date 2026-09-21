import time

import pytest

from deeploop.budget import BudgetExceeded, BudgetTracker, Usage
from deeploop.contract import Budget


def test_iteration_cap_is_hard() -> None:
    tracker = BudgetTracker(Budget(max_iterations=2))
    tracker.start_iteration()
    tracker.start_iteration()
    with pytest.raises(BudgetExceeded) as exc:
        tracker.start_iteration()
    assert exc.value.kind == "iterations"
    assert tracker.iterations == 2


def test_usd_cap_raises_on_check() -> None:
    tracker = BudgetTracker(Budget(max_usd=0.10, max_iterations=10))
    tracker.add_usage(Usage(role="actor", model="m", cost_usd=0.05))
    tracker.check()
    tracker.add_usage(Usage(role="actor", model="m", cost_usd=0.06))
    with pytest.raises(BudgetExceeded) as exc:
        tracker.check()
    assert exc.value.kind == "usd"


def test_wall_clock_cap() -> None:
    tracker = BudgetTracker(Budget(max_wall_clock_minutes=0.01))
    tracker.started_at = time.time() - 60
    with pytest.raises(BudgetExceeded) as exc:
        tracker.check()
    assert exc.value.kind == "wall_clock"


def test_extend_raises_limits_and_records() -> None:
    tracker = BudgetTracker(Budget(max_usd=1.0, max_iterations=1))
    tracker.start_iteration()
    with pytest.raises(BudgetExceeded):
        tracker.start_iteration()
    record = tracker.extend(usd=0.5, iterations=2, minutes=5, reason="test")
    assert record["usd"] == 0.5
    tracker.start_iteration()
    tracker.start_iteration()
    assert tracker.iterations == 3
    assert tracker.budget.max_usd == 1.5


def test_snapshot_reports_remaining() -> None:
    tracker = BudgetTracker(Budget(max_usd=2.0, max_iterations=4))
    tracker.start_iteration()
    tracker.add_usage(Usage(role="actor", model="m", prompt_tokens=100, completion_tokens=50, cost_usd=0.25))
    snapshot = tracker.snapshot()
    assert snapshot["remaining_usd"] == pytest.approx(1.75)
    assert snapshot["remaining_iterations"] == 3
    assert snapshot["calls"] == 1
    assert snapshot["prompt_tokens"] == 100


def test_near_limit_warning_threshold() -> None:
    tracker = BudgetTracker(Budget(max_usd=1.0))
    tracker.add_usage(Usage(role="actor", model="m", cost_usd=0.79))
    assert tracker.near_limit() is False
    tracker.add_usage(Usage(role="actor", model="m", cost_usd=0.02))
    assert tracker.near_limit() is True
