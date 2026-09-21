from pathlib import Path

import pytest

from deeploop.ledger import Ledger, MissionState


def test_append_and_summarize(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".deeploop" / "ledger.jsonl")
    ledger.append("mission_started", goal="g")
    ledger.append("iteration_started", iteration=1)
    ledger.append("llm_call", role="actor", prompt_tokens=100, completion_tokens=20)
    ledger.append("cost", cost_usd=0.0123)
    ledger.append("llm_call", role="judge", prompt_tokens=50, completion_tokens=10)
    ledger.append("cost", cost_usd=0.004)
    ledger.append("verification", iteration=1, results=[{"id": "a", "passed": False}])
    ledger.append("checkpoint", sha="abc123", green=True)
    ledger.append("iteration_started", iteration=2)
    ledger.append("verification", iteration=2, results=[{"id": "a", "passed": True}])
    ledger.append("mission_finished", status="done", summary="ok")

    summary = ledger.summarize()
    assert summary.iterations == 2
    assert summary.calls == 2
    assert summary.prompt_tokens == 150
    assert summary.completion_tokens == 30
    assert summary.spent_usd == pytest.approx(0.0163)
    assert summary.status == "done"
    assert summary.criteria == {"a": True}
    assert summary.last_green_sha == "abc123"


def test_tolerates_partial_lines(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append("mission_started", goal="g")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "truncated"\n')
    assert len(ledger.entries()) == 1


def test_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = MissionState(
        iteration=3,
        spent_usd=0.42,
        status="running",
        last_green_sha="deadbeef",
        plan="1. do the thing",
        history=["iter 1: 0 file(s) changed | verify: FAIL a"],
        criteria={"a": False},
    )
    state.save(path)
    loaded = MissionState.load(path)
    assert loaded is not None
    assert loaded.iteration == 3
    assert loaded.plan == "1. do the thing"
    assert loaded.criteria == {"a": False}
    assert loaded.last_green_sha == "deadbeef"


def test_state_load_missing_or_corrupt(tmp_path: Path) -> None:
    assert MissionState.load(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert MissionState.load(bad) is None
