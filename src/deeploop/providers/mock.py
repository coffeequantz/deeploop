"""Scripted provider for tests and keyless demos.

Turns are consumed in order for the `actor` role. Planner, critic and judge
roles get deterministic canned answers so an entire mission can run with no
network access at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ChatMessage, Completion, Provider, ToolCall, Usage

DEMO_FIX = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def multiply(a, b):\n"
    "    return a * b\n"
)


@dataclass
class MockTurn:
    """One actor response: either tool calls or a closing message."""

    tool_calls: List[ToolCall] = field(default_factory=list)
    content: str = ""


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"mock_{name}_{abs(hash((name, tuple(sorted(arguments.items(), key=str))))) % 10**8}",
                    name=name, arguments=arguments, raw_arguments=json.dumps(arguments))


def demo_turns() -> List[MockTurn]:
    """Two-iteration scripted mission for examples/demo-project."""
    return [
        MockTurn(tool_calls=[_call("run_command", command="pytest -q")], content=""),
        MockTurn(content="The suite fails: add() returns a - b. I will fix mathlib.py next."),
        MockTurn(tool_calls=[_call("write_file", path="mathlib.py", content=DEMO_FIX)], content=""),
        MockTurn(tool_calls=[_call("run_command", command="pytest -q")], content=""),
        MockTurn(content="mathlib.add now returns a + b and the suite passes."),
    ]


def demo_questions() -> Dict[str, Any]:
    """Scripted clarifying questions, used when the mock provider interviews."""
    return {
        "questions": [
            {
                "id": "verify-command",
                "question": "Which command should prove the work is done?",
                "choices": ["pytest -q"],
                "default": "pytest -q",
                "affects": "criteria",
                "why": "it becomes the machine-checkable success criterion",
            },
            {
                "id": "edit-in-place",
                "question": "Should the fix be applied in place, or in a new directory?",
                "choices": ["in place", "new directory"],
                "default": "in place",
                "affects": "workspace",
                "why": "it sets permissions.workspace",
            },
        ]
    }


def demo_proposal() -> Dict[str, Any]:
    """Scripted contract proposal, used when the mock provider is asked to derive
    a mission from a brief (keyless demos and tests)."""
    return {
        "goal": "Make the demo project's test suite pass without weakening the tests.",
        "workspace": ".",
        "criteria": [
            {
                "id": "tests-pass",
                "description": "The project test suite exits 0.",
                "check": "pytest -q",
            }
        ],
        "allow_commands": ["pytest"],
        "budget": {"max_usd": 0.5, "max_iterations": 6, "max_wall_clock_minutes": 10},
        "notes": "Mock proposal for the bundled demo project; no network calls are made.",
    }


class MockProvider(Provider):
    name = "mock"

    def __init__(
        self,
        turns: Optional[List[MockTurn]] = None,
        judge_pass: bool = True,
        critic_progress: bool = True,
        planner_text: str = "1. Run the tests.\n2. Fix the failing code.\n3. Re-run the tests.",
        interviewer_text: Optional[str] = None,
        judge_reason: str = "mock judge: evidence satisfies the criterion",
        critic_reason: str = "mock critic: iteration produced observable change",
    ) -> None:
        self.turns = list(turns) if turns is not None else demo_turns()
        self.judge_pass = judge_pass
        self.critic_progress = critic_progress
        self.planner_text = planner_text
        self.interviewer_text = (
            interviewer_text if interviewer_text is not None else json.dumps(demo_questions())
        )
        self.judge_reason = judge_reason
        self.critic_reason = critic_reason
        self.index = 0
        self.calls: List[Dict[str, Any]] = []

    async def complete(
        self,
        messages: List[ChatMessage],
        *,
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        role: str = "actor",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Completion:
        self.calls.append(
            {"role": role, "model": model, "message_count": len(messages), "messages": list(messages)}
        )
        if role == "interviewer":
            message = ChatMessage.assistant(self.interviewer_text)
        elif role == "planner":
            message = ChatMessage.assistant(self.planner_text)
        elif role == "critic":
            message = ChatMessage.assistant(
                json.dumps(
                    {
                        "progress": self.critic_progress,
                        "reason": self.critic_reason,
                        "hint": "" if self.critic_progress else "try a different approach",
                    }
                )
            )
        elif role == "judge":
            message = ChatMessage.assistant(
                json.dumps({"passed": self.judge_pass, "reason": self.judge_reason})
            )
        else:
            if self.index < len(self.turns):
                turn = self.turns[self.index]
                self.index += 1
            else:
                turn = MockTurn(content="[mock] script exhausted; stopping this iteration.")
            message = ChatMessage(role="assistant", content=turn.content, tool_calls=list(turn.tool_calls))
        usage = Usage(prompt_tokens=100, completion_tokens=50, cached_tokens=0, cost_usd=0.0)
        return Completion(message=message, usage=usage, model=model, latency_ms=1, role=role)

    async def aclose(self) -> None:
        return None
