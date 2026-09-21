"""Event bus used to decouple the mission controller from its front ends."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Union


class E:
    """Event kind constants."""

    MISSION_STARTED = "mission_started"
    MISSION_FINISHED = "mission_finished"
    ITERATION_STARTED = "iteration_started"
    PLAN = "plan"
    ASSISTANT_TEXT = "assistant_text"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    VERIFICATION = "verification"
    CRITIC = "critic"
    BUDGET = "budget"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    BRIEF = "brief_loaded"
    STUCK = "stuck"
    ESCALATION = "escalation"
    ESCALATION_RESOLVED = "escalation_resolved"
    INTERJECTION = "interjection"
    STATE = "state"
    LOG = "log"


@dataclass
class Event:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Subscriber = Callable[[Event], Union[None, Awaitable[None]]]


class EventBus:
    """Minimal async pub/sub. Subscribers may be sync or async callables."""

    def __init__(self) -> None:
        self._subs: List[Subscriber] = []

    def subscribe(self, fn: Subscriber) -> None:
        self._subs.append(fn)

    def unsubscribe(self, fn: Subscriber) -> None:
        if fn in self._subs:
            self._subs.remove(fn)

    async def emit(self, event_kind: str, **data: Any) -> Event:
        event = Event(kind=event_kind, data=data)
        for fn in list(self._subs):
            try:
                result = fn(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # a broken subscriber must not kill the mission
                print(f"[deeploop] event subscriber error: {exc!r}")
        return event
