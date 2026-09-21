"""Human escalation interface. Escalation is a policy, not a vibe: the contract
says what happens, this module asks."""

from __future__ import annotations

import asyncio
import sys
from typing import List


class HumanInterface:
    async def ask(self, reason: str, options: List[str]) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class AutoHaltHuman(HumanInterface):
    """Non-interactive default: choose the first option (usually halt)."""

    def __init__(self, default: str = "halt") -> None:
        self.default = default
        self.asked: List[str] = []

    async def ask(self, reason: str, options: List[str]) -> str:
        self.asked.append(reason)
        return self.default if self.default in options else options[0]


class ConsoleHuman(HumanInterface):
    """Interactive terminal escalation. Falls back to auto-halt when stdin is
    not a TTY (CI, piped output)."""

    def __init__(self, default: str = "halt") -> None:
        self.default = default
        self._fallback = AutoHaltHuman(default=default)

    async def ask(self, reason: str, options: List[str]) -> str:
        if not sys.stdin.isatty():
            return await self._fallback.ask(reason, options)
        prompt = f"\n[deeploop] escalation: {reason}\noptions ({'/'.join(options)}): "
        answer = (await asyncio.to_thread(input, prompt)).strip().lower()
        if answer in options:
            return answer
        for option in options:
            if option.startswith(answer) and answer:
                return option
        return self.default if self.default in options else options[0]
