"""Tool layer: everything the actor can touch goes through here."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..events import E, EventBus
from ..permissions import PermissionDenied, PermissionGate
from ..util import truncate_middle


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, max_chars: int = 6000) -> str:
        if self.ok:
            body = self.output or "(no output)"
        else:
            body = self.error or "(failed)"
        return truncate_middle(body, max_chars)

    def as_message_content(self, max_chars: int = 6000) -> str:
        status = "ok" if self.ok else "error"
        return f"[{status}] {self.render(max_chars)}"


@dataclass
class ToolContext:
    workspace: Path
    gate: PermissionGate
    max_output_chars: int = 6000
    command_timeout_seconds: int = 300
    pycache_prefix: Optional[Path] = None
    checkpoints: Optional[Any] = None
    bus: Optional[EventBus] = None


class Tool(ABC):
    name: str = "tool"
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: List[Tool]) -> None:
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in tools}

    def names(self) -> List[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, name: str, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        started = time.monotonic()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool: {name}", duration_ms=0)
        try:
            ctx.gate.check_tool(name)
            result = await tool.run(args or {}, ctx)
        except PermissionDenied as exc:
            result = ToolResult(ok=False, error=f"permission denied: {exc}")
            if ctx.bus is not None:
                await ctx.bus.emit(E.LOG, level="warning", message=f"permission denied: {exc}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not raised
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        if not result.duration_ms:
            result.duration_ms = int((time.monotonic() - started) * 1000)
        return result
