"""Git checkpointing (harness-owned) and read-only git tools for the actor.

The actor may inspect history but never rewrites it: commits and rollbacks are
performed only by the harness after each iteration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..contract import CheckpointConfig
from ..events import E, EventBus
from ..util import truncate_middle
from .base import Tool, ToolContext, ToolResult

GIT_IDENTITY = ["-c", "user.name=deeploop", "-c", "user.email=deeploop@localhost"]


class CheckpointManager:
    def __init__(
        self,
        workspace: Path,
        config: Optional[CheckpointConfig] = None,
        bus: Optional[EventBus] = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.config = config or CheckpointConfig()
        self.bus = bus
        self.enabled = self.config.enabled
        self.ready = False

    async def _git(self, *args: str, check: bool = False) -> Tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(self.workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        text = (stdout or b"").decode("utf-8", errors="replace")
        code = proc.returncode or 0
        if check and code != 0:
            raise RuntimeError(f"git {' '.join(args)} failed ({code}): {text.strip()}")
        return code, text

    async def is_repo(self) -> bool:
        code, out = await self._git("rev-parse", "--is-inside-work-tree")
        return code == 0 and out.strip() == "true"

    async def ensure_repo(self) -> bool:
        if not self.enabled:
            self.ready = False
            return False
        if not await self.is_repo():
            if not self.config.init_repo:
                self.ready = False
                return False
            await self._git("init", "-b", "main", check=True)
            exclude = self.workspace / ".git" / "info" / "exclude"
            try:
                exclude.parent.mkdir(parents=True, exist_ok=True)
                existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
                additions = [
                    line
                    for line in (
                        ".deeploop/",
                        "__pycache__/",
                        ".pytest_cache/",
                        ".ruff_cache/",
                        ".mypy_cache/",
                    )
                    if line not in existing
                ]
                if additions:
                    exclude.write_text(existing + "\n" + "\n".join(additions) + "\n", encoding="utf-8")
            except OSError:
                pass
        if self.config.branch:
            code, _ = await self._git("rev-parse", "--verify", self.config.branch)
            if code == 0:
                await self._git("checkout", "--quiet", self.config.branch, check=True)
            else:
                await self._git("checkout", "--quiet", "-b", self.config.branch, check=True)
        self.ready = True
        return True

    async def head_sha(self) -> Optional[str]:
        code, out = await self._git("rev-parse", "HEAD")
        if code != 0:
            return None
        return out.strip() or None

    async def changed_files(self) -> List[str]:
        code, out = await self._git("status", "--porcelain")
        if code != 0:
            return []
        files: List[str] = []
        for line in out.splitlines():
            entry = line[3:].strip()
            if " -> " in entry:
                entry = entry.split(" -> ", 1)[1]
            if entry:
                files.append(entry)
        return files

    async def commit(self, message: str) -> Optional[str]:
        if not self.enabled:
            return None
        if not await self.is_repo():
            return None
        await self._git("add", "-A")
        code, out = await self._git(*GIT_IDENTITY, "commit", "--no-verify", "-m", message)
        if code != 0 and "nothing to commit" not in out and "no changes added" not in out:
            if self.bus is not None:
                await self.bus.emit(
                    E.LOG, level="warning", message=f"checkpoint commit failed: {out.strip()}"
                )
            return None
        sha = await self.head_sha()
        if self.bus is not None and sha:
            await self.bus.emit(E.CHECKPOINT, sha=sha, message=message)
        return sha

    async def diff_since(self, base: Optional[str], max_chars: int = 8000) -> str:
        if base:
            code, out = await self._git("diff", "--stat", base)
            code2, detail = await self._git("diff", base)
            text = f"{out.strip()}\n\n{detail.strip()}" if out.strip() or detail.strip() else ""
        else:
            _, out = await self._git("status", "--porcelain")
            _, detail = await self._git("diff", "--cached")
            text = f"{out.strip()}\n{detail.strip()}".strip()
        return truncate_middle(text, max_chars) if text else ""

    async def show_commit(self, sha: str) -> str:
        _, out = await self._git("show", "--stat", "--oneline", sha)
        return out.strip()

    async def rollback(self, sha: str, clean: bool = True) -> bool:
        if not self.enabled or not sha:
            return False
        code, out = await self._git("reset", "--hard", sha)
        if code != 0:
            if self.bus is not None:
                await self.bus.emit(E.LOG, level="warning", message=f"rollback failed: {out.strip()}")
            return False
        if clean:
            await self._git("clean", "-fd")
        if self.bus is not None:
            await self.bus.emit(E.ROLLBACK, sha=sha)
        return True


class _GitReadTool(Tool):
    subcommand: List[str] = []

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        manager: Optional[CheckpointManager] = getattr(ctx, "checkpoints", None)
        if manager is None or not await manager.is_repo():
            return ToolResult(ok=False, error="workspace is not a git repository")
        extra = [str(a) for a in (args.get("args") or [])] if isinstance(args.get("args"), list) else []
        code, out = await manager._git(*self.subcommand, *extra)
        body = truncate_middle(out.strip() or "(no output)", ctx.max_output_chars)
        return ToolResult(ok=code == 0, output=body if code == 0 else "", error=body if code != 0 else "")


class GitStatusTool(_GitReadTool):
    name = "git_status"
    description = "Show git working tree status."
    parameters = {"type": "object", "properties": {}, "required": []}
    subcommand = ["status", "--short", "--branch"]


class GitDiffTool(_GitReadTool):
    name = "git_diff"
    description = "Show git diff (working tree, or a revision range passed via args)."
    parameters = {
        "type": "object",
        "properties": {
            "args": {"type": "array", "items": {"type": "string"}, "description": "Extra git diff arguments."}
        },
        "required": [],
    }
    subcommand = ["diff"]


class GitLogTool(_GitReadTool):
    name = "git_log"
    description = "Show recent commit history."
    parameters = {
        "type": "object",
        "properties": {
            "args": {"type": "array", "items": {"type": "string"}, "description": "Extra git log arguments."}
        },
        "required": [],
    }
    subcommand = ["log", "--oneline", "-n", "20"]
