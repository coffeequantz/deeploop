"""Shell execution tool with allowlist enforcement and hard timeouts."""

from __future__ import annotations

import asyncio
import shlex
from typing import Any, Dict

from ..util import truncate_middle
from .base import Tool, ToolContext, ToolResult


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command inside the workspace and return its output. "
        "Only commands on the mission allowlist are permitted; destructive "
        "commands are blocked by the harness."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command line to execute."},
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional timeout override, capped by the mission limit.",
            },
        },
        "required": ["command"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args.get("command", "")).strip()
        ctx.gate.check_command(command)
        timeout = min(
            int(args.get("timeout_seconds") or ctx.command_timeout_seconds),
            ctx.command_timeout_seconds,
        )
        env = ctx.gate.scrubbed_env(pycache_prefix=ctx.pycache_prefix)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(ctx.workspace),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return ToolResult(ok=False, error=f"failed to spawn command: {exc}")
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(
                ok=False,
                error=f"command timed out after {timeout}s: {command}",
                metadata={"timeout": True, "exit_code": None},
            )
        text = (stdout or b"").decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0
        body = truncate_middle(text, ctx.max_output_chars)
        output = f"$ {command}\n{body}\n[exit {exit_code}]"
        return ToolResult(
            ok=exit_code == 0,
            output=output if exit_code == 0 else "",
            error=output if exit_code != 0 else "",
            metadata={"exit_code": exit_code, "command": command},
        )


def split_command(command: str) -> list:
    try:
        return shlex.split(command)
    except ValueError:
        return []
