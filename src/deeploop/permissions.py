"""Execution-layer permission enforcement.

Nothing here is advisory: tools call the gate before touching the filesystem or
spawning a process, and a denial is a hard failure the model cannot talk its way
past. Shell scoping is best-effort (documented); use a container for hard
isolation.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contract import Permissions

SHELL_OPERATORS = re.compile(r"\|\||&&|;|\n|\|")
SEGMENT_OPERATORS = {";", "&&", "||", "|", "&"}
OPERATOR_CHARS = set(";&|")
ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
SUBSTITUTION = re.compile(r"\$\(|`")
SAFE_ABSOLUTE_PATHS = {"/dev/null", "/dev/stdout", "/dev/stderr"}


class PermissionDenied(Exception):
    def __init__(self, message: str, tool: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.tool = tool
        self.detail = detail


class PermissionGate:
    def __init__(self, perms: Permissions, workspace: Path) -> None:
        self.perms = perms
        self.workspace = Path(workspace).resolve()
        self.read_roots = self._roots(perms.read_paths)
        self.write_roots = self._roots(perms.write_paths)
        self.allow_commands = {c.strip() for c in perms.allow_commands if c.strip()}
        self.deny_commands = [c for c in perms.deny_commands if c]
        self.allow_all_commands = perms.allow_all_commands
        self.network = perms.network
        self.require_approval = {t.strip() for t in perms.require_approval if t.strip()}

    def _roots(self, raw_paths: List[str]) -> List[Path]:
        roots = []
        for raw in raw_paths:
            p = Path(raw)
            if not p.is_absolute():
                p = self.workspace / p
            roots.append(p.resolve())
        return roots

    def _within(self, path: Path, roots: List[Path]) -> bool:
        for root in roots:
            if path == root or root in path.parents:
                return True
        return False

    def check_path(self, raw: str, mode: str = "read") -> Path:
        if not raw or not str(raw).strip():
            raise PermissionDenied("empty path", tool=f"file_{mode}")
        p = Path(str(raw))
        if not p.is_absolute():
            p = self.workspace / p
        resolved = p.resolve()
        roots = self.write_roots if mode == "write" else self.read_roots
        if not self._within(resolved, roots):
            raise PermissionDenied(
                f"{mode} path outside allowed roots: {resolved}",
                tool=f"file_{mode}",
                detail=f"allowed: {', '.join(str(r) for r in roots)}",
            )
        return resolved

    def check_command(self, command: str) -> None:
        if not command or not command.strip():
            raise PermissionDenied("empty command", tool="run_command")
        for pattern in self.deny_commands:
            if pattern in command or re.search(re.escape(pattern), command):
                raise PermissionDenied(
                    f"command matches deny pattern {pattern!r}", tool="run_command"
                )
        if self.allow_all_commands:
            return
        if SUBSTITUTION.search(command):
            raise PermissionDenied(
                "command substitution is not allowed unless allow_all_commands=true",
                tool="run_command",
            )
        for tokens in self._split_segments(command):
            tokens = [t for t in tokens if not ENV_ASSIGNMENT.match(t)]
            if not tokens:
                continue
            binary = tokens[0]
            name = Path(binary).name
            if name in {"sudo", "su", "doas"}:
                raise PermissionDenied(f"privilege escalation blocked: {name}", tool="run_command")
            if binary not in self.allow_commands and name not in self.allow_commands:
                raise PermissionDenied(
                    f"command not in allowlist: {binary}",
                    tool="run_command",
                    detail="add it to permissions.allow_commands or set allow_all_commands=true",
                )
            if name == "cd" and len(tokens) > 1:
                self._check_command_paths(tokens[1:2])
            self._check_command_paths(tokens[1:])

    def _split_segments(self, command: str) -> List[List[str]]:
        """Quote-aware split on shell operators, so `python3 -c 'a; b'` stays one segment."""
        segments: List[List[str]] = []
        for line in command.splitlines():
            if not line.strip():
                continue
            try:
                lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
                lexer.whitespace_split = True
                tokens = list(lexer)
            except ValueError as exc:
                raise PermissionDenied(f"unparseable command: {exc}", tool="run_command")
            current: List[str] = []
            for token in tokens:
                if token in SEGMENT_OPERATORS or (token and set(token) <= OPERATOR_CHARS):
                    if current:
                        segments.append(current)
                        current = []
                    continue
                current.append(token)
            if current:
                segments.append(current)
        return segments

    def _check_command_paths(self, tokens: List[str]) -> None:
        for token in tokens:
            if token in SAFE_ABSOLUTE_PATHS:
                continue
            if token.startswith("-") or "=" in token:
                continue
            candidate = Path(token)
            if candidate.is_absolute() and not self._within(candidate.resolve(), self.read_roots):
                raise PermissionDenied(
                    f"absolute path outside workspace: {token}",
                    tool="run_command",
                    detail="extend permissions.read_paths if this is intentional",
                )

    def check_tool(self, tool: str) -> None:
        if tool in self.require_approval:
            raise PermissionDenied(
                f"tool requires explicit human approval: {tool}",
                tool=tool,
                detail="remove it from permissions.require_approval to allow",
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "network": self.network,
            "allow_all_commands": self.allow_all_commands,
            "allow_commands": sorted(self.allow_commands),
            "deny_commands": list(self.deny_commands),
            "read_roots": [str(p) for p in self.read_roots],
            "write_roots": [str(p) for p in self.write_roots],
            "require_approval": sorted(self.require_approval),
        }

    def scrubbed_env(
        self,
        base: Optional[Dict[str, str]] = None,
        pycache_prefix: Optional[Path] = None,
    ) -> Dict[str, str]:
        import os

        env = dict(base if base is not None else os.environ)
        for key in list(env):
            upper = key.upper()
            if any(marker in upper for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
                env.pop(key, None)
        if not self.network:
            env["http_proxy"] = "http://127.0.0.1:9"
            env["https_proxy"] = "http://127.0.0.1:9"
            env["HTTP_PROXY"] = "http://127.0.0.1:9"
            env["HTTPS_PROXY"] = "http://127.0.0.1:9"
            env["NO_PROXY"] = ""
            env["no_proxy"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if pycache_prefix is not None:
            env["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)
        return env
