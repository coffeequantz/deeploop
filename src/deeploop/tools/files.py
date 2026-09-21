"""Filesystem tools, path-scoped by the permission gate."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List

from ..util import truncate_middle, truncate_tail
from .base import Tool, ToolContext, ToolResult

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".deeploop", "dist", "build", ".pytest_cache"}


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file from the workspace. Optionally read a line range."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace."},
            "start_line": {"type": "integer", "description": "1-based first line to read."},
            "max_lines": {"type": "integer", "description": "Maximum number of lines to return."},
        },
        "required": ["path"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.gate.check_path(str(args.get("path", "")), mode="read")
        if not path.exists():
            return ToolResult(ok=False, error=f"file not found: {path}")
        if path.is_dir():
            return ToolResult(ok=False, error=f"path is a directory: {path}")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(ok=False, error=f"cannot read {path}: {exc}")
        lines = text.splitlines()
        start = max(1, int(args.get("start_line") or 1))
        max_lines = int(args.get("max_lines") or 0)
        selected = lines[start - 1 :]
        if max_lines > 0:
            selected = selected[:max_lines]
        numbered = "\n".join(f"{start + i:>5} | {line}" for i, line in enumerate(selected))
        body = truncate_middle(numbered, ctx.max_output_chars)
        header = f"{path} ({len(lines)} lines total)"
        return ToolResult(ok=True, output=f"{header}\n{body}", metadata={"path": str(path)})


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a file with the given content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace."},
            "content": {"type": "string", "description": "Full file content to write."},
        },
        "required": ["path", "content"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.gate.check_path(str(args.get("path", "")), mode="write")
        content = str(args.get("content", ""))
        existed = path.exists()
        previous_size = path.stat().st_size if existed else 0
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(ok=False, error=f"cannot write {path}: {exc}")
        action = "overwrote" if existed else "created"
        return ToolResult(
            ok=True,
            output=f"{action} {path} ({previous_size} -> {len(content.encode('utf-8'))} bytes)",
            metadata={"path": str(path), "created": not existed},
        )


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. Fails unless the match is unique, "
        "so include enough surrounding context."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.gate.check_path(str(args.get("path", "")), mode="write")
        if not path.exists():
            return ToolResult(ok=False, error=f"file not found: {path}")
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        if not old:
            return ToolResult(ok=False, error="old_string must not be empty")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ToolResult(ok=False, error="old_string not found in file")
        replace_all = bool(args.get("replace_all"))
        if count > 1 and not replace_all:
            return ToolResult(
                ok=False,
                error=f"old_string matches {count} times; add context or set replace_all=true",
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"edited {path}: {count if replace_all else 1} replacement(s)",
            metadata={"path": str(path), "replacements": count if replace_all else 1},
        )


class ListFilesTool(Tool):
    name = "list_files"
    description = "List workspace files, optionally filtered by a glob pattern."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list, relative to the workspace."},
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'."},
            "max_entries": {"type": "integer", "description": "Cap on returned entries."},
        },
        "required": [],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.gate.check_path(str(args.get("path") or "."), mode="read")
        pattern = str(args.get("pattern") or "**/*")
        max_entries = min(int(args.get("max_entries") or 200), 1000)
        entries: List[str] = []
        if root.is_file():
            entries.append(str(root.relative_to(ctx.workspace)))
        else:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for name in sorted(filenames):
                    full = Path(dirpath) / name
                    try:
                        rel = full.relative_to(ctx.workspace)
                    except ValueError:
                        rel = full
                    if fnmatch.fnmatch(str(rel), pattern) or fnmatch.fnmatch(name, pattern):
                        entries.append(str(rel))
                    if len(entries) >= max_entries:
                        break
                if len(entries) >= max_entries:
                    break
        body = "\n".join(entries) if entries else "(no matches)"
        return ToolResult(
            ok=True,
            output=truncate_tail(f"{len(entries)} file(s) under {root}:\n{body}", ctx.max_output_chars),
            metadata={"count": len(entries)},
        )
