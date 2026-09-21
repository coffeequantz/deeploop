"""Tool registry construction."""

from __future__ import annotations

from typing import List

from .base import Tool, ToolContext, ToolRegistry, ToolResult
from .files import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from .git import CheckpointManager, GitDiffTool, GitLogTool, GitStatusTool
from .shell import RunCommandTool

__all__ = [
    "CheckpointManager",
    "EditFileTool",
    "ListFilesTool",
    "ReadFileTool",
    "RunCommandTool",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "WriteFileTool",
    "default_registry",
]


def default_registry() -> ToolRegistry:
    tools: List[Tool] = [
        RunCommandTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListFilesTool(),
        GitStatusTool(),
        GitDiffTool(),
        GitLogTool(),
    ]
    return ToolRegistry(tools)
