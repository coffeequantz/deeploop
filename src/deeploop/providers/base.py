"""Provider abstraction: one OpenAI-compatible client covers DeepSeek,
OpenRouter and Ollama. A scripted mock provider powers tests and demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass
class ChatMessage:
    role: str
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    parts: Optional[List[Dict[str, Any]]] = None

    def to_openai(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role}
        if self.parts:
            msg["content"] = self.parts
        elif self.content:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                }
                for call in self.tool_calls
            ]
            msg.setdefault("content", "")
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(role="assistant", content=content)

    @classmethod
    def tool_result(cls, call_id: str, name: str, content: str) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=call_id, name=name)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Completion:
    message: ChatMessage
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: int = 0
    role: str = "actor"


class ProviderError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class Provider:
    name = "base"

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
        raise NotImplementedError

    async def aclose(self) -> None:
        return None
