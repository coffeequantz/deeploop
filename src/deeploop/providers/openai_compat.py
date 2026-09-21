"""OpenAI-compatible chat completions client (DeepSeek, OpenRouter, Ollama)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..contract import ProviderConfig
from .base import ChatMessage, Completion, Provider, ProviderError, ToolCall, Usage
from .pricing import estimate_cost

DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}

DEFAULT_KEY_ENVS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
}

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatProvider(Provider):
    def __init__(self, config: ProviderConfig, name: Optional[str] = None) -> None:
        self.config = config
        self.name = name or config.name
        self.base_url = (config.base_url or DEFAULT_BASE_URLS.get(self.name, "")).rstrip("/")
        if not self.base_url:
            raise ProviderError(f"no base_url configured for provider {self.name!r}")
        key_env = config.api_key_env or DEFAULT_KEY_ENVS.get(self.name)
        self.api_key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not self.api_key:
            raise ProviderError(
                f"missing API key: set {key_env} in the environment "
                f"(or point provider.api_key_env at another variable)"
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds, connect=30.0),
        )

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
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai() for m in messages],
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens:
            payload["max_tokens"] = max_tokens

        started = time.monotonic()
        response = await self._post("/chat/completions", payload)
        latency_ms = int((time.monotonic() - started) * 1000)

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"provider returned non-JSON response: {exc}", response.status_code) from exc
        if "error" in body and not body.get("choices"):
            raise ProviderError(f"provider error: {body['error']}", response.status_code, str(body["error"]))

        choices = body.get("choices") or []
        if not choices:
            raise ProviderError("provider returned no choices", response.status_code, json.dumps(body)[:2000])
        choice = choices[0].get("message") or {}
        tool_calls = self._parse_tool_calls(choice.get("tool_calls") or [])
        message = ChatMessage(
            role="assistant",
            content=choice.get("content") or "",
            tool_calls=tool_calls,
        )
        usage = self._parse_usage(body.get("usage") or {}, model)
        return Completion(message=message, usage=usage, model=model, latency_ms=latency_ms, role=role)

    def _parse_tool_calls(self, raw_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for index, raw in enumerate(raw_calls):
            function = raw.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            args: Dict[str, Any] = {}
            if isinstance(raw_args, dict):
                args = raw_args
                raw_args = json.dumps(raw_args)
            else:
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    args = {}
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{index}",
                    name=function.get("name") or "",
                    arguments=args,
                    raw_arguments=raw_args,
                )
            )
        return calls

    def _parse_usage(self, usage: Dict[str, Any], model: str) -> Usage:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        cached = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = max(cached, int(details.get("cached_tokens", 0) or 0))
        cost = usage.get("cost")
        if cost is None:
            cost = estimate_cost(model, prompt_tokens, completion_tokens, cached, self.config.pricing)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached,
            cost_usd=float(cost or 0.0),
        )

    async def _post(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        attempts = 3
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                response = await self._client.post(path, json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise ProviderError(f"network error calling {self.base_url}: {exc}") from exc
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            if response.status_code in RETRY_STATUS and attempt < attempts - 1:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    f"provider HTTP {response.status_code} from {self.base_url}{path}",
                    response.status_code,
                    response.text[:2000],
                )
            return response
        raise ProviderError(f"provider request failed after {attempts} attempts: {last_error}")

    async def aclose(self) -> None:
        await self._client.aclose()
