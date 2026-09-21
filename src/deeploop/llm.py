"""Single choke point for model calls: budget check, usage accounting, ledger."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .budget import BudgetTracker, Usage
from .contract import TaskContract
from .events import E, EventBus
from .ledger import Ledger
from .providers.base import ChatMessage, Completion, Provider


class ModelRunner:
    def __init__(
        self,
        provider: Provider,
        contract: TaskContract,
        budget: BudgetTracker,
        ledger: Ledger,
        bus: EventBus,
        actor_model_override: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.contract = contract
        self.budget = budget
        self.ledger = ledger
        self.bus = bus
        self.actor_model_override = actor_model_override
        self.current_iteration: Optional[int] = None

    def model_for(self, role: str) -> str:
        if role == "actor" and self.actor_model_override:
            return self.actor_model_override
        return self.contract.model.for_role(role)

    async def call(
        self,
        role: str,
        messages: List[ChatMessage],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Completion:
        self.budget.check(kind="llm_call")
        chosen = model or self.model_for(role)
        completion = await self.provider.complete(
            messages,
            model=chosen,
            tools=tools,
            role=role,
            max_tokens=max_tokens,
        )
        usage = Usage(
            role=role,
            model=chosen,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            cached_tokens=completion.usage.cached_tokens,
            cost_usd=completion.usage.cost_usd,
        )
        self.budget.add_usage(usage)
        self.ledger.append(
            "llm_call",
            iteration=self.current_iteration,
            role=role,
            model=chosen,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            latency_ms=completion.latency_ms,
        )
        self.ledger.append(
            "cost", iteration=self.current_iteration, role=role, model=chosen, cost_usd=usage.cost_usd
        )
        await self.bus.emit(E.BUDGET, **self.budget.snapshot())
        if self.budget.near_limit():
            await self.bus.emit(
                E.LOG,
                level="warning",
                message=(
                    f"budget warning: ${self.budget.spent_usd:.4f} of "
                    f"${self.budget.budget.max_usd:.2f} spent"
                ),
            )
        return completion
