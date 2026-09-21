"""Token pricing table. Prices change; override per mission in
`provider.pricing` of the contract. Values are USD per 1M tokens."""

from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # DeepSeek (cache-hit / cache-miss input split)
    "deepseek-chat": {"input": 0.27, "cached_input": 0.07, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "cached_input": 0.14, "output": 2.19},
    "deepseek-coder": {"input": 0.27, "cached_input": 0.07, "output": 1.10},
    # Free local inference
    "ollama": {"input": 0.0, "cached_input": 0.0, "output": 0.0},
}


def price_for(
    model: str, overrides: Optional[Dict[str, Dict[str, float]]] = None
) -> Optional[Dict[str, float]]:
    if overrides:
        if model in overrides:
            return overrides[model]
        for key, value in overrides.items():
            if model.startswith(key):
                return value
    if model in DEFAULT_PRICING:
        return DEFAULT_PRICING[model]
    for key, value in DEFAULT_PRICING.items():
        if model.startswith(key):
            return value
    if "/" in model:  # openrouter style: vendor/model — fall back to the tail
        tail = model.split("/")[-1]
        return price_for(tail, overrides)
    return None


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    price = price_for(model, overrides)
    if not price:
        return 0.0
    cached = max(0, min(cached_tokens, prompt_tokens))
    fresh = max(0, prompt_tokens - cached)
    per_m = 1_000_000.0
    cost = (
        fresh * price.get("input", 0.0)
        + cached * price.get("cached_input", price.get("input", 0.0))
        + completion_tokens * price.get("output", 0.0)
    ) / per_m
    return round(cost, 8)


def known_models(overrides: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
    table = dict(DEFAULT_PRICING)
    if overrides:
        table.update(overrides)
    return table


def as_table_rows(overrides: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    rows = []
    for model, price in known_models(overrides).items():
        rows.append(
            {
                "model": model,
                "input_per_m": price.get("input", 0.0),
                "cached_input_per_m": price.get("cached_input", price.get("input", 0.0)),
                "output_per_m": price.get("output", 0.0),
            }
        )
    return {"models": rows}
