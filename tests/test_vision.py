import asyncio
from pathlib import Path

from test_brief import make_brief

from deeploop.brief import load_brief
from deeploop.budget import BudgetTracker
from deeploop.contract import TaskContract
from deeploop.events import EventBus
from deeploop.ledger import Ledger
from deeploop.llm import ModelRunner
from deeploop.providers.mock import MockProvider, MockTurn
from deeploop.vision import describe_images, image_cache_path


def make_runner(tmp_path: Path, turns=None, planner_text: str = "plan") -> ModelRunner:
    contract = TaskContract.model_validate(
        {"goal": "g", "success_criteria": [{"id": "x", "check": "true"}]}
    )
    provider = MockProvider(turns=turns, planner_text=planner_text)
    return ModelRunner(
        provider, contract, BudgetTracker(contract.budget), Ledger(tmp_path / "ledger.jsonl"), EventBus()
    )


def test_describe_images_calls_vision_model_and_caches(tmp_path: Path) -> None:
    bundle = load_brief(make_brief(tmp_path))
    runner = make_runner(tmp_path, turns=[MockTurn(content="A red dialog showing a stack trace.")])
    cache_path = image_cache_path(tmp_path / "artifacts")

    cache = asyncio.run(
        describe_images(runner, bundle, cache_path, model="mock-vision", max_images=1)
    )
    assert len(cache) == 1
    described = [image for image in bundle.images() if image.description]
    assert described and "stack trace" in described[0].description
    assert cache_path.exists()
    assert any(call["role"] == "vision" for call in runner.provider.calls)

    first_call_count = len(runner.provider.calls)
    bundle2 = load_brief(make_brief(tmp_path))
    asyncio.run(describe_images(runner, bundle2, cache_path, model="mock-vision", max_images=1))
    assert len(runner.provider.calls) == first_call_count
    assert any(image.description for image in bundle2.images())


def test_no_vision_model_means_no_descriptions(tmp_path: Path) -> None:
    bundle = load_brief(make_brief(tmp_path))
    runner = make_runner(tmp_path)
    cache = asyncio.run(describe_images(runner, bundle, image_cache_path(tmp_path / "artifacts"), model=None))
    assert cache == {}
    assert all(not image.description for image in bundle.images())


def test_vision_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    from deeploop.providers.base import ProviderError

    class BrokenProvider(MockProvider):
        async def complete(
            self, messages, *, model, tools=None, role="actor", temperature=None, max_tokens=None
        ):
            if role == "vision":
                raise ProviderError("no vision support", status=400)
            return await super().complete(
                messages, model=model, tools=tools, role=role, temperature=temperature, max_tokens=max_tokens
            )

    bundle = load_brief(make_brief(tmp_path))
    contract = TaskContract.model_validate(
        {"goal": "g", "success_criteria": [{"id": "x", "check": "true"}]}
    )
    runner = ModelRunner(
        BrokenProvider(),
        contract,
        BudgetTracker(contract.budget),
        Ledger(tmp_path / "ledger.jsonl"),
        EventBus(),
    )
    asyncio.run(
        describe_images(runner, bundle, image_cache_path(tmp_path / "artifacts"), model="mock-vision")
    )
    assert any("vision failed" in (image.description or "") for image in bundle.images())
