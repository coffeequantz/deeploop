"""Optional vision pass: describe brief images once so text-only loops can use them.

Descriptions are cached in `.deeploop/artifacts/image_descriptions.json`, keyed by
file hash, so re-runs and resumes never pay for the same image twice.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Dict, Optional

from .brief import BriefBundle, BriefFile
from .events import E, EventBus
from .llm import ModelRunner
from .providers.base import ChatMessage

VISION_SYSTEM = """You describe images for a coding agent that cannot see them.

Be concrete and complete: transcribe visible text, describe layout and structure,
list colors and typography when they look deliberate, and note anything that looks
like a requirement (annotations, arrows, redlines, dimensions, states).

Respond in plain text, under 250 words, no preamble."""

MAX_IMAGE_BYTES = 4_000_000
MAX_IMAGES = 12


async def describe_images(
    runner: ModelRunner,
    bundle: BriefBundle,
    cache_path: Path,
    bus: Optional[EventBus] = None,
    model: Optional[str] = None,
    max_images: int = MAX_IMAGES,
) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cache = {str(k): str(v) for k, v in loaded.items()}
        except (json.JSONDecodeError, OSError):
            cache = {}

    described = 0
    for image in bundle.images()[:max_images]:
        key = image.sha256 or image.rel
        if key in cache:
            image.description = cache[key]
            continue
        if image.size > MAX_IMAGE_BYTES:
            image.description = f"[not described: {image.size} bytes exceeds the vision limit]"
            continue
        if not model:
            continue
        try:
            description = await _describe_one(runner, image, model)
        except Exception as exc:  # noqa: BLE001 - a failed description must not stop the mission
            image.description = f"[vision failed: {type(exc).__name__}: {exc}]"
            if bus is not None:
                await bus.emit(E.LOG, level="warning", message=f"vision failed for {image.rel}: {exc}")
            continue
        image.description = description
        cache[key] = description
        described += 1
        if bus is not None:
            await bus.emit(E.LOG, level="info", message=f"described image {image.rel}")

    if described:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


async def _describe_one(runner: ModelRunner, image: BriefFile, model: str) -> str:
    raw = image.path.read_bytes()
    mime = mimetypes.guess_type(image.path.name)[0] or "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    message = ChatMessage(
        role="user",
        content=f"Image file: {image.rel}",
        parts=[
            {"type": "text", "text": f"Describe this image for the agent. File: {image.rel}"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ],
    )
    completion = await runner.call(
        "vision",
        [ChatMessage.system(VISION_SYSTEM), message],
        model=model,
    )
    return (completion.message.content or "").strip()


def image_cache_path(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "image_descriptions.json"
