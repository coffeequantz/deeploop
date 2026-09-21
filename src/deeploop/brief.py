"""Brief folders: a directory of human input the agent reads but does not own.

Layout convention:

    brief/
      BRIEF.md          # narrative goal, constraints, non-goals
      context/*.md      # reference docs
      assets/*.png      # images: references to interpret or assets to use

The contract's machine-checkable parts (criteria, budget, permissions) stay in
mission.yaml; the brief supplies narrative context and evidence material.
"""

from __future__ import annotations

import fnmatch
import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .contract import BriefConfig

SKIP_DIRS = {
    ".git",
    ".deeploop",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
BRIEF_NAMES = ("brief.md", "goal.md", "readme.md")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}


@dataclass
class BriefFile:
    rel: str
    path: Path
    kind: str  # "doc" | "image"
    size: int
    summary: str = ""
    text: Optional[str] = None
    sha256: str = ""
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "rel": self.rel,
            "kind": self.kind,
            "size": self.size,
            "summary": self.summary,
            "sha256": self.sha256,
            "described": bool(self.description),
        }


@dataclass
class BriefBundle:
    dir: Path
    brief_text: str = ""
    brief_file: Optional[str] = None
    files: List[BriefFile] = field(default_factory=list)

    def docs(self) -> List[BriefFile]:
        return [f for f in self.files if f.kind == "doc"]

    def images(self) -> List[BriefFile]:
        return [f for f in self.files if f.kind == "image"]

    def hashes(self) -> Dict[str, str]:
        return {f.rel: f.sha256 for f in self.files}

    def manifest(self, max_chars: int = 6000) -> str:
        parts: List[str] = [f"BRIEF DIRECTORY: {self.dir}"]
        if self.brief_text:
            label = self.brief_file or "BRIEF.md"
            parts.append(f"{label}:\n{_clip(self.brief_text, max_chars // 2)}")
        docs = [f for f in self.docs() if f.rel.lower() != (self.brief_file or "").lower()]
        if docs:
            lines = ["CONTEXT FILES (open them with read_file when relevant):"]
            for doc in docs:
                lines.append(f"- {doc.rel} ({_human_size(doc.size)}) — {doc.summary or 'no summary'}")
            parts.append("\n".join(lines))
        images = self.images()
        if images:
            lines = ["IMAGES:"]
            for image in images:
                detail = image.description or image.summary or "not described"
                lines.append(f"- {image.rel} ({_human_size(image.size)}) — {detail}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


def load_brief(
    brief_dir: Path,
    config: Optional[BriefConfig] = None,
    exclude: Optional[List[Path]] = None,
) -> BriefBundle:
    config = config or BriefConfig()
    brief_dir = Path(brief_dir).resolve()
    excluded = {Path(p).resolve() for p in (exclude or [])}
    bundle = BriefBundle(dir=brief_dir)
    if not brief_dir.exists():
        return bundle

    for path in sorted(brief_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        resolved = path.resolve()
        if any(resolved == ex or ex in resolved.parents for ex in excluded):
            continue
        rel = str(path.relative_to(brief_dir))
        if _matches(rel, config.images) or path.suffix.lower() in IMAGE_EXTENSIONS:
            size = _safe_size(path)
            bundle.files.append(
                BriefFile(
                    rel=rel,
                    path=path,
                    kind="image",
                    size=size,
                    summary=_image_summary(path),
                    sha256=_hash_file(path),
                )
            )
            continue
        if not _matches(rel, config.include):
            continue
        text = _read_text(path, config.max_file_bytes)
        if text is None:
            continue
        bundle.files.append(
            BriefFile(
                rel=rel,
                path=path,
                kind="doc",
                size=_safe_size(path),
                summary=_summarize_text(text),
                text=text,
                sha256=_hash_text(text),
            )
        )

    for candidate in BRIEF_NAMES:
        for item in bundle.files:
            if item.kind == "doc" and Path(item.rel).name.lower() == candidate:
                bundle.brief_text = item.text or ""
                bundle.brief_file = item.rel
                break
        if bundle.brief_text:
            break
    return bundle


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with Path(path).open("rb") as fh:
            head = fh.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                width, height = struct.unpack(">II", head[16:24])
                return int(width), int(height)
            if head[:6] in (b"GIF87a", b"GIF89a") and len(head) >= 10:
                width, height = struct.unpack("<HH", head[6:10])
                return int(width), int(height)
            if head.startswith(b"\xff\xd8"):
                return _jpeg_size(fh)
    except OSError:
        return None
    return None


def _jpeg_size(fh) -> Optional[Tuple[int, int]]:
    fh.seek(2)
    while True:
        marker = fh.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            return None
        code = marker[1]
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            length = struct.unpack(">H", fh.read(2))[0]
            if length < 7:
                return None
            data = fh.read(5)
            height, width = struct.unpack(">HH", data[1:5])
            return int(width), int(height)
        length_bytes = fh.read(2)
        if len(length_bytes) < 2:
            return None
        length = struct.unpack(">H", length_bytes)[0]
        fh.seek(length - 2, 1)


def _image_summary(path: Path) -> str:
    size = image_size(path)
    if size:
        return f"{size[0]}x{size[1]}px"
    return "image"


def _matches(rel: str, patterns: List[str]) -> bool:
    name = Path(rel).name
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:]):
            return True
    return False


def _summarize_text(text: str, limit: int = 120) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:limit]
    return ""


def _read_text(path: Path, max_bytes: int) -> Optional[str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw[:max_bytes].decode("utf-8", errors="replace")


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()[:16]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _human_size(size: int) -> str:
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1000:
        return f"{size / 1000:.1f} KB"
    return f"{size} B"


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = f"\n… [{len(text) - max_chars} chars omitted] …\n"
    return text[: max_chars // 2] + omitted + text[-max_chars // 2 :]
