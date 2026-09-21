import struct
import zlib
from pathlib import Path

from deeploop.brief import BriefConfig, image_size, load_brief

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def make_png(path: Path, width: int = 120, height: int = 80) -> None:
    raw = b"".join(b"\x00" + bytes((10, 20, 30)) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + tag + data + crc

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(raw))
    path.write_bytes(PNG_HEADER + ihdr + idat + chunk(b"IEND", b""))


def make_brief(tmp_path: Path) -> Path:
    brief = tmp_path / "brief"
    (brief / "context").mkdir(parents=True, exist_ok=True)
    (brief / "assets").mkdir(parents=True, exist_ok=True)
    (brief / "BRIEF.md").write_text("# Goal\n\nMake the widget work.\n\n## Constraints\n\n- no new deps\n")
    (brief / "context" / "notes.md").write_text("# Notes\n\nBackground detail.\n")
    (brief / "assets" / "mock.png").write_bytes(b"not really a png")
    make_png(brief / "assets" / "status.png", 320, 200)
    return brief


def test_load_brief_finds_brief_docs_and_images(tmp_path: Path) -> None:
    brief_dir = make_brief(tmp_path)
    bundle = load_brief(brief_dir, BriefConfig(dir="brief"))
    assert bundle.brief_file == "BRIEF.md"
    assert "Make the widget work" in bundle.brief_text
    docs = {doc.rel for doc in bundle.docs()}
    assert {"BRIEF.md", "context/notes.md"} <= docs
    images = {image.rel: image for image in bundle.images()}
    assert set(images) == {"assets/mock.png", "assets/status.png"}
    assert images["assets/status.png"].summary == "320x200px"
    assert images["assets/mock.png"].summary == "image"


def test_manifest_lists_context_and_hashes(tmp_path: Path) -> None:
    bundle = load_brief(make_brief(tmp_path))
    manifest = bundle.manifest()
    assert "BRIEF DIRECTORY" in manifest
    assert "context/notes.md" in manifest
    assert "320x200px" in manifest
    assert "Make the widget work" in manifest
    hashes = bundle.hashes()
    assert hashes["BRIEF.md"] and hashes["assets/status.png"]


def test_manifest_includes_vision_descriptions(tmp_path: Path) -> None:
    bundle = load_brief(make_brief(tmp_path))
    bundle.images()[0].description = "A red error dialog with a stack trace."
    assert "A red error dialog" in bundle.manifest()


def test_excluded_paths_are_skipped(tmp_path: Path) -> None:
    brief_dir = make_brief(tmp_path)
    bundle = load_brief(brief_dir, exclude=[brief_dir / "context"])
    assert all("context/" not in doc.rel for doc in bundle.docs())


def test_skips_caches_and_hidden_dirs(tmp_path: Path) -> None:
    brief_dir = make_brief(tmp_path)
    (brief_dir / ".deeploop").mkdir()
    (brief_dir / ".deeploop" / "ledger.jsonl").write_text("{}")
    (brief_dir / "__pycache__").mkdir()
    (brief_dir / "__pycache__" / "x.md").write_text("cached")
    bundle = load_brief(brief_dir)
    rels = {item.rel for item in bundle.files}
    assert not any(rel.startswith(".deeploop") or rel.startswith("__pycache__") for rel in rels)


def test_missing_brief_dir_returns_empty(tmp_path: Path) -> None:
    bundle = load_brief(tmp_path / "nope")
    assert bundle.files == []
    assert bundle.brief_text == ""


def test_image_size_reads_png_and_gif(tmp_path: Path) -> None:
    png = tmp_path / "a.png"
    make_png(png, 64, 48)
    assert image_size(png) == (64, 48)
    gif = tmp_path / "b.gif"
    gif.write_bytes(b"GIF89a" + struct.pack("<HH", 12, 34) + b"\x00" * 10)
    assert image_size(gif) == (12, 34)
    assert image_size(tmp_path / "missing.png") is None
