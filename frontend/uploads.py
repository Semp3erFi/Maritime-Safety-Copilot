"""Save small Markdown uploads to the shared local knowledge directory."""

import re
from pathlib import Path
from uuid import uuid4

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MAX_UPLOAD_BYTES = 1024 * 1024


def save_markdown(filename: str, content: bytes, directory: Path = DATA_DIR) -> Path:
    basename = re.split(r"[/\\]", filename)[-1]
    if Path(basename).suffix.lower() != ".md":
        raise ValueError("Choose a Markdown (.md) file.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a file no larger than 1 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The Markdown file must use UTF-8 encoding.") from exc
    if not text.strip() or "\x00" in text:
        raise ValueError("The Markdown file must contain non-empty text without null bytes.")
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(basename).stem).strip("_")[:60] or "note"
    # Prefix avoids Windows reserved names; a unique suffix preserves existing documents.
    target = directory.resolve() / f"upload_{stem}_{uuid4().hex}.md"
    temporary = target.with_suffix(".tmp")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as file:
            file.write(content)
        # The loader ignores .tmp files and only sees the complete Markdown document.
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
