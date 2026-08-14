from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "lecture_notes_ko.md"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "default_lecture_notes_ko.md"
SETTINGS_PATH = PROJECT_ROOT / "settings.json"
LOCAL_CODEX_ROOT = PROJECT_ROOT / "tools" / "codex-cli"


def safe_filename(value: str, max_length: int = 80) -> str:
    """Return a portable, readable file or directory name."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return (cleaned or "untitled")[:max_length].rstrip(" .")
