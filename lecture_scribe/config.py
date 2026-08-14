from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .paths import OUTPUT_ROOT, PROMPT_PATH, SETTINGS_PATH


@dataclass(slots=True)
class AppSettings:
    output_dir: str = str(OUTPUT_ROOT)
    last_url: str = ""
    whisper_model: str = "large-v3"
    language: str = "auto"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    clip_start_seconds: float = 0.0
    clip_end_seconds: float | None = None
    keep_audio: bool = False
    keep_video: bool = False
    auto_terminology: bool = False
    terminology_lecture_name: str = ""
    cookie_browser: str = "none"
    codex_model: str = "gpt-5.6-sol"
    reasoning_effort: str = "high"
    yt_dlp_channel: str = "nightly"
    prompt_path: str = str(PROMPT_PATH)

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "AppSettings":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            allowed = {field.name for field in fields(cls)}
            values = {key: value for key, value in raw.items() if key in allowed}
            return cls(**values)
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path = SETTINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def prompt_file(self) -> Path:
        return Path(self.prompt_path).expanduser().resolve()


def load_prompt(settings: AppSettings) -> str:
    prompt_path = settings.prompt_file()
    if not prompt_path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def save_prompt(settings: AppSettings, content: str) -> None:
    prompt_path = settings.prompt_file()
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(content.rstrip() + "\n", encoding="utf-8")
