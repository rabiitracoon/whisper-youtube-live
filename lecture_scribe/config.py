from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from uuid import uuid4

from .paths import DEFAULT_PROMPT_PATH, OUTPUT_ROOT, PROMPT_PATH, SETTINGS_PATH


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
    terminology_model: str = "gpt-5.6-luna"
    cookie_browser: str = "none"
    ai_provider: str = "openai"
    codex_model: str = "gpt-5.6-sol"
    reasoning_effort: str = "high"
    claude_model: str = "opus"
    claude_reasoning_effort: str = "high"
    claude_terminology_model: str = "sonnet"
    yt_dlp_channel: str = "nightly"
    prompt_path: str = str(PROMPT_PATH)
    prompt_slots: list[dict[str, str]] = field(default_factory=list)
    active_prompt_slot_id: str = ""

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "AppSettings":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            allowed = {field.name for field in fields(cls)}
            values = {key: value for key, value in raw.items() if key in allowed}
            settings = cls(**values)
            settings.ensure_prompt_slots()
            return settings
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

    def ai_selection(self, provider: str | None = None) -> tuple[str, str, str]:
        """Return (notes model, reasoning effort, terminology model) for a provider."""
        if (provider or self.ai_provider) == "claude":
            return self.claude_model, self.claude_reasoning_effort, self.claude_terminology_model
        return self.codex_model, self.reasoning_effort, self.terminology_model

    def set_ai_selection(
        self, provider: str, notes_model: str, reasoning_effort: str, terminology_model: str
    ) -> None:
        if provider == "claude":
            self.claude_model = notes_model
            self.claude_reasoning_effort = reasoning_effort
            self.claude_terminology_model = terminology_model
        else:
            self.codex_model = notes_model
            self.reasoning_effort = reasoning_effort
            self.terminology_model = terminology_model

    def prompt_file(self) -> Path:
        return Path(self.prompt_path).expanduser().resolve()

    def ensure_prompt_slots(self) -> None:
        """Migrate the former single prompt into the first reusable slot."""
        valid_slots = [
            {"id": str(item["id"]), "name": str(item["name"]), "content": str(item["content"])}
            for item in self.prompt_slots
            if isinstance(item, dict)
            and item.get("id")
            and item.get("name")
            and isinstance(item.get("content"), str)
        ]
        if not valid_slots:
            try:
                content = self.prompt_file().read_text(encoding="utf-8")
            except OSError:
                content = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
            valid_slots = [{"id": "default", "name": "강의용 기본", "content": content}]
        self.prompt_slots = valid_slots
        if not any(slot["id"] == self.active_prompt_slot_id for slot in valid_slots):
            self.active_prompt_slot_id = valid_slots[0]["id"]

    def active_prompt_slot(self) -> dict[str, str]:
        self.ensure_prompt_slots()
        return next(
            slot for slot in self.prompt_slots if slot["id"] == self.active_prompt_slot_id
        )

    def add_prompt_slot(self, name: str, content: str) -> dict[str, str]:
        self.ensure_prompt_slots()
        slot = {"id": uuid4().hex, "name": name, "content": content}
        self.prompt_slots.append(slot)
        self.active_prompt_slot_id = slot["id"]
        return slot

    def remove_active_prompt_slot(self) -> bool:
        self.ensure_prompt_slots()
        if len(self.prompt_slots) == 1:
            return False
        index = next(
            index
            for index, slot in enumerate(self.prompt_slots)
            if slot["id"] == self.active_prompt_slot_id
        )
        self.prompt_slots.pop(index)
        self.active_prompt_slot_id = self.prompt_slots[max(0, index - 1)]["id"]
        return True


def load_prompt(settings: AppSettings) -> str:
    return settings.active_prompt_slot()["content"]


def save_prompt(settings: AppSettings, content: str) -> None:
    settings.active_prompt_slot()["content"] = content.rstrip() + "\n"
    prompt_path = settings.prompt_file()
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(settings.active_prompt_slot()["content"], encoding="utf-8")
