"""One interface over the OAuth-backed AI CLIs (Codex for ChatGPT, Claude Code for Claude)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from . import claude_client, codex_client
from .codex_client import AuthStatus, LogCallback


OPENAI = "openai"
CLAUDE = "claude"
PROVIDER_NAMES = {OPENAI: "ChatGPT", CLAUDE: "Claude"}
PROVIDER_CHOICES = (
    (OPENAI, "ChatGPT · OpenAI 계정"),
    (CLAUDE, "Claude · Anthropic 계정"),
)
TOOL_NAMES = {OPENAI: "Codex CLI", CLAUDE: "Claude Code CLI"}


@dataclass(frozen=True, slots=True)
class ModelInfo:
    value: str
    label: str
    description: str = ""
    efforts: tuple[str, ...] = ()
    default_effort: str = ""


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    provider: str
    models: tuple[ModelInfo, ...]
    live: bool
    detail: str = ""


_GPT_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Shown until the installed CLI reports the account's current catalog, and
# used when it cannot (not installed, offline).
FALLBACK_MODELS: dict[str, tuple[ModelInfo, ...]] = {
    OPENAI: (
        ModelInfo("gpt-6-astra", "GPT-6-Astra", "", _GPT_EFFORTS + ("ultra",), "low"),
        ModelInfo("gpt-5.6-sol", "GPT-5.6-Sol", "", _GPT_EFFORTS + ("ultra",), "low"),
        ModelInfo("gpt-5.6-terra", "GPT-5.6-Terra", "", _GPT_EFFORTS + ("ultra",), "medium"),
        ModelInfo("gpt-5.6-luna", "GPT-5.6-Luna", "", _GPT_EFFORTS, "medium"),
        ModelInfo("gpt-5.5", "GPT-5.5", "", ("low", "medium", "high", "xhigh"), "medium"),
    ),
    CLAUDE: (
        ModelInfo("opus", "Opus 5.5", "Best for everyday, complex tasks", _CLAUDE_EFFORTS, "high"),
        ModelInfo(
            "claude-fable-5-1[1m]",
            "Fable 5.1",
            "Most capable for your hardest and longest-running tasks",
            _CLAUDE_EFFORTS,
            "high",
        ),
        ModelInfo("sonnet", "Sonnet 5", "Efficient for routine tasks", _CLAUDE_EFFORTS, "high"),
        ModelInfo("haiku", "Haiku 4.5", "Fastest for quick answers"),
    ),
}


def normalize_provider(provider: str) -> str:
    return provider if provider in PROVIDER_NAMES else OPENAI


def _client(provider: str):
    return claude_client if normalize_provider(provider) == CLAUDE else codex_client


def find_tool(provider: str) -> Path | None:
    if normalize_provider(provider) == CLAUDE:
        return claude_client.find_claude()
    return codex_client.find_codex()


def auth_status(provider: str) -> AuthStatus:
    return _client(provider).auth_status()


def login(
    provider: str,
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> AuthStatus:
    return _client(provider).login(log, cancel_event)


def logout(provider: str) -> AuthStatus:
    return _client(provider).logout()


def fallback_catalog(provider: str, detail: str = "") -> ModelCatalog:
    provider = normalize_provider(provider)
    return ModelCatalog(provider, FALLBACK_MODELS[provider], live=False, detail=detail)


def fetch_catalog(provider: str) -> ModelCatalog:
    """Load the current model list from the provider's CLI, or fall back to the built-in one."""
    provider = normalize_provider(provider)
    try:
        raw = _client(provider).fetch_model_catalog()
        models = (
            _parse_claude_models(raw) if provider == CLAUDE else _parse_codex_models(raw)
        )
    except (OSError, RuntimeError) as exc:
        return fallback_catalog(provider, str(exc))
    if not models:
        return fallback_catalog(provider, "CLI가 사용할 수 있는 모델을 알려주지 않았습니다.")
    return ModelCatalog(provider, models, live=True)


def _parse_codex_models(raw: list[dict]) -> tuple[ModelInfo, ...]:
    listed = [
        item
        for item in raw
        if isinstance(item, dict) and item.get("slug") and item.get("visibility", "list") == "list"
    ]
    listed.sort(key=lambda item: item.get("priority", 1_000))
    models = []
    for item in listed:
        efforts = tuple(
            str(level["effort"])
            for level in item.get("supported_reasoning_levels") or ()
            if isinstance(level, dict) and level.get("effort")
        )
        models.append(
            ModelInfo(
                value=str(item["slug"]),
                label=str(item.get("display_name") or item["slug"]),
                description=str(item.get("description") or ""),
                efforts=efforts,
                default_effort=str(item.get("default_reasoning_level") or ""),
            )
        )
    return tuple(models)


def _parse_claude_models(raw: list[dict]) -> tuple[ModelInfo, ...]:
    models = []
    for item in raw:
        # "default" is a moving pointer to one of the other entries.
        if not isinstance(item, dict) or not item.get("value") or item["value"] == "default":
            continue
        # e.g. "Opus 5.5 · Best for everyday, complex tasks · $4/$20 per Mtok"
        parts = [part.strip() for part in str(item.get("description") or "").split("·")]
        label = parts[0] if parts and parts[0] else str(item.get("displayName") or item["value"])
        efforts = tuple(str(level) for level in item.get("supportedEffortLevels") or ())
        models.append(
            ModelInfo(
                value=str(item["value"]),
                label=label,
                description=" · ".join(part for part in parts[1:] if part),
                efforts=efforts,
                default_effort="high" if "high" in efforts else (efforts[0] if efforts else ""),
            )
        )
    return tuple(models)


def generate_notes(
    prompt: str,
    output_path: Path,
    *,
    provider: str = OPENAI,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "high",
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> Path:
    final_text = _client(provider).generate_text(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        log=log,
        cancel_event=cancel_event,
        activity="강의 노트를 생성합니다",
        failure_message="AI 노트 생성에 실패했습니다.",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_text.rstrip() + "\n", encoding="utf-8")
    return output_path.resolve()


def generate_transcription_terms(
    lecture_name: str,
    *,
    provider: str = OPENAI,
    video_title: str = "",
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "medium",
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
    max_terms: int = 40,
) -> tuple[str, ...]:
    name = " ".join(lecture_name.split()).strip()
    if not name:
        raise ValueError("전문용어를 만들 강의명을 입력해주세요.")
    prompt = f"""당신은 음성 인식용 전문용어 사전을 만드는 조교입니다.

사용자가 입력한 강의명: {name}
YouTube 영상 제목: {video_title.strip() or '(제공되지 않음)'}

이 강의에서 실제로 발음될 가능성이 높은 한국어·영어 전문용어, 고유명사, 약어, 인명, 제품명, 라이브러리명, 수식 명칭을 최대 {max_terms}개 선정하세요.
Whisper 음성 인식의 철자 정확도를 높이는 것이 목적입니다. 일반적인 쉬운 단어, 설명, 번역, 문장, 추측성이 큰 단어는 제외하세요.
각 줄에는 전문용어 하나만 쓰고 번호, 글머리표, Markdown 코드블록, 부연 설명을 절대 넣지 마세요.
"""
    raw = _client(provider).generate_text(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        log=log,
        cancel_event=cancel_event,
        activity=f"‘{name}’ 강의의 전사 전문용어를 생성합니다",
        failure_message="전사 전문용어 생성에 실패했습니다.",
    )
    terms = _parse_transcription_terms(raw, max_terms=max_terms)
    if not terms:
        raise RuntimeError(
            f"{PROVIDER_NAMES[normalize_provider(provider)]}가 사용할 수 있는 전문용어를 반환하지 않았습니다."
        )
    return terms


def _parse_transcription_terms(value: str, max_terms: int = 40) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    cleaned = codex_client._clean_output(value).replace("```text", "").replace("```", "")
    for line in cleaned.splitlines():
        candidate = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        candidate = candidate.strip("`'\" ")
        if not candidate or len(candidate) > 80 or candidate.endswith(":"):
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(candidate)
        if len(terms) >= max_terms:
            break
    return tuple(terms)
