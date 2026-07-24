from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptContext:
    title: str
    channel: str
    url: str
    duration: str
    language: str
    transcript: str


PLACEHOLDERS = {
    "{{title}}": "영상 제목",
    "{{channel}}": "채널명",
    "{{url}}": "영상 URL",
    "{{duration}}": "영상 길이",
    "{{language}}": "감지 언어",
    "{{transcript}}": "타임스탬프 포함 전사문",
}


def validate_prompt(template: str) -> None:
    if "{{transcript}}" not in template:
        raise ValueError("프롬프트에는 반드시 {{transcript}} 자리표시자가 있어야 합니다.")
    if len(template.strip()) < 80:
        raise ValueError("프롬프트가 너무 짧습니다. 원하는 노트 구조와 제약을 더 구체적으로 적어주세요.")


def render_prompt(template: str, context: PromptContext) -> str:
    validate_prompt(template)
    values = {
        "{{title}}": context.title,
        "{{channel}}": context.channel,
        "{{url}}": context.url,
        "{{duration}}": context.duration,
        "{{language}}": context.language,
        "{{transcript}}": context.transcript,
    }
    rendered = template
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, value)
    return rendered

