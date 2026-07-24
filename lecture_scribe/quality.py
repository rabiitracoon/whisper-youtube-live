from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .transcription import TranscriptResult
from .youtube import format_duration


@dataclass(frozen=True, slots=True)
class TranscriptReview:
    text_path: Path
    preview_text: str
    status: str
    headline: str
    metrics: str
    warnings: tuple[str, ...]


def assess_transcript(transcript: TranscriptResult) -> TranscriptReview:
    segments = transcript.segments
    text = transcript.text_path.read_text(encoding="utf-8").strip()
    clip_duration = max(
        1.0, sum(end - start for start, end in transcript.clip_ranges)
    )
    speech_duration = sum(max(0.0, item.end - item.start) for item in segments)
    word_count = len(re.findall(r"\S+", text))
    words_per_minute = word_count / (clip_duration / 60)
    speech_ratio = min(1.0, speech_duration / clip_duration)
    logprobs = [item.avg_logprob for item in segments if item.avg_logprob is not None]
    average_logprob = sum(logprobs) / len(logprobs) if logprobs else None

    warnings: list[str] = []
    if transcript.language_probability and transcript.language_probability < 0.65:
        warnings.append("강의 언어를 확실히 구분하지 못했어요. 언어 설정과 앞부분을 확인해주세요.")
    if clip_duration >= 300 and words_per_minute < 18:
        warnings.append("선택한 시간에 비해 말의 양이 매우 적어요. 음악이나 말이 없는 구간이 포함됐을 수 있습니다.")
    if words_per_minute > 330:
        warnings.append("옮겨진 말이 지나치게 많아요. 같은 문장이 반복되거나 잘못 포함된 구간이 없는지 확인해주세요.")
    if average_logprob is not None and average_logprob < -1.15:
        warnings.append("정확하게 듣지 못한 부분이 있을 수 있어요. 고유명사와 전문용어를 특히 확인해주세요.")

    normalized = [re.sub(r"\W+", "", item.text.lower()) for item in segments]
    normalized = [item for item in normalized if len(item) >= 6]
    duplicate_ratio = 0.0
    if normalized:
        duplicate_ratio = 1 - len(set(normalized)) / len(normalized)
    if len(normalized) >= 10 and duplicate_ratio > 0.25:
        warnings.append("같은 문장이 여러 번 반복된 흔적이 있어요. 음악 구간에서 잘못 만들어진 문장인지 확인해주세요.")
    if clip_duration >= 600 and len(segments) < 8:
        warnings.append("긴 구간인데 옮겨진 문장이 매우 적어요. 시작과 종료 시간이 강의 구간에 맞는지 확인해주세요.")

    status = "warning" if warnings else "good"
    headline = "한 번 살펴볼 부분이 있어요" if warnings else "자동 확인 결과가 자연스러워요"
    probability = f"{transcript.language_probability:.0%}" if transcript.language_probability else "정보 없음"
    language_name = {
        "ko": "한국어",
        "en": "영어",
        "ja": "일본어",
        "zh": "중국어",
    }.get(transcript.language, transcript.language)
    metrics = (
        f"선택한 강의 {format_duration(clip_duration)} · {len(segments):,}문장 · 단어 {word_count:,}개\n"
        f"찾은 언어 {language_name} ({probability}) · 말소리가 있는 시간 {speech_ratio:.0%}"
    )
    return TranscriptReview(
        text_path=transcript.text_path.resolve(),
        preview_text=text,
        status=status,
        headline=headline,
        metrics=metrics,
        warnings=tuple(warnings),
    )
