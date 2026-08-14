from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from .audio_edit import map_edited_time_to_original
from .paths import PROJECT_ROOT
from .youtube import VideoInfo, format_duration


ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]
MLX_MODEL_ALIASES = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
}
LOCAL_MLX_MODELS = {
    "large-v3": PROJECT_ROOT / "models" / "whisper-large-v3-mlx",
    "large-v2": PROJECT_ROOT / "models" / "whisper-large-v2-mlx",
    "medium": PROJECT_ROOT / "models" / "whisper-medium-mlx",
}


class TranscriptionCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_probability: float | None = None
    compression_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    text_path: Path
    markdown_path: Path
    srt_path: Path
    language: str
    language_probability: float
    segments: tuple[TranscriptSegment, ...]
    clip_start: float
    clip_end: float
    clip_ranges: tuple[tuple[float, float], ...]


def resolve_mlx_model(model_name: str) -> str:
    """Accept the old UI aliases as well as an MLX Hugging Face repo/local path."""
    local_path = LOCAL_MLX_MODELS.get(model_name)
    if local_path and (local_path / "config.json").is_file() and (
        (local_path / "weights.npz").is_file()
        or (local_path / "weights.safetensors").is_file()
    ):
        return str(local_path)
    return MLX_MODEL_ALIASES.get(model_name, model_name)


def verify_mlx() -> str:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX 전사는 Apple Silicon(arm64) Mac에서만 사용할 수 있습니다.")
    try:
        import mlx.core as mx

        # Force Metal initialization here so failures are reported before downloading audio.
        mx.eval(mx.zeros((1,)))
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "MLX가 Apple Metal GPU를 사용할 수 없습니다. install.command를 다시 실행하고 "
            "일반 macOS 터미널에서 앱을 실행해주세요."
        ) from exc
    return "Apple Silicon · MLX Metal"


def _audio_duration(path: Path) -> float:
    """Read the actual edited file duration without decoding it a second time."""
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is not None and stream.duration is not None:
                return float(stream.duration * stream.time_base)
    except (ImportError, OSError, StopIteration, ValueError):
        pass
    return 0.0


def transcribe_audio(
    audio_path: Path,
    output_dir: Path,
    video: VideoInfo,
    model_name: str = "large-v3",
    language: str = "auto",
    compute_type: str = "float16",
    beam_size: int = 5,
    vad_filter: bool = True,
    clip_start: float = 0.0,
    clip_end: float | None = None,
    clip_ranges: tuple[tuple[float, float], ...] | None = None,
    original_ranges: tuple[tuple[float, float], ...] | None = None,
    prompt_terms: tuple[str, ...] = (),
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> TranscriptResult:
    model_repo = resolve_mlx_model(model_name)
    if cancel_event and cancel_event.is_set():
        raise TranscriptionCancelled("사용자가 전사를 취소했습니다.")
    requested_end = float(clip_end) if clip_end is not None else None
    effective_end = requested_end or float(video.duration or 0)
    if effective_end > 0 and effective_end <= clip_start:
        raise ValueError("전사 종료 시간은 시작 시간보다 뒤여야 합니다.")
    timestamp_map = tuple(original_ranges or ())
    normalized_ranges = tuple(clip_ranges or ())
    for start, end in normalized_ranges:
        if start < 0 or end <= start:
            raise ValueError("파형 편집에서 전달된 전사 구간이 올바르지 않습니다.")
    has_explicit_clip = (
        not timestamp_map
        and (bool(normalized_ranges) or clip_start > 0 or clip_end is not None)
    )
    clip_timestamps: str | list[float] = "0"
    if normalized_ranges:
        clip_timestamps = [value for pair in normalized_ranges for value in pair]
        clip_start = normalized_ranges[0][0]
        effective_end = normalized_ranges[-1][1]
    elif has_explicit_clip:
        clip_timestamps = [clip_start] if clip_end is None else [clip_start, clip_end]
    if log:
        if timestamp_map:
            range_text = "저장된 연속 편집본 전체"
        else:
            end_text = format_duration(effective_end) if effective_end > 0 else "영상 끝"
            range_text = f"{format_duration(clip_start)}~{end_text}"
        if not timestamp_map and len(normalized_ranges) > 1:
            range_text = f"{len(normalized_ranges)}개 강의 구간"
        log(f"정확도 우선 FP16 설정으로 MLX Metal 전사를 시작합니다. 구간: {range_text}")
        log(f"MLX 모델: {model_repo} (첫 실행에는 모델 다운로드가 필요합니다.)")
        if timestamp_map:
            log("반복 억제를 적용하며, 결과 시간은 원본 영상 위치로 다시 연결합니다.")
        if vad_filter:
            log("쉬는 시간 편집 단계의 Silero VAD 결과를 사용하고 Whisper 반복 억제를 적용합니다.")
        if prompt_terms:
            log(f"전문용어 {len(prompt_terms)}개를 Whisper 문맥에 자동 적용합니다.")
    import mlx_whisper

    initial_prompt = _build_initial_prompt(prompt_terms)

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        task="transcribe",
        language=None if language == "auto" else language,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        clip_timestamps=clip_timestamps,
        word_timestamps=False,
        fp16=compute_type != "float32",
        verbose=None,
    )
    raw_segments = result.get("segments", [])
    full_duration = _audio_duration(audio_path) or max(
        (float(item.get("end", 0)) for item in raw_segments), default=float(video.duration or 1)
    )
    if timestamp_map:
        clip_start = timestamp_map[0][0]
        effective_end = timestamp_map[-1][1]
        normalized_ranges = ((0.0, full_duration),)
    effective_end = min(effective_end or full_duration, full_duration)
    if timestamp_map:
        result_ranges = timestamp_map
    elif not normalized_ranges:
        normalized_ranges = ((clip_start, effective_end),)
        result_ranges = normalized_ranges
    else:
        normalized_ranges = tuple(
            (start, min(end, full_duration))
            for start, end in normalized_ranges
            if start < full_duration and min(end, full_duration) > start
        )
        result_ranges = normalized_ranges
    selected_duration = max(1.0, sum(end - start for start, end in normalized_ranges))
    segments: list[TranscriptSegment] = []
    for raw in raw_segments:
        if cancel_event and cancel_event.is_set():
            raise TranscriptionCancelled("사용자가 전사를 취소했습니다.")
        text = str(raw.get("text", "")).strip()
        if text:
            raw_start = float(raw.get("start", 0))
            raw_end = float(raw.get("end", raw_start))
            segment_start = raw_start
            segment_end = raw_end
            if timestamp_map:
                segment_start = map_edited_time_to_original(raw_start, timestamp_map)
                segment_end = map_edited_time_to_original(raw_end, timestamp_map)
            segments.append(
                TranscriptSegment(
                    start=segment_start,
                    end=segment_end,
                    text=text,
                    avg_logprob=_optional_float(raw, "avg_logprob"),
                    no_speech_probability=_optional_float(raw, "no_speech_prob"),
                    compression_ratio=_optional_float(raw, "compression_ratio"),
                )
            )
        if progress:
            completed = sum(
                max(0.0, min(float(raw.get("end", 0)), end) - start)
                for start, end in normalized_ranges
            )
            progress(min(1.0, completed / selected_duration), f"전사 중 · {format_duration(float(raw.get('end', 0)))}")

    if not segments:
        raise RuntimeError("음성을 감지하지 못해 전사 결과가 비어 있습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "transcript.txt"
    markdown_path = output_dir / "transcript.md"
    srt_path = output_dir / "transcript.srt"
    text_path.write_text("\n".join(item.text for item in segments) + "\n", encoding="utf-8")
    markdown_path.write_text(_to_markdown(video, segments), encoding="utf-8")
    srt_path.write_text(_to_srt(segments), encoding="utf-8")
    detected_language = str(result.get("language", language))
    # mlx-whisper returns the selected language but not its probability. Avoid a
    # false low-confidence warning in the downstream review.
    probability = 1.0
    if progress:
        progress(1.0, "전사 파일 저장 완료")
    return TranscriptResult(
        text_path=text_path,
        markdown_path=markdown_path,
        srt_path=srt_path,
        language=detected_language,
        language_probability=probability,
        segments=tuple(segments),
        clip_start=clip_start,
        clip_end=result_ranges[-1][1],
        clip_ranges=result_ranges,
    )


def _build_initial_prompt(prompt_terms: tuple[str, ...], limit: int = 600) -> str | None:
    """Build a compact Whisper prompt without overflowing its short text context."""
    selected: list[str] = []
    seen: set[str] = set()
    current_length = 0
    for raw_term in prompt_terms:
        term = " ".join(str(raw_term).split()).strip(" ,")
        key = term.casefold()
        if not term or key in seen:
            continue
        added = len(term) + (2 if selected else 0)
        if current_length + added > limit:
            break
        seen.add(key)
        selected.append(term)
        current_length += added
    if not selected:
        return None
    return "이 강의의 주요 전문용어: " + ", ".join(selected)


def _optional_float(value: object, attribute: str) -> float | None:
    raw = value.get(attribute) if isinstance(value, dict) else getattr(value, attribute, None)
    return float(raw) if raw is not None else None


def _to_markdown(video: VideoInfo, segments: list[TranscriptSegment]) -> str:
    lines = [
        f"# {video.title} — 전사문",
        "",
        f"- 채널: {video.channel}",
        f"- 원본: {video.webpage_url}",
        f"- 길이: {format_duration(video.duration)}",
        "",
        "## 타임스탬프 전사",
        "",
    ]
    for item in segments:
        seconds = int(item.start)
        link = f"https://www.youtube.com/watch?v={video.video_id}&t={seconds}s"
        lines.append(f"- [{format_duration(item.start)}]({link}) {item.text}")
    return "\n".join(lines) + "\n"


def _to_srt(segments: list[TranscriptSegment]) -> str:
    blocks = []
    for index, item in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{_srt_time(item.start)} --> {_srt_time(item.end)}\n{item.text}\n"
        )
    return "\n".join(blocks)


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
