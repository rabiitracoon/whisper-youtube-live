from __future__ import annotations

import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from .audio_edit import map_edited_time_to_original
from .youtube import VideoInfo, format_duration


ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]
_DLL_HANDLES: list[object] = []
_MODEL_CACHE: dict[tuple[str, str], object] = {}


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


def _add_dll_directory(path: Path) -> None:
    if not path.exists():
        return
    path_text = str(path.resolve())
    current = os.environ.get("PATH", "")
    if path_text.lower() not in current.lower().split(os.pathsep):
        os.environ["PATH"] = path_text + os.pathsep + current
    if hasattr(os, "add_dll_directory"):
        try:
            _DLL_HANDLES.append(os.add_dll_directory(path_text))
        except OSError:
            pass


def prepare_cuda_runtime() -> None:
    """Expose CUDA 12/cuDNN DLLs bundled by the GPU PyTorch wheel to CTranslate2."""
    roots = {Path(item) for item in site.getsitepackages() if item}
    roots.add(Path(sys.prefix) / "Lib" / "site-packages")
    for root in roots:
        _add_dll_directory(root / "torch" / "lib")
        nvidia_root = root / "nvidia"
        if nvidia_root.exists():
            for bin_dir in nvidia_root.glob("*/bin"):
                _add_dll_directory(bin_dir)
    for env_name in ("CUDA_PATH", "CUDA_PATH_V12_8", "CUDA_PATH_V12_6"):
        value = os.environ.get(env_name)
        if value:
            _add_dll_directory(Path(value) / "bin")
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch가 NVIDIA GPU를 인식하지 못했습니다.")
        # Loading one CUDA tensor eagerly loads the CUDA/cuDNN dependencies used by CTranslate2.
        torch.empty(1, device="cuda")
    except ImportError as exc:
        raise RuntimeError("GPU용 PyTorch가 설치되지 않았습니다. install.bat을 다시 실행해주세요.") from exc


def verify_cuda() -> str:
    prepare_cuda_runtime()
    import ctranslate2
    import torch

    count = ctranslate2.get_cuda_device_count()
    if count < 1:
        raise RuntimeError("CTranslate2가 사용할 수 있는 NVIDIA CUDA 장치를 찾지 못했습니다.")
    return str(torch.cuda.get_device_name(0))


def _get_model(model_name: str, compute_type: str, log: LogCallback | None):
    key = (model_name, compute_type)
    if key in _MODEL_CACHE:
        if log:
            log(f"메모리에 로드된 Whisper {model_name} 모델을 재사용합니다.")
        return _MODEL_CACHE[key]
    prepare_cuda_runtime()
    from faster_whisper import WhisperModel

    if log:
        log(f"Whisper {model_name} 모델을 불러옵니다. 첫 실행은 약 3GB 다운로드가 필요합니다.")
    model = WhisperModel(
        model_name,
        device="cuda",
        compute_type=compute_type,
        num_workers=1,
    )
    _MODEL_CACHE[key] = model
    return model


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
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> TranscriptResult:
    model = _get_model(model_name, compute_type, log)
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
    effective_vad = vad_filter and not has_explicit_clip
    if log:
        if timestamp_map:
            range_text = "저장된 연속 편집본 전체"
        else:
            end_text = format_duration(effective_end) if effective_end > 0 else "영상 끝"
            range_text = f"{format_duration(clip_start)}~{end_text}"
        if not timestamp_map and len(normalized_ranges) > 1:
            range_text = f"{len(normalized_ranges)}개 강의 구간"
        log(f"정확도 우선 설정(beam={beam_size}, FP16)으로 GPU 전사를 시작합니다. 구간: {range_text}")
        if timestamp_map:
            log("VAD와 반복 억제를 적용하며, 결과 시간은 원본 영상 위치로 다시 연결합니다.")
        if vad_filter and has_explicit_clip:
            log("선택 구간 전사에서는 Whisper 구간 지정이 VAD보다 우선 적용됩니다.")
    raw_segments, info = model.transcribe(
        str(audio_path),
        task="transcribe",
        language=None if language == "auto" else language,
        beam_size=beam_size,
        best_of=max(beam_size, 5),
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=effective_vad,
        vad_parameters={"min_silence_duration_ms": 500} if effective_vad else None,
        clip_timestamps=clip_timestamps,
        word_timestamps=False,
    )
    full_duration = float(getattr(info, "duration", 0) or video.duration or 1)
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
        text = str(raw.text).strip()
        if text:
            raw_start = float(raw.start)
            raw_end = float(raw.end)
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
                max(0.0, min(float(raw.end), end) - start)
                for start, end in normalized_ranges
            )
            progress(min(1.0, completed / selected_duration), f"전사 중 · {format_duration(raw.end)}")

    if not segments:
        raise RuntimeError("음성을 감지하지 못해 전사 결과가 비어 있습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "transcript.txt"
    markdown_path = output_dir / "transcript.md"
    srt_path = output_dir / "transcript.srt"
    text_path.write_text("\n".join(item.text for item in segments) + "\n", encoding="utf-8")
    markdown_path.write_text(_to_markdown(video, segments), encoding="utf-8")
    srt_path.write_text(_to_srt(segments), encoding="utf-8")
    detected_language = str(getattr(info, "language", language))
    probability = float(getattr(info, "language_probability", 0.0) or 0.0)
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


def _optional_float(value: object, attribute: str) -> float | None:
    raw = getattr(value, attribute, None)
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
