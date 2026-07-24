from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import Callable


class WaveformCancelled(RuntimeError):
    pass


class AutoDetectionCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioEditRequest:
    audio_path: Path
    duration: float
    peaks: tuple[float, ...]
    range_start: float
    range_end: float


def extract_waveform(
    audio_path: Path,
    fallback_duration: float,
    bins: int = 12_000,
    cancel_event: Event | None = None,
    progress: Callable[[float], None] | None = None,
) -> tuple[float, tuple[float, ...]]:
    """Decode lightweight frame-level RMS values for a responsive long-audio overview."""
    import av
    import numpy as np

    with av.open(str(audio_path)) as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise RuntimeError("다운로드한 파일에서 오디오 스트림을 찾지 못했습니다.")
        duration = float(fallback_duration or 0)
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = float(container.duration / av.time_base)
        duration = max(duration, 1.0)
        peak_values = np.zeros(max(600, bins), dtype=np.float32)
        elapsed = 0.0
        last_sampled_index = -1
        for frame in container.decode(stream):
            if cancel_event and cancel_event.is_set():
                raise WaveformCancelled("사용자가 오디오 편집 준비를 취소했습니다.")
            position = float(frame.time) if frame.time is not None else elapsed
            index = min(len(peak_values) - 1, max(0, int(position / duration * len(peak_values))))
            if index != last_sampled_index:
                raw_values = frame.to_ndarray()
                values = raw_values.astype(np.float64, copy=False)
                level = float(np.sqrt(np.mean(np.square(values))))
                if np.issubdtype(raw_values.dtype, np.integer):
                    level /= float(np.iinfo(raw_values.dtype).max)
                peak_values[index] = max(peak_values[index], level)
                last_sampled_index = index
                if progress and index % 200 == 0:
                    progress(min(1.0, position / duration))
            sample_rate = float(frame.sample_rate or stream.codec_context.sample_rate or 48_000)
            elapsed += float(frame.samples) / sample_rate

    nonzero = peak_values[peak_values > 0]
    if nonzero.size:
        ceiling = float(np.percentile(nonzero, 98)) or 1.0
        peak_values = np.clip(peak_values / ceiling, 0.0, 1.0)
    if progress:
        progress(1.0)
    return duration, tuple(float(item) for item in peak_values)


def find_long_non_speech_ranges(
    speech_ranges: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    range_start: float,
    range_end: float,
    min_gap_seconds: float = 45.0,
    safety_padding_seconds: float = 2.0,
) -> tuple[tuple[float, float], ...]:
    """Invert detected speech ranges into conservative long exclusion candidates."""
    if range_end <= range_start:
        raise ValueError("자동 인식 범위가 올바르지 않습니다.")
    merged_speech = normalize_exclusions(speech_ranges, range_start, range_end)
    candidates: list[tuple[float, float]] = []
    cursor = range_start
    for speech_start, speech_end in merged_speech:
        start = cursor if cursor <= range_start else cursor + safety_padding_seconds
        end = speech_start - safety_padding_seconds
        if end - start >= min_gap_seconds:
            candidates.append((start, end))
        cursor = max(cursor, speech_end)
    trailing_start = (
        cursor if cursor <= range_start else cursor + safety_padding_seconds
    )
    if range_end - trailing_start >= min_gap_seconds:
        candidates.append((trailing_start, range_end))
    return normalize_exclusions(candidates, range_start, range_end)


def refine_edge_exclusion_ranges(
    speech_ranges: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    candidates: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    range_start: float,
    range_end: float,
    edge_min_gap_seconds: float = 5.0,
    safety_padding_seconds: float = 2.0,
    edge_extension_seconds: float = 120.0,
) -> tuple[tuple[float, float], ...]:
    """Include short intro/outro tails and absorb brief VAD music false positives."""
    if range_end <= range_start:
        raise ValueError("자동 인식 범위가 올바르지 않습니다.")
    speech = normalize_exclusions(speech_ranges, range_start, range_end)
    refined = list(normalize_exclusions(candidates, range_start, range_end))

    if speech:
        leading_end = speech[0][0] - safety_padding_seconds
        if leading_end - range_start >= edge_min_gap_seconds:
            refined.append((range_start, leading_end))
        trailing_start = speech[-1][1] + safety_padding_seconds
        if range_end - trailing_start >= edge_min_gap_seconds:
            refined.append((trailing_start, range_end))

    refined = list(normalize_exclusions(refined, range_start, range_end))
    # Music can briefly look speech-like to VAD. Extend every candidate touching
    # the first/last edge window so the intervening false-positive islands merge.
    for index, (start, end) in enumerate(refined):
        if start - range_start <= edge_extension_seconds:
            refined[index] = (range_start, end)
    for index, (start, end) in enumerate(refined):
        if range_end - end <= edge_extension_seconds:
            refined[index] = (start, range_end)
    return normalize_exclusions(refined, range_start, range_end)


def detect_long_non_speech(
    audio_path: Path,
    range_start: float,
    range_end: float,
    min_gap_seconds: float = 45.0,
    cancel_event: Event | None = None,
    progress: Callable[[float, str], None] | None = None,
    chunk_seconds: int = 900,
) -> tuple[tuple[float, float], ...]:
    """Detect long music/silence candidates with bundled Silero VAD in bounded memory."""
    import av
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    sample_rate = 16_000
    chunk_samples = max(sample_rate * 60, sample_rate * int(chunk_seconds))
    speech_ranges: list[tuple[float, float]] = []
    processed_samples = 0
    pending_parts: list[object] = []
    pending_samples = 0
    options = VadOptions(
        threshold=0.5,
        min_speech_duration_ms=1000,
        min_silence_duration_ms=800,
        speech_pad_ms=1000,
    )

    def check_cancelled() -> None:
        if cancel_event and cancel_event.is_set():
            raise AutoDetectionCancelled("자동 인식을 중지했습니다.")

    def analyze(samples) -> None:
        nonlocal processed_samples
        check_cancelled()
        if samples.size == 0:
            return
        offset = processed_samples / sample_rate
        for item in get_speech_timestamps(
            samples.astype(np.float32, copy=False),
            options,
            sampling_rate=sample_rate,
        ):
            speech_ranges.append(
                (
                    offset + float(item["start"]) / sample_rate,
                    offset + float(item["end"]) / sample_rate,
                )
            )
        processed_samples += int(samples.size)
        if progress:
            ratio = min(0.99, processed_samples / max(1.0, range_end * sample_rate))
            progress(ratio, f"음성 활동 분석 중 · {ratio:.0%}")

    def append_frame(frame) -> None:
        nonlocal pending_parts, pending_samples
        array = frame.to_ndarray().reshape(-1)
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
        else:
            array = array.astype(np.float32, copy=False)
        pending_parts.append(array)
        pending_samples += int(array.size)
        if pending_samples < chunk_samples:
            return
        combined = np.concatenate(pending_parts)
        offset = 0
        while combined.size - offset >= chunk_samples:
            analyze(combined[offset : offset + chunk_samples])
            offset += chunk_samples
        remainder = combined[offset:]
        pending_parts = [remainder] if remainder.size else []
        pending_samples = int(remainder.size)

    with av.open(str(audio_path)) as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise RuntimeError("자동 인식할 오디오 스트림을 찾지 못했습니다.")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            check_cancelled()
            converted = resampler.resample(frame)
            for converted_frame in converted if isinstance(converted, list) else [converted]:
                if converted_frame is not None:
                    append_frame(converted_frame)
        flushed = resampler.resample(None)
        for flushed_frame in flushed if isinstance(flushed, list) else [flushed]:
            if flushed_frame is not None:
                append_frame(flushed_frame)

    if pending_parts:
        analyze(np.concatenate(pending_parts))
    check_cancelled()
    candidates = find_long_non_speech_ranges(
        speech_ranges,
        range_start,
        range_end,
        min_gap_seconds=min_gap_seconds,
        safety_padding_seconds=2.0,
    )
    candidates = refine_edge_exclusion_ranges(
        speech_ranges,
        candidates,
        range_start,
        range_end,
        edge_min_gap_seconds=5.0,
        safety_padding_seconds=2.0,
        edge_extension_seconds=120.0,
    )
    if progress:
        progress(1.0, f"자동 인식 완료 · 후보 {len(candidates)}개")
    return candidates


def normalize_exclusions(
    exclusions: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    range_start: float,
    range_end: float,
) -> tuple[tuple[float, float], ...]:
    clipped: list[tuple[float, float]] = []
    for raw_start, raw_end in exclusions:
        start = max(range_start, min(float(raw_start), range_end))
        end = max(range_start, min(float(raw_end), range_end))
        if end - start >= 0.25:
            clipped.append((start, end))
    clipped.sort()
    merged: list[list[float]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1] + 0.25:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def build_included_ranges(
    range_start: float,
    range_end: float,
    exclusions: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    if range_end <= range_start:
        raise ValueError("오디오 편집 범위가 올바르지 않습니다.")
    normalized = normalize_exclusions(exclusions, range_start, range_end)
    included: list[tuple[float, float]] = []
    cursor = range_start
    for start, end in normalized:
        if start - cursor >= 0.25:
            included.append((cursor, start))
        cursor = max(cursor, end)
    if range_end - cursor >= 0.25:
        included.append((cursor, range_end))
    if not included:
        raise ValueError("전체 구간이 제외되었습니다. 강의 음성이 있는 부분을 하나 이상 남겨주세요.")
    return tuple(included)


def cuts_to_segments(
    cuts: tuple[float, ...] | list[float],
    range_start: float,
    range_end: float,
) -> tuple[tuple[float, float], ...]:
    points = sorted(
        {float(value) for value in cuts if range_start + 0.1 < value < range_end - 0.1}
    )
    boundaries = [range_start, *points, range_end]
    return tuple(
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
    )


def render_edited_audio(
    source_path: Path,
    output_path: Path,
    included_ranges: tuple[tuple[float, float], ...],
    source_duration: float,
    cancel_event: Event | None = None,
    progress: Callable[[float], None] | None = None,
) -> Path:
    """Render kept lecture clips into one contiguous 16 kHz mono lossless FLAC."""
    import av

    if not included_ranges:
        raise ValueError("편집본에 남길 강의 구간이 없습니다.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.stem + ".partial.flac")
    if partial_path.exists():
        partial_path.unlink()
    try:
        with av.open(str(source_path)) as source, av.open(str(partial_path), mode="w") as target:
            input_stream = next(
                (item for item in source.streams if item.type == "audio"), None
            )
            if input_stream is None:
                raise RuntimeError("원본 파일에서 오디오 스트림을 찾지 못했습니다.")
            output_stream = target.add_stream("flac", rate=16_000)
            output_stream.layout = "mono"
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
            range_index = 0
            next_pts = 0
            last_progress_second = -1

            def encode_frame(frame) -> None:
                nonlocal next_pts
                frame.pts = next_pts
                frame.time_base = Fraction(1, 16_000)
                next_pts += frame.samples
                for packet in output_stream.encode(frame):
                    target.mux(packet)

            for frame in source.decode(input_stream):
                if cancel_event and cancel_event.is_set():
                    raise WaveformCancelled("사용자가 편집본 저장을 취소했습니다.")
                sample_rate = float(
                    frame.sample_rate or input_stream.codec_context.sample_rate or 48_000
                )
                frame_start = float(frame.time or 0.0)
                frame_end = frame_start + float(frame.samples) / sample_rate
                midpoint = (frame_start + frame_end) / 2
                while (
                    range_index < len(included_ranges)
                    and midpoint >= included_ranges[range_index][1]
                ):
                    range_index += 1
                if range_index >= len(included_ranges):
                    break
                start, end = included_ranges[range_index]
                if start <= midpoint < end:
                    converted = resampler.resample(frame)
                    frames = converted if isinstance(converted, list) else [converted]
                    for converted_frame in frames:
                        if converted_frame is not None:
                            encode_frame(converted_frame)
                progress_second = int(max(0.0, frame_start))
                if progress and progress_second != last_progress_second:
                    last_progress_second = progress_second
                    progress(min(1.0, frame_start / max(1.0, source_duration)))

            flushed = resampler.resample(None)
            flushed_frames = flushed if isinstance(flushed, list) else [flushed]
            for flushed_frame in flushed_frames:
                if flushed_frame is not None:
                    encode_frame(flushed_frame)
            for packet in output_stream.encode(None):
                target.mux(packet)
        partial_path.replace(output_path)
        if progress:
            progress(1.0)
        return output_path.resolve()
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise


def map_edited_time_to_original(
    seconds: float, included_ranges: tuple[tuple[float, float], ...]
) -> float:
    remaining = max(0.0, seconds)
    for start, end in included_ranges:
        duration = end - start
        if remaining <= duration:
            return start + remaining
        remaining -= duration
    return included_ranges[-1][1]
