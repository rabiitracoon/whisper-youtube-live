"""Optional Qwen3-ASR engine (Alibaba, Apache 2.0) alongside Whisper.

macOS (Apple Silicon) runs it through ``mlx-qwen3-asr`` and Windows (CUDA)
through the official ``qwen-asr`` package. Word timestamps come from
Qwen3-ForcedAligner-0.6B, which also supports Korean, and are regrouped into
sentence-sized segments so the rest of the app can treat the output like
Whisper's.
"""

from __future__ import annotations

import platform
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Event

import numpy as np

SAMPLE_RATE = 16_000
QWEN_MODELS = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
}
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE_NAMES = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
LANGUAGE_CODES = {name.casefold(): code for code, name in LANGUAGE_NAMES.items()}
# Long files are fed in pieces so progress and cancellation stay responsive.
PIECE_SECONDS = 120.0
SENTENCE_END = re.compile(r"[.?!。？！]")

_CUDA_CACHE: dict[str, object] = {}


class QwenUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AlignedWord:
    text: str
    start: float
    end: float


def is_qwen_model(model_name: str) -> bool:
    return model_name in QWEN_MODELS


def language_name(code: str) -> str | None:
    return None if code in {"", "auto"} else LANGUAGE_NAMES.get(code, code)


def language_code(name: str) -> str:
    return LANGUAGE_CODES.get(str(name).strip().casefold(), str(name))


def context_from_terms(prompt_terms: Sequence[str], extra: str = "") -> str:
    terms = ", ".join(term for term in (" ".join(str(item).split()) for item in prompt_terms) if term)
    parts = [part for part in (terms and f"강의 전문용어: {terms}", extra.strip()) if part]
    return "\n".join(parts)


def group_words(
    text: str,
    words: Sequence[AlignedWord],
    max_seconds: float = 12.0,
    pause_seconds: float = 0.8,
) -> list[tuple[float, float, str]]:
    """Group aligned words into sentence segments while keeping the original spacing.

    The aligner strips spaces and punctuation (Korean is tokenized by soynlp), so
    each word is located in ``text`` and segment text is sliced from it.
    """
    if not words:
        stripped = text.strip()
        return [(0.0, 0.0, stripped)] if stripped else []
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for word in words:
        index = text.find(word.text, cursor) if word.text else -1
        if index < 0:
            spans.append(None)
            continue
        spans.append((index, index + len(word.text)))
        cursor = index + len(word.text)

    segments: list[tuple[float, float, str]] = []
    group_start = 0
    char_start = 0
    for position, word in enumerate(words):
        is_last = position == len(words) - 1
        next_word = None if is_last else words[position + 1]
        span = spans[position]
        next_span = None if is_last else spans[position + 1]
        between = ""
        if span is not None:
            between = text[span[1]: next_span[0] if next_span else len(text)]
        should_break = (
            is_last
            or bool(SENTENCE_END.search(between))
            or (next_word is not None and next_word.start - word.end >= pause_seconds)
            or word.end - words[group_start].start >= max_seconds
        )
        if not should_break:
            continue
        if span is not None and next_span is not None:
            char_end = next_span[0]
        elif is_last:
            char_end = len(text)
        else:
            char_end = span[1] if span is not None else char_start
        segment_text = text[char_start:char_end].strip()
        if not segment_text or spans[group_start] is None:
            segment_text = " ".join(item.text for item in words[group_start: position + 1])
        segments.append((words[group_start].start, word.end, segment_text))
        group_start = position + 1
        char_start = char_end
    return [segment for segment in segments if segment[2]]


def quietest_cut(samples: np.ndarray, search_seconds: float) -> int:
    frame = SAMPLE_RATE // 10
    count = len(samples) // frame
    if count < 3:
        return len(samples)
    frames = samples[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    search = min(count - 1, max(1, int(search_seconds * 10)))
    index = count - search + int(np.argmin(rms[-search:]))
    return max(1, index) * frame


def split_pieces(samples: np.ndarray, piece_seconds: float = PIECE_SECONDS) -> list[tuple[int, int]]:
    """Split audio into ≤ piece_seconds windows, cutting at the quietest nearby point."""
    limit = int(piece_seconds * SAMPLE_RATE)
    pieces: list[tuple[int, int]] = []
    start = 0
    while start < len(samples):
        end = min(len(samples), start + limit)
        if end < len(samples):
            end = start + quietest_cut(samples[start:end], search_seconds=10.0)
        pieces.append((start, end))
        start = end
    return pieces


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _missing_package(exc: ImportError) -> QwenUnavailable:
    installer = "install.bat" if platform.system() == "Windows" else "install.command"
    return QwenUnavailable(
        f"Qwen3-ASR 엔진 패키지가 설치되지 않았습니다({exc.name or exc}). {installer}을 다시 "
        "실행하거나 음성 인식 모델을 Whisper로 바꿔주세요."
    )


_MLX_ALIGNER: list[object] = []


def _transcribe_mlx(
    samples: np.ndarray, model_id: str, language: str | None, context: str, timestamps: bool
) -> tuple[str, str, list[AlignedWord]]:
    try:
        import mlx_qwen3_asr
    except ImportError as exc:
        raise _missing_package(exc) from exc
    result = mlx_qwen3_asr.transcribe(
        samples,
        model=model_id,
        context=context,
        language=language,
        return_timestamps=timestamps,
        forced_aligner=_mlx_aligner(mlx_qwen3_asr) if timestamps else None,
    )
    words = [
        AlignedWord(str(item["text"]), float(item["start"]), float(item["end"]))
        for item in (result.segments or [])
    ]
    return str(result.text), str(result.language), words


def _mlx_aligner(module: object) -> object:
    # One aligner instance keeps its weights loaded between pieces.
    if not _MLX_ALIGNER:
        _MLX_ALIGNER.append(module.ForcedAligner(ALIGNER_MODEL))
    return _MLX_ALIGNER[0]


def _get_cuda_model(model_id: str, timestamps: bool, log: Callable[[str], None] | None):
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise _missing_package(exc) from exc
    model = _CUDA_CACHE.get(model_id)
    if model is None:
        if log:
            log(f"Qwen3-ASR 모델({model_id})을 불러옵니다. 첫 실행은 모델 다운로드가 필요합니다.")
        model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=4,
            max_new_tokens=1024,
        )
        _CUDA_CACHE[model_id] = model
    if timestamps and getattr(model, "forced_aligner", None) is None:
        from qwen_asr import Qwen3ForcedAligner

        if log:
            log(f"타임스탬프 정렬 모델({ALIGNER_MODEL})을 불러옵니다.")
        model.forced_aligner = Qwen3ForcedAligner.from_pretrained(
            ALIGNER_MODEL, dtype=torch.bfloat16, device_map="cuda:0"
        )
    return model


def _transcribe_cuda(
    samples: np.ndarray, model_id: str, language: str | None, context: str,
    timestamps: bool, log: Callable[[str], None] | None,
) -> tuple[str, str, list[AlignedWord]]:
    model = _get_cuda_model(model_id, timestamps, log)
    result = model.transcribe(
        audio=(samples, SAMPLE_RATE),
        context=context,
        language=language,
        return_time_stamps=timestamps,
    )[0]
    words = [
        AlignedWord(str(item.text), float(item.start_time), float(item.end_time))
        for item in (result.time_stamps or [])
    ]
    return str(result.text), str(result.language), words


def transcribe_qwen(
    samples: np.ndarray,
    model_name: str,
    language: str = "auto",
    context: str = "",
    timestamps: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[list[tuple[float, float, str]], str]:
    """Transcribe one in-memory buffer; returns ((start, end, text) segments, language)."""
    model_id = QWEN_MODELS[model_name]
    samples = np.asarray(samples, dtype=np.float32)
    forced = language_name(language)
    if platform.system() == "Windows":
        text, detected, words = _transcribe_cuda(samples, model_id, forced, context, timestamps, log)
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        text, detected, words = _transcribe_mlx(samples, model_id, forced, context, timestamps)
    else:
        raise QwenUnavailable("Qwen3-ASR 엔진은 Windows CUDA 또는 Apple Silicon Mac에서만 사용할 수 있습니다.")
    text = text.strip()
    detected = language_code(detected)
    if not text:
        return [], detected
    if not timestamps:
        return [(0.0, len(samples) / SAMPLE_RATE, text)], detected
    segments = group_words(text, words)
    if not segments:
        segments = [(0.0, len(samples) / SAMPLE_RATE, text)]
    return segments, detected


def transcribe_qwen_long(
    samples: np.ndarray,
    model_name: str,
    language: str,
    context: str,
    progress: Callable[[float, float], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel_event: Event | None = None,
    is_cancelled_error: type[Exception] = RuntimeError,
) -> tuple[list[tuple[float, float, str]], str]:
    """Transcribe a whole lecture piece by piece with timestamps on the full timeline."""
    pieces = split_pieces(samples)
    segments: list[tuple[float, float, str]] = []
    languages: list[str] = []
    for start, end in pieces:
        if cancel_event and cancel_event.is_set():
            raise is_cancelled_error("사용자가 전사를 취소했습니다.")
        piece = samples[start:end]
        offset = start / SAMPLE_RATE
        if piece.size and float(np.sqrt(np.mean(piece * piece))) > 10 ** (-50 / 20):
            piece_segments, detected = transcribe_qwen(
                piece, model_name, language, context, timestamps=True, log=log
            )
            languages.append(detected)
            segments.extend((offset + s, offset + e, text) for s, e, text in piece_segments)
        if progress:
            progress(end / SAMPLE_RATE, len(samples) / SAMPLE_RATE)
    detected_language = max(set(languages), key=languages.count) if languages else language
    return segments, detected_language
