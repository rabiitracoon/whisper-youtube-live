"""Real-time captions for a running lecture stream.

The audio is pulled straight from the network (YouTube live HLS via yt-dlp, or a
local file for rehearsal) and decoded by ffmpeg into 16 kHz mono PCM, so the
computer does not need speakers, a loopback device or "Stereo Mix".
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

import numpy as np

from .model_download import DownloadCancelled, DownloadProgress, prefetch_model
from .paths import safe_filename
from .transcription import TranscriptSegment, _build_initial_prompt, _srt_time, transcribe_samples
from .youtube import _base_options

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.1
LogCallback = Callable[[str], None]

# Phrases Whisper tends to invent on silence or music in Korean/English streams.
HALLUCINATION_PHRASES = (
    "시청해주셔서 감사합니다",
    "시청해 주셔서 감사합니다",
    "구독과 좋아요",
    "구독 좋아요",
    "좋아요와 구독",
    "다음 영상에서 만나요",
    "자막 제공",
    "자막by",
    "mbc 뉴스",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
)


class LiveCaptionCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LiveSource:
    title: str
    input: str
    is_live: bool
    headers: dict[str, str] = field(default_factory=dict)
    page_url: str = ""


@dataclass(frozen=True, slots=True)
class LiveCaption:
    start: float
    end: float
    text: str
    wall_time: datetime


@dataclass(frozen=True, slots=True)
class LiveCaptionOptions:
    model_name: str = "large-v3-turbo"
    language: str = "auto"
    compute_type: str = "float16"
    beam_size: int = 1
    prompt_terms: tuple[str, ...] = ()
    # Re-transcribe the unfinished sentence this often to show a draft line.
    partial_interval: float = 1.5
    # A sentence is closed after this much trailing quiet...
    min_chunk: float = 3.0
    trailing_silence: float = 0.5
    # ...or forcibly at the quietest point once it grows this long.
    max_chunk: float = 12.0
    # Above this much unread audio we skip draft lines to catch up.
    max_backlog: float = 6.0


# --------------------------------------------------------------------------- #
# Source resolution and decoding
# --------------------------------------------------------------------------- #


def resolve_live_source(
    source: str, cookie_browser: str = "none", log: LogCallback | None = None
) -> LiveSource:
    text = source.strip().strip('"')
    if not text:
        raise ValueError("실시간 자막을 만들 라이브 링크를 입력해주세요.")
    path = Path(text).expanduser()
    if not re.match(r"^https?://", text, re.IGNORECASE):
        if path.is_file():
            return LiveSource(title=path.stem, input=str(path.resolve()), is_live=False)
        raise ValueError("http(s) 라이브 링크 또는 연습용 영상·음성 파일 경로를 입력해주세요.")
    if re.search(r"\.m3u8(\?|$)", text, re.IGNORECASE):
        return LiveSource(title="라이브 스트림", input=text, is_live=True, page_url=text)

    from yt_dlp import YoutubeDL

    options = _base_options(cookie_browser, log)
    options.update({"skip_download": True, "format": "bestaudio/worst"})
    with YoutubeDL(options) as ydl:
        data = ydl.extract_info(text, download=False)
    if not data or data.get("_type") == "playlist":
        raise ValueError("재생목록이 아닌 단일 라이브 방송 링크를 입력해주세요.")
    status = data.get("live_status")
    if status == "is_upcoming":
        raise ValueError("아직 시작하지 않은 방송입니다. 방송이 시작된 뒤 다시 눌러주세요.")
    media_url = data.get("url")
    if not media_url:
        requested = data.get("requested_formats") or []
        media_url = next((item.get("url") for item in requested if item.get("url")), None)
    if not media_url:
        raise RuntimeError("방송의 오디오 주소를 찾지 못했습니다. yt-dlp를 업데이트한 뒤 다시 시도하세요.")
    headers = {
        str(key): str(value)
        for key, value in (data.get("http_headers") or {}).items()
        if value
    }
    return LiveSource(
        title=str(data.get("title") or "라이브 방송"),
        input=str(media_url),
        is_live=status == "is_live" or bool(data.get("is_live")),
        headers=headers,
        page_url=str(data.get("webpage_url") or text),
    )


def build_ffmpeg_command(ffmpeg: str, source: LiveSource) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if source.input.lower().startswith(("http://", "https://")):
        if source.headers:
            header_text = "".join(f"{key}: {value}\r\n" for key, value in source.headers.items())
            command += ["-headers", header_text]
        command += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    if not source.is_live:
        # Replay recordings at real-time speed so they behave like a broadcast.
        command.append("-re")
    command += [
        "-i", source.input,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ]
    return command


class PcmStream:
    """Run ffmpeg and drain its PCM output on a background thread.

    Draining continuously keeps ffmpeg from stalling (and the live playlist from
    expiring) while Whisper is busy with an earlier sentence.
    """

    def __init__(self, command: list[str]):
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._errors: deque[str] = deque(maxlen=12)
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._stderr.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        while True:
            chunk = self.process.stdout.read(SAMPLE_RATE // 5)
            if not chunk:
                break
            with self._lock:
                self._buffer.extend(chunk)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in iter(self.process.stderr.readline, b""):
            text = line.decode("utf-8", "replace").strip()
            if text:
                self._errors.append(text)

    def buffered_seconds(self) -> float:
        with self._lock:
            return len(self._buffer) / 2 / SAMPLE_RATE

    def take(self) -> np.ndarray:
        with self._lock:
            usable = len(self._buffer) - len(self._buffer) % 2
            data = bytes(self._buffer[:usable])
            del self._buffer[:usable]
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    @property
    def ended(self) -> bool:
        return not self._reader.is_alive() and self.buffered_seconds() == 0

    def error_text(self) -> str:
        return "\n".join(self._errors)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(3)
            except subprocess.TimeoutExpired:
                self.process.kill()


# --------------------------------------------------------------------------- #
# Sentence segmentation helpers (pure functions, unit tested)
# --------------------------------------------------------------------------- #


def frame_rms(samples: np.ndarray, frame_seconds: float = FRAME_SECONDS) -> np.ndarray:
    size = max(1, int(SAMPLE_RATE * frame_seconds))
    count = len(samples) // size
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: count * size].reshape(count, size)
    return np.sqrt(np.mean(frames * frames, axis=1))


def quiet_threshold(rms: np.ndarray) -> float:
    """Frames below this level count as a pause between sentences."""
    absolute_floor = 10 ** (-50 / 20)
    if rms.size == 0:
        return absolute_floor
    return max(absolute_floor, 0.2 * float(np.percentile(rms, 90)))


def is_silent(samples: np.ndarray) -> bool:
    if samples.size == 0:
        return True
    rms = frame_rms(samples)
    if rms.size == 0:
        return True
    # Speech must be audible for a meaningful share of the window.
    return float(np.percentile(rms, 90)) < 10 ** (-45 / 20)


def trailing_quiet_seconds(samples: np.ndarray) -> float:
    rms = frame_rms(samples)
    if rms.size == 0:
        return 0.0
    threshold = quiet_threshold(rms)
    count = 0
    for value in rms[::-1]:
        if value >= threshold:
            break
        count += 1
    return count * FRAME_SECONDS


def quietest_cut(samples: np.ndarray, search_seconds: float = 5.0) -> int:
    """Return a sample index near the end of ``samples`` where speech is quietest."""
    rms = frame_rms(samples)
    if rms.size < 3:
        return len(samples)
    search_frames = min(rms.size - 1, max(1, int(search_seconds / FRAME_SECONDS)))
    smoothed = np.convolve(rms, np.ones(3) / 3, mode="same")
    window = smoothed[-search_frames:]
    frame_index = rms.size - search_frames + int(np.argmin(window))
    return max(1, frame_index) * int(SAMPLE_RATE * FRAME_SECONDS)


def _normalize(text: str) -> str:
    return re.sub(r"[\s.,!?~·…'\"]+", "", text).casefold()


def keep_segment(segment: TranscriptSegment, previous_text: str = "") -> bool:
    text = _normalize(segment.text)
    if not text:
        return False
    if any(_normalize(phrase) in text and len(text) <= len(_normalize(phrase)) + 6
           for phrase in HALLUCINATION_PHRASES):
        return False
    if (segment.no_speech_probability or 0.0) > 0.6 and (segment.avg_logprob or 0.0) < -1.0:
        return False
    if (segment.compression_ratio or 0.0) > 2.4:
        return False
    return text != _normalize(previous_text)


def build_live_prompt(terms: tuple[str, ...], recent_text: str, limit: int = 160) -> str | None:
    base = _build_initial_prompt(terms, limit=400)
    tail = " ".join(recent_text.split())[-limit:]
    parts = [part for part in (base, tail) if part]
    return " ".join(parts) or None


# --------------------------------------------------------------------------- #
# Caption files
# --------------------------------------------------------------------------- #


class CaptionWriter:
    """Append captions to txt/srt files as they arrive so nothing is lost on a crash."""

    def __init__(self, output_root: Path, title: str, started_at: datetime):
        folder_name = safe_filename(f"{started_at:%Y-%m-%d %H%M} 실시간 자막 {title}")
        self.folder = output_root / folder_name
        self.folder.mkdir(parents=True, exist_ok=True)
        self.text_path = self.folder / "live_captions.txt"
        self.srt_path = self.folder / "live_captions.srt"
        self._index = 0
        header = f"# {title}\n# 시작: {started_at:%Y-%m-%d %H:%M:%S}\n\n"
        self.text_path.write_text(header, encoding="utf-8")
        self.srt_path.write_text("", encoding="utf-8")

    def append(self, caption: LiveCaption) -> None:
        self._index += 1
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{caption.wall_time:%H:%M:%S}] {caption.text}\n")
        with self.srt_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{self._index}\n{_srt_time(caption.start)} --> {_srt_time(caption.end)}\n"
                f"{caption.text}\n\n"
            )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


class LiveCaptioner:
    def __init__(
        self,
        options: LiveCaptionOptions,
        on_caption: Callable[[LiveCaption], None],
        on_partial: Callable[[str], None],
        log: LogCallback | None = None,
        transcribe: Callable[..., list[TranscriptSegment]] = transcribe_samples,
    ):
        self.options = options
        self.on_caption = on_caption
        self.on_partial = on_partial
        self.log = log or (lambda _message: None)
        self.transcribe = transcribe
        self.pending = np.zeros(0, dtype=np.float32)
        self.pending_start = 0.0
        self.started_at = datetime.now()
        self.recent_text = ""
        self._last_partial_size = 0

    # Separate from ``run`` so the segmentation logic can be driven by tests.
    def feed(self, samples: np.ndarray, backlog: float = 0.0) -> None:
        if samples.size:
            self.pending = np.concatenate([self.pending, samples])
        options = self.options
        max_samples = int(options.max_chunk * SAMPLE_RATE)
        if len(self.pending) >= max_samples:
            while len(self.pending) >= max_samples:
                cut = quietest_cut(
                    self.pending[:max_samples], search_seconds=min(5.0, options.max_chunk / 2)
                )
                self._finalize(cut)
            return
        duration = len(self.pending) / SAMPLE_RATE
        if duration >= options.min_chunk and (
            trailing_quiet_seconds(self.pending) >= options.trailing_silence
        ):
            self._finalize(len(self.pending))
            return
        if duration >= 2.0 and is_silent(self.pending):
            # Nothing said for a while: drop it so the window stays short.
            self._advance(len(self.pending))
            self.on_partial("")
            return
        new_audio = (len(self.pending) - self._last_partial_size) / SAMPLE_RATE
        if backlog <= options.max_backlog and new_audio >= options.partial_interval:
            self._last_partial_size = len(self.pending)
            if is_silent(self.pending):
                return
            segments = self._transcribe(self.pending)
            self.on_partial(" ".join(segment.text for segment in segments))

    def flush(self) -> None:
        if self.pending.size:
            self._finalize(len(self.pending))

    def _transcribe(self, samples: np.ndarray) -> list[TranscriptSegment]:
        options = self.options
        segments = self.transcribe(
            samples,
            model_name=options.model_name,
            language=options.language,
            initial_prompt=build_live_prompt(options.prompt_terms, self.recent_text),
            compute_type=options.compute_type,
            beam_size=options.beam_size,
            log=self.log,
        )
        kept: list[TranscriptSegment] = []
        previous = self.recent_text.split("\n")[-1] if self.recent_text else ""
        for segment in segments:
            if keep_segment(segment, previous):
                kept.append(segment)
                previous = segment.text
        return kept

    def _finalize(self, cut: int) -> None:
        chunk = self.pending[:cut]
        chunk_start = self.pending_start
        self._advance(cut)
        self.on_partial("")
        if is_silent(chunk):
            return
        chunk_seconds = len(chunk) / SAMPLE_RATE
        for segment in self._transcribe(chunk):
            start = chunk_start + max(0.0, min(segment.start, chunk_seconds))
            end = chunk_start + max(0.0, min(segment.end, chunk_seconds))
            caption = LiveCaption(
                start=start,
                end=max(end, start + 0.5),
                text=segment.text,
                wall_time=self.started_at + timedelta(seconds=start),
            )
            self.recent_text = (self.recent_text + "\n" + segment.text)[-600:]
            self.on_caption(caption)

    def _advance(self, cut: int) -> None:
        self.pending = self.pending[cut:]
        self.pending_start += cut / SAMPLE_RATE
        self._last_partial_size = 0


def run_live_captions(
    source_text: str,
    options: LiveCaptionOptions,
    output_root: Path,
    on_caption: Callable[[LiveCaption], None],
    on_partial: Callable[[str], None],
    on_status: Callable[[str], None],
    cancel_event: Event,
    cookie_browser: str = "none",
    log: LogCallback | None = None,
    on_progress: Callable[[bool, float | None], None] | None = None,
) -> Path:
    """Caption ``source_text`` until cancelled or the broadcast ends.

    ``on_progress(visible, fraction)`` drives a progress bar while the model is
    prepared; ``fraction`` is None when the amount of work is unknown.
    Returns the folder holding the saved caption files.
    """
    import imageio_ffmpeg

    log = log or (lambda _message: None)
    on_status("방송 오디오 주소를 찾는 중")
    source = resolve_live_source(source_text, cookie_browser, log)
    log(f"자막 대상: {source.title} ({'라이브' if source.is_live else '녹화본 실시간 재생'})")
    writer_holder: list[CaptionWriter] = []

    def handle_caption(caption: LiveCaption) -> None:
        writer_holder[0].append(caption)
        on_caption(caption)

    captioner = LiveCaptioner(options, handle_caption, on_partial, log)
    writer_holder.append(CaptionWriter(output_root, source.title, captioner.started_at))
    log(f"자막 저장 위치: {writer_holder[0].folder}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    show_progress = on_progress or (lambda _visible, _fraction: None)

    def report_download(progress: DownloadProgress) -> None:
        on_status(progress.message)
        show_progress(True, progress.fraction)

    try:
        on_status(f"{options.model_name} 모델 확인 중")
        show_progress(True, None)
        try:
            prefetch_model(options.model_name, report_download, cancel_event, log)
        except DownloadCancelled:
            on_status("모델 다운로드를 취소했습니다 · 받은 부분은 다음에 이어받습니다")
            return writer_holder[0].folder
        on_status(f"{options.model_name} 모델을 메모리에 올리는 중 · 처음에는 1분 정도 걸릴 수 있습니다")
        show_progress(True, None)
        # Warm the model up before audio starts piling up.
        captioner.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32), model_name=options.model_name,
            language=options.language, compute_type=options.compute_type,
            beam_size=options.beam_size, log=log,
        )
    finally:
        show_progress(False, None)

    reconnects = 0
    captioner.started_at = datetime.now()
    stream = PcmStream(build_ffmpeg_command(ffmpeg, source))
    on_status("자막 생성 중")
    last_lag_warning = 0.0
    try:
        while True:
            if cancel_event.is_set():
                captioner.flush()
                break
            samples = stream.take()
            if samples.size == 0:
                if stream.ended:
                    captioner.flush()
                    if not source.is_live or reconnects >= 5:
                        details = stream.error_text()
                        if details and captioner.pending_start < 1:
                            raise RuntimeError(f"방송 오디오를 받지 못했습니다.\n{details}")
                        on_status("방송 오디오가 끝났습니다")
                        break
                    reconnects += 1
                    on_status(f"방송 연결이 끊겨 다시 연결하는 중 ({reconnects}/5)")
                    log(f"ffmpeg 종료: {stream.error_text() or '원인 불명'} · 재연결합니다.")
                    stream.close()
                    time.sleep(min(10, 2 * reconnects))
                    try:
                        source = resolve_live_source(source_text, cookie_browser, log)
                    except Exception as exc:  # broadcast ended or network down
                        log(f"재연결 실패: {exc}")
                        continue
                    if not source.is_live:
                        on_status("방송이 종료되었습니다")
                        break
                    stream = PcmStream(build_ffmpeg_command(ffmpeg, source))
                    # Keep caption times aligned with the clock across the gap.
                    captioner.pending_start = (
                        datetime.now() - captioner.started_at
                    ).total_seconds()
                    on_status("자막 생성 중")
                    continue
                time.sleep(0.1)
                continue
            reconnects = 0
            # Audio that piled up while Whisper was busy with the previous step.
            backlog = samples.size / SAMPLE_RATE
            if backlog > 20 and time.monotonic() - last_lag_warning > 60:
                last_lag_warning = time.monotonic()
                log(
                    f"전사가 방송보다 {backlog:.0f}초 늦습니다. "
                    "더 가벼운 음성 인식 모델(large-v3-turbo, medium)을 선택해보세요."
                )
            captioner.feed(samples, backlog=backlog)
    finally:
        stream.close()
    return writer_holder[0].folder
