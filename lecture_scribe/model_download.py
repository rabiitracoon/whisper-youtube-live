"""Download speech models from Hugging Face while reporting real progress.

``faster-whisper`` and ``transformers`` hide the download behind the model
constructor, so a first run looks frozen for minutes. Fetching the files up front
lets the UI show bytes, percent, speed and whether data is still arriving.

Progress is measured from the files growing inside the Hugging Face cache rather
than from tqdm hooks. The ``hf_xet`` transfer backend is switched off because it
keeps the data in memory and writes the file in one burst at the end, which makes
a large download look stuck at 0% until it suddenly finishes.
"""

from __future__ import annotations

import fnmatch
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from .qwen_asr_engine import QWEN_MODELS, is_qwen_model
from .transcription import resolve_mlx_model, runtime_backend

LogCallback = Callable[[str], None]
POLL_SECONDS = 0.5
# Speed is averaged over this window so the number does not jump around.
SPEED_WINDOW_SECONDS = 6.0
STALL_NOTICE_SECONDS = 15.0
# The files faster-whisper loads; everything else in the repo is skipped.
FASTER_WHISPER_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
)


class DownloadCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    label: str
    done: int
    total: int | None
    speed: float
    stalled: float

    @property
    def fraction(self) -> float | None:
        if not self.total:
            return None
        return min(1.0, self.done / self.total)

    @property
    def message(self) -> str:
        parts = [f"{self.label} 다운로드 중"]
        if self.total:
            percent = int((self.fraction or 0.0) * 100)
            parts.append(f"{format_bytes(self.done)} / {format_bytes(self.total)} ({percent}%)")
        else:
            parts.append(f"{format_bytes(self.done)} 받음")
        if self.stalled >= STALL_NOTICE_SECONDS:
            parts.append(f"{int(self.stalled)}초째 진행이 없습니다 · 네트워크 상태를 확인해주세요")
        elif self.speed > 0:
            parts.append(f"{format_bytes(self.speed)}/초")
            if self.total and self.done < self.total:
                parts.append(f"남은 시간 약 {format_eta((self.total - self.done) / self.speed)}")
        else:
            parts.append("연결 중")
        return " · ".join(parts)


@dataclass(frozen=True, slots=True)
class ModelRepo:
    repo_id: str
    label: str
    allow_patterns: tuple[str, ...] | None = None


def format_bytes(size: float) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def format_eta(seconds: float) -> str:
    total = max(1, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes}분"
    if minutes:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"


def model_repositories(model_name: str) -> list[ModelRepo]:
    """Hugging Face repositories that ``model_name`` needs on this machine."""
    if is_qwen_model(model_name):
        # Only the CUDA runner is known to read the standard Hugging Face cache.
        if runtime_backend() != "cuda":
            return []
        return [ModelRepo(QWEN_MODELS[model_name], f"Qwen3-ASR {model_name.rsplit('-', 1)[-1].upper()} 모델")]
    label = f"Whisper {model_name} 모델"
    backend = runtime_backend()
    if backend == "cuda":
        repo_id = _faster_whisper_repo(model_name)
        if repo_id is None:
            return []
        return [ModelRepo(repo_id, label, FASTER_WHISPER_PATTERNS)]
    if backend == "mlx":
        repo_id = resolve_mlx_model(model_name)
        if os.path.isdir(repo_id):
            return []
        return [ModelRepo(repo_id, label)]
    return []


def _faster_whisper_repo(model_name: str) -> str | None:
    if "/" in model_name:
        return model_name
    try:
        from faster_whisper.utils import _MODELS
    except ImportError:
        return None
    return _MODELS.get(model_name)


def _repo_dir(repo_id: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name

    return Path(HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model")


def _downloaded_bytes(repo_dir: Path) -> int:
    """Bytes of this repo already on disk, finished or still downloading."""
    total = 0
    try:
        names = os.listdir(repo_dir / "blobs")
    except OSError:
        names = []
    for name in names:
        # os.path.getsize, not DirEntry.stat(): on Windows the directory entry keeps
        # a stale size while another handle is still writing the file.
        try:
            total += os.path.getsize(repo_dir / "blobs" / name)
        except OSError:
            continue
    # Without symlink support (typical on Windows) a finished file leaves blobs/
    # and sits in snapshots/ as a regular file instead.
    for root, _dirs, files in os.walk(repo_dir / "snapshots"):
        for name in files:
            path = os.path.join(root, name)
            try:
                if not os.path.islink(path):
                    total += os.path.getsize(path)
            except OSError:
                continue
    return total


def is_cached(repo: ModelRepo) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo.repo_id, allow_patterns=repo.allow_patterns, local_files_only=True
        )
    except Exception:  # not (completely) in the cache
        return False
    return True


def remote_size(repo: ModelRepo) -> int | None:
    """Total bytes still to fetch for ``repo``, or None when the hub cannot be asked."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo.repo_id, files_metadata=True)
        patterns = repo.allow_patterns
        total = 0
        for sibling in info.siblings or []:
            if patterns and not any(fnmatch.fnmatch(sibling.rfilename, p) for p in patterns):
                continue
            total += int(sibling.size or 0)
    except Exception:
        return None
    return total or None


class _SpeedMeter:
    def __init__(self) -> None:
        self._samples: deque[tuple[float, int]] = deque()
        self._last_growth = time.monotonic()
        self._last_size = -1

    def update(self, size: int) -> tuple[float, float]:
        """Return (bytes per second, seconds since the size last grew)."""
        now = time.monotonic()
        if size != self._last_size:
            if self._last_size >= 0 and size > self._last_size:
                self._last_growth = now
            self._last_size = size
        self._samples.append((now, size))
        while len(self._samples) > 2 and now - self._samples[0][0] > SPEED_WINDOW_SECONDS:
            self._samples.popleft()
        first_time, first_size = self._samples[0]
        elapsed = now - first_time
        speed = max(0.0, (size - first_size) / elapsed) if elapsed >= 1.0 else 0.0
        return speed, now - self._last_growth


def download_repo(
    repo: ModelRepo,
    on_progress: Callable[[DownloadProgress], None],
    cancel_event: Event | None = None,
    log: LogCallback | None = None,
) -> bool:
    """Make sure ``repo`` is in the local cache. Returns True when a download happened."""
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    if is_cached(repo):
        return False
    # A windowed app has no console for the library's own tqdm bars.
    disable_progress_bars()
    constants.HF_HUB_DISABLE_XET = True
    total = remote_size(repo)
    repo_dir = _repo_dir(repo.repo_id)
    if log:
        size_text = format_bytes(total) if total else "크기 미확인"
        log(f"{repo.label} 다운로드를 시작합니다 ({size_text}). 이어받기를 지원합니다.")

    outcome: list[BaseException] = []

    def fetch() -> None:
        try:
            snapshot_download(repo.repo_id, allow_patterns=repo.allow_patterns)
        except BaseException as exc:  # reported on the calling thread
            outcome.append(exc)

    thread = threading.Thread(target=fetch, name="model-download", daemon=True)
    thread.start()
    meter = _SpeedMeter()
    started_with = _downloaded_bytes(repo_dir)
    while thread.is_alive():
        if cancel_event is not None and cancel_event.is_set():
            # The partial files stay in the cache and resume on the next run.
            raise DownloadCancelled("모델 다운로드를 취소했습니다.")
        done = _downloaded_bytes(repo_dir)
        if total:
            done = min(done, total)
        speed, stalled = meter.update(done)
        on_progress(DownloadProgress(repo.label, done, total, speed, stalled))
        thread.join(POLL_SECONDS)
    if outcome:
        raise RuntimeError(
            f"{repo.label}을 다운로드하지 못했습니다. 인터넷 연결을 확인하고 다시 시도해주세요. "
            f"(받던 파일은 보존되어 이어받습니다)\n{outcome[0]}"
        ) from outcome[0]
    final = _downloaded_bytes(repo_dir)
    on_progress(DownloadProgress(repo.label, total or final, total or final, 0.0, 0.0))
    if log:
        log(f"{repo.label} 다운로드를 마쳤습니다 ({format_bytes(max(final - started_with, 0))} 받음).")
    return True


def prefetch_model(
    model_name: str,
    on_progress: Callable[[DownloadProgress], None],
    cancel_event: Event | None = None,
    log: LogCallback | None = None,
    repositories: Sequence[ModelRepo] | None = None,
) -> bool:
    """Download everything ``model_name`` needs; True if anything had to be fetched."""
    fetched = False
    for repo in repositories if repositories is not None else model_repositories(model_name):
        fetched = download_repo(repo, on_progress, cancel_event, log) or fetched
    return fetched
