from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from urllib.parse import urlparse

ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]


class DownloadCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VideoInfo:
    video_id: str
    title: str
    channel: str
    duration: float
    webpage_url: str


def is_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return host in {
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "youtube-nocookie.com",
    }


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _base_options(cookie_browser: str, log: LogCallback | None = None) -> dict:
    import imageio_ffmpeg

    options: dict = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "windowsfilenames": True,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        # Some YouTube audio formats are HLS streams. yt-dlp needs ffmpeg even
        # when no post-processing is requested, so always point it at the
        # executable bundled by imageio-ffmpeg instead of relying on PATH.
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
    }
    node_path = shutil.which("node")
    if node_path:
        options["js_runtimes"] = {"node": {"path": node_path}}
    if cookie_browser and cookie_browser != "none":
        options["cookiesfrombrowser"] = (cookie_browser,)
    if log:
        options["logger"] = _YtdlpLogger(log)
    return options


class _YtdlpLogger:
    def __init__(self, callback: LogCallback):
        self.callback = callback

    def debug(self, message: str) -> None:
        if message.startswith("[debug]"):
            return
        self.callback(message)

    def info(self, message: str) -> None:
        self.callback(message)

    def warning(self, message: str) -> None:
        if message.lower().startswith("ffmpeg not found"):
            # Kept as a guard for older yt-dlp versions with misleading probes.
            return
        self.callback(f"경고: {message}")

    def error(self, message: str) -> None:
        self.callback(f"오류: {message}")


def fetch_video_info(
    url: str,
    cookie_browser: str = "none",
    log: LogCallback | None = None,
) -> VideoInfo:
    if not is_youtube_url(url):
        raise ValueError("올바른 YouTube 영상 URL을 입력해주세요.")
    from yt_dlp import YoutubeDL

    options = _base_options(cookie_browser, log)
    options["skip_download"] = True
    with YoutubeDL(options) as ydl:
        data = ydl.extract_info(url.strip(), download=False)
    if not data or data.get("_type") == "playlist":
        raise ValueError("재생목록이 아닌 단일 YouTube 영상 링크를 입력해주세요.")
    return VideoInfo(
        video_id=str(data.get("id") or "video"),
        title=str(data.get("title") or "제목 없는 영상"),
        channel=str(data.get("channel") or data.get("uploader") or "알 수 없음"),
        duration=float(data.get("duration") or 0),
        webpage_url=str(data.get("webpage_url") or url.strip()),
    )


def download_audio(
    url: str,
    video_id: str,
    destination: Path,
    cookie_browser: str = "none",
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> Path:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    destination.mkdir(parents=True, exist_ok=True)
    options = _base_options(cookie_browser, log)
    options.update(
        {
            "format": "bestaudio/best",
            "outtmpl": str(destination / f"{video_id}.%(ext)s"),
            "progress_hooks": [
                _progress_hook(
                    progress=progress,
                    cancel_event=cancel_event,
                    media_label="오디오",
                )
            ],
        }
    )
    try:
        with YoutubeDL(options) as ydl:
            data = ydl.extract_info(url.strip(), download=True)
    except DownloadError as exc:
        if not _is_http_403(exc):
            raise
        if log:
            log(
                "YouTube가 기본 오디오 요청을 거부했습니다. "
                "모바일 웹 재생 경로로 다시 시도합니다."
            )
        fallback_options = dict(options)
        fallback_options.update(
            {
                "format": "bestaudio/best",
                "extractor_args": {
                    "youtube": {"player_client": ["mweb"]}
                },
                "continuedl": False,
            }
        )
        try:
            with YoutubeDL(fallback_options) as ydl:
                data = ydl.extract_info(url.strip(), download=True)
        except DownloadError as fallback_exc:
            raise RuntimeError(
                "YouTube가 기본 경로와 모바일 웹 경로의 오디오 다운로드를 모두 거부했습니다. "
                "앱 설정에서 로그인된 브라우저 쿠키를 선택한 뒤 다시 시도하세요. "
                "그래도 실패하면 이 영상 또는 네트워크에는 PO Token Provider가 필요합니다."
            ) from fallback_exc

    candidates: list[Path] = []
    for item in (data or {}).get("requested_downloads", []):
        if item.get("filepath"):
            candidates.append(Path(item["filepath"]))
    if data and data.get("_filename"):
        candidates.append(Path(data["_filename"]))
    candidates.extend(destination.glob(f"{video_id}.*"))
    for candidate in candidates:
        if candidate.exists() and candidate.suffix not in {".part", ".ytdl"}:
            return candidate.resolve()
    raise FileNotFoundError("다운로드는 완료되었지만 오디오 파일을 찾지 못했습니다.")


def download_video(
    url: str,
    video_id: str,
    destination: Path,
    cookie_browser: str = "none",
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> Path:
    """Download a maximum-1080p video with audio and merge it as MP4."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    destination.mkdir(parents=True, exist_ok=True)
    options = _base_options(cookie_browser, log)
    options.update(
        {
            "format": "bv*[height<=1080]+ba/b[height<=1080]/best",
            "merge_output_format": "mp4",
            "outtmpl": str(destination / "original_video.%(ext)s"),
            "progress_hooks": [
                _progress_hook(
                    progress=progress,
                    cancel_event=cancel_event,
                    media_label="영상",
                )
            ],
        }
    )
    try:
        with YoutubeDL(options) as ydl:
            data = ydl.extract_info(url.strip(), download=True)
    except DownloadError as exc:
        if not _is_http_403(exc):
            raise
        if log:
            log(
                "YouTube가 기본 영상 요청을 거부했습니다. "
                "모바일 웹 재생 경로로 다시 시도합니다."
            )
        fallback_options = dict(options)
        fallback_options.update(
            {
                "format": "best[height<=1080]/best",
                "extractor_args": {"youtube": {"player_client": ["mweb"]}},
                "continuedl": False,
            }
        )
        try:
            with YoutubeDL(fallback_options) as ydl:
                data = ydl.extract_info(url.strip(), download=True)
        except DownloadError as fallback_exc:
            raise RuntimeError(
                "YouTube가 기본 경로와 모바일 웹 경로의 영상 다운로드를 모두 거부했습니다. "
                "앱 설정에서 로그인된 브라우저 쿠키를 선택한 뒤 다시 시도하세요."
            ) from fallback_exc

    candidates: list[Path] = list(destination.glob("original_video.mp4"))
    for item in (data or {}).get("requested_downloads", []):
        if item.get("filepath"):
            candidates.append(Path(item["filepath"]))
    if data and data.get("_filename"):
        candidates.append(Path(data["_filename"]))
    candidates.extend(destination.glob("original_video.*"))
    for candidate in candidates:
        if (
            candidate.exists()
            and candidate.stem == "original_video"
            and candidate.suffix not in {".part", ".ytdl"}
        ):
            return candidate.resolve()
    raise FileNotFoundError("다운로드는 완료되었지만 영상 파일을 찾지 못했습니다.")


def _is_http_403(error: BaseException) -> bool:
    """Return whether yt-dlp's wrapped error represents a denied media URL."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if "http error 403" in message or "403: forbidden" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _progress_hook(
    progress: ProgressCallback | None,
    cancel_event: Event | None,
    media_label: str = "미디어",
) -> Callable[[dict], None]:
    def hook(data: dict) -> None:
        if cancel_event and cancel_event.is_set():
            raise DownloadCancelled("사용자가 다운로드를 취소했습니다.")
        if not progress:
            return
        status = data.get("status")
        if status == "finished":
            progress(1.0, f"{media_label} 다운로드 완료")
            return
        if status != "downloading":
            return
        downloaded = float(data.get("downloaded_bytes") or 0)
        total = float(data.get("total_bytes") or data.get("total_bytes_estimate") or 0)
        ratio = min(1.0, downloaded / total) if total else 0.0
        speed = data.get("_speed_str", "").strip()
        eta = data.get("_eta_str", "").strip()
        detail = f"{media_label} 다운로드 중"
        if speed:
            detail += f" · {speed}"
        if eta:
            detail += f" · 남은 시간 {eta}"
        progress(ratio, detail)

    return hook
