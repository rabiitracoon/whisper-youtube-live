from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from .paths import safe_filename
from .youtube import VideoInfo


class LocalMediaCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LocalMedia:
    path: Path
    video: VideoInfo
    has_video: bool


def resolve_local_media(value: str) -> LocalMedia:
    """Validate a local audio/video file and collect enough metadata for a job."""
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError("선택한 영상 또는 음성 파일을 찾을 수 없습니다.")
    path = path.resolve()
    try:
        import av

        with av.open(str(path)) as container:
            streams = list(container.streams)
            has_audio = any(stream.type == "audio" for stream in streams)
            has_video = any(stream.type == "video" for stream in streams)
            if not has_audio:
                raise ValueError("음성 트랙이 있는 영상 또는 음성 파일을 선택해주세요.")
            duration = (
                float(container.duration / av.time_base)
                if container.duration is not None
                else 0.0
            )
            if duration <= 0:
                stream_durations = [
                    float(stream.duration * stream.time_base)
                    for stream in streams
                    if stream.duration is not None and stream.time_base is not None
                ]
                duration = max(stream_durations, default=0.0)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "파일을 읽을 수 없습니다. 지원되는 영상 또는 음성 파일인지 확인해주세요."
        ) from exc
    if duration <= 0:
        raise ValueError("파일 길이를 확인하지 못했습니다. 다른 영상 또는 음성 파일을 선택해주세요.")

    stat = path.stat()
    fingerprint = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    media_id = "local_" + hashlib.sha256(fingerprint).hexdigest()[:12]
    title = safe_filename(path.stem, 120)
    return LocalMedia(
        path=path,
        video=VideoInfo(
            video_id=media_id,
            title=title,
            channel="내 컴퓨터의 파일",
            duration=duration,
            webpage_url=str(path),
        ),
        has_video=has_video,
    )


def copy_local_media(
    media: LocalMedia,
    destination: Path,
    *,
    cancel_event: Event | None = None,
) -> Path:
    """Copy the source into the job folder; never modify or remove the user's file."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"original_media{media.path.suffix.lower()}"
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with media.path.open("rb") as source, partial.open("wb") as output:
            while chunk := source.read(4 * 1024 * 1024):
                if cancel_event and cancel_event.is_set():
                    raise LocalMediaCancelled("사용자가 파일 가져오기를 취소했습니다.")
                output.write(chunk)
        shutil.copystat(media.path, partial)
        partial.replace(target)
        return target.resolve()
    except Exception:
        partial.unlink(missing_ok=True)
        raise
