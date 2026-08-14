from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from threading import Event


class VideoRenderCancelled(RuntimeError):
    pass


def build_video_filter_graph(
    included_ranges: tuple[tuple[float, float], ...],
) -> str:
    if not included_ranges:
        raise ValueError("영상에 포함할 강의 구간이 없습니다.")
    filters: list[str] = []
    inputs: list[str] = []
    for index, (start, end) in enumerate(included_ranges):
        if end <= start:
            raise ValueError("영상 편집 구간이 올바르지 않습니다.")
        filters.append(
            f"[0:v]trim=start={start:.6f}:end={end:.6f},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        filters.append(
            f"[0:a]atrim=start={start:.6f}:end={end:.6f},"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        inputs.append(f"[v{index}][a{index}]")
    filters.append(
        "".join(inputs)
        + f"concat=n={len(included_ranges)}:v=1:a=1[outv][outa]"
    )
    return ";\n".join(filters)


def render_edited_video(
    source_path: Path,
    output_path: Path,
    included_ranges: tuple[tuple[float, float], ...],
    cancel_event: Event | None = None,
    progress: Callable[[float], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Render selected source ranges as one broadly compatible MP4 file."""
    import imageio_ffmpeg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.stem + ".partial.mp4")
    filter_path = output_path.with_name(output_path.stem + ".filters.txt")
    for stale_path in (partial_path, filter_path):
        if stale_path.exists():
            stale_path.unlink()
    filter_path.write_text(
        build_video_filter_graph(included_ranges), encoding="utf-8"
    )
    total_duration = sum(end - start for start, end in included_ranges)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command_prefix = [
        ffmpeg_exe,
        "-hide_banner",
        "-y",
        "-i",
        str(source_path),
        "-filter_complex_script",
        str(filter_path),
        "-map",
        "[outv]",
        "-map",
        "[outa]",
    ]
    command_suffix = [
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        str(partial_path),
    ]
    try:
        encoder_args, encoder_label = _select_hardware_encoder(ffmpeg_exe)
    except RuntimeError:
        filter_path.unlink(missing_ok=True)
        raise
    try:
        if log:
            log(f"영상 인코더: {encoder_label}")
        try:
            _run_ffmpeg(
                [*command_prefix, *encoder_args, *command_suffix],
                cancel_event,
                progress,
                total_duration,
            )
        except VideoRenderCancelled:
            raise
        except RuntimeError as exc:
            raise RuntimeError(
                f"{encoder_label} 영상 인코딩에 실패했습니다. "
                "CPU 인코더로 대체하지 않았습니다.\n"
                f"{exc}"
            ) from exc
        partial_path.replace(output_path)
        if progress:
            progress(1.0)
        return output_path.resolve()
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise
    finally:
        if filter_path.exists():
            filter_path.unlink()


@lru_cache(maxsize=8)
def _ffmpeg_supports_encoder(ffmpeg_exe: str, encoder: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and encoder in result.stdout


def _select_hardware_encoder(ffmpeg_exe: str) -> tuple[list[str], str]:
    if platform.system() == "Darwin":
        if not _ffmpeg_videotoolbox_usable(ffmpeg_exe):
            raise RuntimeError(
                "Apple VideoToolbox 하드웨어 인코더를 초기화하지 못했습니다. "
                "일반 macOS 로그인 세션에서 다시 실행해주세요. "
                "영상 편집본은 CPU 인코더로 대체하지 않았습니다."
            )
        return (
            [
                "-c:v", "h264_videotoolbox", "-allow_sw", "0", "-realtime", "0",
                "-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M",
            ],
            "Apple VideoToolbox (하드웨어 가속)",
        )
    if platform.system() == "Windows":
        if not _ffmpeg_nvenc_usable(ffmpeg_exe):
            raise RuntimeError(
                "NVIDIA NVENC를 초기화하지 못했습니다. NVIDIA 드라이버 상태를 확인하고 "
                "PC를 재부팅한 뒤 다시 시도하세요. 영상 편집본은 CPU로 대체하지 않았습니다."
            )
        return (
            ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21", "-b:v", "0"],
            "NVIDIA NVENC (GPU 전용)",
        )
    raise RuntimeError("영상 하드웨어 인코딩은 Windows 또는 macOS에서만 지원합니다.")


@lru_cache(maxsize=8)
def _ffmpeg_videotoolbox_usable(ffmpeg_exe: str) -> bool:
    return _ffmpeg_encoder_usable(ffmpeg_exe, "h264_videotoolbox", ("-allow_sw", "0"))


@lru_cache(maxsize=8)
def _ffmpeg_nvenc_usable(ffmpeg_exe: str) -> bool:
    return _ffmpeg_encoder_usable(ffmpeg_exe, "h264_nvenc")


def _ffmpeg_encoder_usable(
    ffmpeg_exe: str,
    encoder: str,
    extra_args: tuple[str, ...] = (),
) -> bool:
    if not _ffmpeg_supports_encoder(ffmpeg_exe, encoder):
        return False
    try:
        result = subprocess.run(
            [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=320x180:rate=1:duration=0.1",
                "-frames:v",
                "1",
                "-c:v",
                encoder,
                *extra_args,
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _run_ffmpeg(
    command: list[str],
    cancel_event: Event | None,
    progress: Callable[[float], None] | None,
    total_duration: float,
) -> None:
    recent_output: list[str] = []
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
        assert process.stdout is not None
        for raw_line in iter(process.stdout.readline, ""):
            if cancel_event and cancel_event.is_set():
                process.terminate()
                process.wait(timeout=10)
                raise VideoRenderCancelled("사용자가 영상 편집본 저장을 취소했습니다.")
            line = raw_line.strip()
            if line:
                recent_output.append(line)
                recent_output = recent_output[-100:]
            if progress and line.startswith("out_time_us="):
                try:
                    current = int(line.partition("=")[2]) / 1_000_000
                except ValueError:
                    continue
                progress(min(0.99, current / max(0.1, total_duration)))
        return_code = process.wait()
        if return_code != 0:
            detail = "\n".join(recent_output[-40:])
            raise RuntimeError(
                "쉬는 시간을 제거한 영상 저장에 실패했습니다."
                + (f"\n{detail}" if detail else "")
            )
        return
    except Exception:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process and process.stdout:
            process.stdout.close()
