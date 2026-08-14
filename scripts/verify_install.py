from __future__ import annotations

import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    try:
        import imageio_ffmpeg
        import PySide6
        import yt_dlp
        from faster_whisper import WhisperModel  # noqa: F401
        from lecture_scribe.transcription import verify_accelerator
        from lecture_scribe.video_edit import (
            _ffmpeg_nvenc_usable,
            _ffmpeg_videotoolbox_usable,
        )
    except (ImportError, OSError) as exc:
        print(f"Dependency import failed: {exc}", file=sys.stderr)
        return 1

    print(f"PySide6 {PySide6.__version__}")
    print(f"yt-dlp {yt_dlp.version.__version__}")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"FFmpeg: {ffmpeg_exe}")
    if platform.system() == "Windows":
        if not _ffmpeg_nvenc_usable(ffmpeg_exe):
            print("NVIDIA NVENC cannot be initialized.", file=sys.stderr)
            return 4
        print("Video encoder: NVIDIA NVENC")
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:
            print(f"MLX dependency import failed: {exc}", file=sys.stderr)
            return 1
        if not _ffmpeg_videotoolbox_usable(ffmpeg_exe):
            print("Apple VideoToolbox encoder cannot be initialized.", file=sys.stderr)
            return 4
        print("Video encoder: Apple VideoToolbox")
    else:
        print("Unsupported operating system or CPU architecture.", file=sys.stderr)
        return 2

    try:
        accelerator = verify_accelerator()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(f"Accelerator: {accelerator}")
    print("Verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
