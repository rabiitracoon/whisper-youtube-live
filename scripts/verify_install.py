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
        import mlx.core as mx
        import mlx_whisper  # noqa: F401
        import imageio_ffmpeg
        import PySide6
        import yt_dlp
        from lecture_scribe.video_edit import _ffmpeg_videotoolbox_usable
    except Exception as exc:
        print(f"Dependency import failed: {exc}", file=sys.stderr)
        return 1
    print(f"PySide6 {PySide6.__version__}")
    print(f"yt-dlp {yt_dlp.version.__version__}")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"FFmpeg: {ffmpeg_exe}")
    if not _ffmpeg_videotoolbox_usable(ffmpeg_exe):
        print("Apple VideoToolbox encoder cannot be initialized.", file=sys.stderr)
        return 4
    print("Video encoder: Apple VideoToolbox")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("MLX requires an Apple Silicon (arm64) Mac.", file=sys.stderr)
        return 2
    try:
        mx.eval(mx.zeros((1,)))
    except RuntimeError as exc:
        print(f"MLX cannot access Metal: {exc}", file=sys.stderr)
        return 3
    print("Accelerator: Apple Silicon MLX Metal")
    print("Verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
