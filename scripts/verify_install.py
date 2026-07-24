from __future__ import annotations

import sys


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    try:
        import PySide6
        import ctranslate2
        import torch
        import yt_dlp
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception as exc:
        print(f"Dependency import failed: {exc}", file=sys.stderr)
        return 1
    print(f"PySide6 {PySide6.__version__}")
    print(f"yt-dlp {yt_dlp.version.__version__}")
    print(f"PyTorch {torch.__version__}")
    if not torch.cuda.is_available():
        print("PyTorch cannot access an NVIDIA CUDA GPU.", file=sys.stderr)
        return 2
    # Load CUDA libraries before asking CTranslate2 to enumerate devices.
    torch.empty(1, device="cuda")
    count = ctranslate2.get_cuda_device_count()
    if count < 1:
        print("CTranslate2 cannot access the CUDA GPU.", file=sys.stderr)
        return 3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CTranslate2 CUDA devices: {count}")
    print("Verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

