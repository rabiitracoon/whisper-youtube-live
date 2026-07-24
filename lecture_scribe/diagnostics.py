from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuInfo:
    available: bool
    name: str
    memory_mb: int
    driver: str
    detail: str


def get_gpu_info() -> GpuInfo:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GpuInfo(False, "NVIDIA GPU 없음", 0, "-", str(exc))
    if result.returncode != 0 or not result.stdout.strip():
        return GpuInfo(False, "NVIDIA GPU 없음", 0, "-", result.stderr.strip())
    parts = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    if len(parts) < 3:
        return GpuInfo(False, "확인 실패", 0, "-", result.stdout.strip())
    try:
        memory = int(float(parts[1]))
    except ValueError:
        memory = 0
    return GpuInfo(True, parts[0], memory, parts[2], "CUDA 드라이버 정상")

