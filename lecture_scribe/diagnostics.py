from __future__ import annotations

import platform
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
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return GpuInfo(False, "Apple Silicon 없음", 0, "-", "arm64 macOS가 아닙니다.")
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GpuInfo(False, "Apple Silicon 확인 실패", 0, "-", str(exc))
    if result.returncode != 0:
        return GpuInfo(False, "Apple Silicon 확인 실패", 0, "-", result.stderr.strip())
    chip = "Apple Silicon"
    memory = 0
    for line in result.stdout.splitlines():
        key, _, value = line.strip().partition(":")
        if key in {"Chip", "Processor Name"} and value.strip():
            chip = value.strip()
        elif key == "Memory" and value.strip():
            amount, _, unit = value.strip().partition(" ")
            try:
                memory = int(float(amount) * (1024 if unit.upper() == "GB" else 1))
            except ValueError:
                pass
    return GpuInfo(True, chip, memory, platform.mac_ver()[0], "MLX Metal 사용 가능")
