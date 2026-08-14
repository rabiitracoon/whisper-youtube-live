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
    if platform.system() == "Windows":
        return _get_nvidia_gpu_info()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return _get_apple_silicon_info()
    return GpuInfo(
        False,
        "지원 환경 없음",
        0,
        "-",
        "NVIDIA GPU Windows 또는 Apple Silicon Mac이 필요합니다.",
    )


def _get_nvidia_gpu_info() -> GpuInfo:
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


def _get_apple_silicon_info() -> GpuInfo:
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
