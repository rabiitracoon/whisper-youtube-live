from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from importlib import metadata
from threading import Event
from typing import Callable

from packaging.version import InvalidVersion, Version


LogCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class UpdateResult:
    previous: str
    latest: str
    current: str
    updated: bool
    channel: str


def current_version() -> str:
    try:
        return metadata.version("yt-dlp")
    except metadata.PackageNotFoundError:
        return "미설치"


def latest_version(channel: str = "nightly", timeout: int = 20) -> str:
    request = urllib.request.Request(
        "https://pypi.org/pypi/yt-dlp/json",
        headers={"User-Agent": "Lecture-Scribe/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    if channel == "stable":
        return str(data["info"]["version"])
    candidates: list[Version] = []
    for raw in data.get("releases", {}):
        try:
            candidates.append(Version(raw))
        except InvalidVersion:
            continue
    if not candidates:
        return str(data["info"]["version"])
    return str(max(candidates))


def check_and_update(
    channel: str = "nightly",
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> UpdateResult:
    previous = current_version()
    if log:
        log(f"PyPI에서 yt-dlp {channel} 채널의 최신 버전을 확인합니다.")
    latest = latest_version(channel)
    needs_update = previous == "미설치" or _is_newer(latest, previous)
    if not needs_update:
        if log:
            log(f"이미 최신 버전입니다: {previous}")
        return UpdateResult(previous, latest, previous, False, channel)
    command = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if channel == "nightly":
        command.append("--pre")
    command.append("yt-dlp[default]")
    if log:
        log(f"yt-dlp {previous} → {latest} 업데이트를 시작합니다.")
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
    for line in iter(process.stdout.readline, ""):
        if cancel_event and cancel_event.is_set():
            process.terminate()
            raise RuntimeError("yt-dlp 업데이트가 취소되었습니다.")
        cleaned = line.strip()
        if cleaned and log:
            log(cleaned)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("yt-dlp 업데이트에 실패했습니다. 로그를 확인해주세요.")
    current = current_version()
    if log:
        log(f"업데이트 완료: yt-dlp {current}")
    return UpdateResult(previous, latest, current, True, channel)


def _is_newer(candidate: str, current: str) -> bool:
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return candidate != current

