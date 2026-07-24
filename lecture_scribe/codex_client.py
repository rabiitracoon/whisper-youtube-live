from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from .paths import LOCAL_CODEX_ROOT, PROJECT_ROOT


LogCallback = Callable[[str], None]
ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class CodexCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AuthStatus:
    available: bool
    logged_in: bool
    detail: str


def find_codex() -> Path | None:
    explicit = os.environ.get("LECTURE_SCRIBE_CODEX")
    if explicit and Path(explicit).exists():
        return Path(explicit).resolve()
    local_candidates = [
        LOCAL_CODEX_ROOT / "node_modules" / ".bin" / "codex.cmd",
        LOCAL_CODEX_ROOT / "node_modules" / ".bin" / "codex.exe",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate.resolve()
    if LOCAL_CODEX_ROOT.exists():
        vendor_bins = list(LOCAL_CODEX_ROOT.glob("node_modules/@openai/codex*/vendor/**/codex.exe"))
        if vendor_bins:
            return vendor_bins[0].resolve()
    located = shutil.which("codex")
    return Path(located).resolve() if located else None


def _command_prefix(executable: Path) -> list[str]:
    if executable.suffix.lower() in {".cmd", ".bat"}:
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(executable)]
    return [str(executable)]


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def auth_status(timeout: int = 20) -> AuthStatus:
    executable = find_codex()
    if not executable:
        return AuthStatus(
            available=False,
            logged_in=False,
            detail="Codex CLI가 없습니다. install.bat을 실행해주세요.",
        )
    try:
        result = subprocess.run(
            _command_prefix(executable) + ["login", "status"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_creation_flags(),
            env=_clean_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AuthStatus(True, False, f"로그인 상태 확인 실패: {exc}")
    detail = _clean_output(result.stdout or result.stderr).strip()
    return AuthStatus(True, result.returncode == 0, detail or "로그인 정보가 없습니다.")


def login(
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> AuthStatus:
    executable = find_codex()
    if not executable:
        raise FileNotFoundError("Codex CLI가 없습니다. install.bat을 실행해주세요.")
    if log:
        log("브라우저에서 ChatGPT 로그인을 완료해주세요. 앱은 OAuth 토큰을 직접 저장하지 않습니다.")
    return_code, _, stderr = _run_streaming(
        _command_prefix(executable) + ["login"],
        input_text=None,
        log=log,
        cancel_event=cancel_event,
        emit_stderr=True,
    )
    if return_code != 0:
        raise RuntimeError(_last_error(stderr, "Codex OAuth 로그인에 실패했습니다."))
    return auth_status()


def logout() -> AuthStatus:
    executable = find_codex()
    if not executable:
        raise FileNotFoundError("Codex CLI가 없습니다.")
    result = subprocess.run(
        _command_prefix(executable) + ["logout"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        creationflags=_creation_flags(),
        env=_clean_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(_last_error(result.stderr, "로그아웃에 실패했습니다."))
    return auth_status()


def generate_notes(
    prompt: str,
    output_path: Path,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "high",
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> Path:
    executable = find_codex()
    if not executable:
        raise FileNotFoundError("Codex CLI가 없습니다. install.bat을 실행해주세요.")
    command = _command_prefix(executable) + [
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--ignore-user-config",
        "--ignore-rules",
        "--color",
        "never",
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-",
    ]
    if log:
        log(f"Codex {model} · reasoning={reasoning_effort}로 강의 노트를 생성합니다.")
    return_code, stdout, stderr = _run_streaming(
        command,
        input_text=prompt,
        log=log,
        cancel_event=cancel_event,
        emit_stderr=False,
    )
    if return_code != 0:
        message = _last_error(stderr, "AI 노트 생성에 실패했습니다.")
        if "login" in message.lower() or "unauthorized" in message.lower():
            message += "\n도구 탭에서 ChatGPT OAuth 로그인을 다시 진행해주세요."
        raise RuntimeError(message)
    final_text = _clean_output(stdout).strip()
    if not final_text:
        raise RuntimeError("Codex 실행은 끝났지만 노트 내용이 비어 있습니다.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_text.rstrip() + "\n", encoding="utf-8")
    return output_path.resolve()


def _run_streaming(
    command: list[str],
    input_text: str | None,
    log: LogCallback | None,
    cancel_event: Event | None,
    emit_stderr: bool,
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        env=_clean_env(),
        bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stream(stream, target: list[str], emit: bool) -> None:
        try:
            for line in iter(stream.readline, ""):
                target.append(line)
                cleaned = _clean_output(line).strip()
                if emit and cleaned and log:
                    log(cleaned)
        finally:
            stream.close()

    out_thread = threading.Thread(
        target=read_stream, args=(process.stdout, stdout_lines, False), daemon=True
    )
    err_thread = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_lines, emit_stderr),
        daemon=True,
    )
    out_thread.start()
    err_thread.start()
    if input_text is not None and process.stdin:
        try:
            process.stdin.write(input_text)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    while process.poll() is None:
        if cancel_event and cancel_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise CodexCancelled("사용자가 AI 노트 생성을 취소했습니다.")
        time.sleep(0.2)
    out_thread.join(timeout=3)
    err_thread.join(timeout=3)
    return process.returncode or 0, "".join(stdout_lines), "".join(stderr_lines)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    return env


def _clean_output(value: str) -> str:
    return ANSI_RE.sub("", value).replace("\r", "")


def _last_error(stderr: str, fallback: str) -> str:
    lines = [line.strip() for line in _clean_output(stderr).splitlines() if line.strip()]
    return "\n".join(lines[-8:]) if lines else fallback
