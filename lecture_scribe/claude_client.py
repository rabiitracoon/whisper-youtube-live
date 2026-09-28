from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from threading import Event

from .codex_client import (
    AuthStatus,
    LogCallback,
    _clean_env,
    _clean_output,
    _command_prefix,
    _creation_flags,
    _last_error,
    _run_streaming,
)
from .paths import LOCAL_CLAUDE_ROOT, PROJECT_ROOT


INSTALLER_NAME = "install.bat" if platform.system() == "Windows" else "install.command"
NOTE_SYSTEM_PROMPT = (
    "You are a meticulous study-notes assistant. Follow the user's instructions exactly "
    "and reply with only the requested document, without preamble or closing remarks."
)


def find_claude() -> Path | None:
    explicit = os.environ.get("LECTURE_SCRIBE_CLAUDE")
    if explicit and Path(explicit).exists():
        return Path(explicit).resolve()
    package_bin = LOCAL_CLAUDE_ROOT / "node_modules" / "@anthropic-ai" / "claude-code" / "bin"
    npm_bin = LOCAL_CLAUDE_ROOT / "node_modules" / ".bin"
    # Prefer the native executable over npm's wrappers so arguments reach the
    # CLI without another shell layer re-parsing them.
    if platform.system() == "Windows":
        local_candidates = [package_bin / "claude.exe", npm_bin / "claude.cmd"]
    else:
        local_candidates = [package_bin / "claude", npm_bin / "claude"]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate.resolve()
    located = shutil.which("claude")
    if located:
        return Path(located).resolve()
    native_name = "claude.exe" if platform.system() == "Windows" else "claude"
    native_install = Path.home() / ".local" / "bin" / native_name
    return native_install.resolve() if native_install.exists() else None


def _require_claude() -> Path:
    executable = find_claude()
    if not executable:
        raise FileNotFoundError(f"Claude Code CLI가 없습니다. {INSTALLER_NAME}을 실행해주세요.")
    return executable


def _run(arguments: list[str], timeout: int, input_text: str | None = None):
    return subprocess.run(
        _command_prefix(_require_claude()) + arguments,
        cwd=PROJECT_ROOT,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_creation_flags(),
        env=_clean_env(),
    )


def auth_status(timeout: int = 20) -> AuthStatus:
    if not find_claude():
        return AuthStatus(
            available=False,
            logged_in=False,
            detail=f"Claude Code CLI가 없습니다. {INSTALLER_NAME}을 실행해주세요.",
        )
    try:
        result = _run(["auth", "status", "--json"], timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AuthStatus(True, False, f"로그인 상태 확인 실패: {exc}")
    try:
        data = json.loads(result.stdout)
    except ValueError:
        detail = _clean_output(result.stdout or result.stderr).strip()
        return AuthStatus(True, False, detail or "로그인 정보가 없습니다.")
    if not data.get("loggedIn"):
        return AuthStatus(True, False, "Claude에 로그인되어 있지 않습니다.")
    parts = [
        str(data[key])
        for key in ("email", "subscriptionType", "authMethod")
        if data.get(key) and data.get(key) != "none"
    ]
    return AuthStatus(True, True, "Claude 로그인됨" + (f" · {' · '.join(parts)}" if parts else ""))


def login(
    log: LogCallback | None = None,
    cancel_event: Event | None = None,
) -> AuthStatus:
    executable = _require_claude()
    if log:
        log("브라우저에서 Claude 로그인을 완료해주세요. 앱은 OAuth 토큰을 직접 저장하지 않습니다.")
    return_code, stdout, stderr = _run_streaming(
        _command_prefix(executable) + ["auth", "login", "--claudeai"],
        input_text=None,
        log=log,
        cancel_event=cancel_event,
        emit_stderr=True,
    )
    if return_code != 0:
        raise RuntimeError(_last_error(stderr or stdout, "Claude OAuth 로그인에 실패했습니다."))
    return auth_status()


def logout() -> AuthStatus:
    result = _run(["auth", "logout"], timeout=30)
    if result.returncode != 0:
        raise RuntimeError(_last_error(result.stderr, "로그아웃에 실패했습니다."))
    return auth_status()


def fetch_model_catalog(timeout: int = 30) -> list[dict]:
    """Ask Claude Code which models this account can use, via the SDK handshake."""
    request = {"type": "control_request", "request_id": "models", "request": {"subtype": "initialize"}}
    try:
        result = _run(
            [
                "-p",
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
                "--verbose",
                "--no-session-persistence",
                "--safe-mode",
            ],
            timeout,
            input_text=json.dumps(request) + "\n",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Claude 모델 목록 확인 실패: {exc}") from exc
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("type") != "control_response":
            continue
        models = message.get("response", {}).get("response", {}).get("models")
        if isinstance(models, list) and models:
            return models
    raise RuntimeError(_last_error(result.stderr, "Claude 모델 목록을 읽지 못했습니다."))


def generate_text(
    prompt: str,
    *,
    model: str,
    reasoning_effort: str,
    log: LogCallback | None,
    cancel_event: Event | None,
    activity: str,
    failure_message: str,
) -> str:
    executable = _require_claude()
    command = _command_prefix(executable) + [
        "-p",
        "--model",
        model,
        "--output-format",
        "text",
        "--no-session-persistence",
        "--safe-mode",
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--system-prompt",
        NOTE_SYSTEM_PROMPT,
    ]
    if reasoning_effort:
        command += ["--effort", reasoning_effort]
    if log:
        log(f"Claude {model} · effort={reasoning_effort or 'default'}로 {activity}.")
    return_code, stdout, stderr = _run_streaming(
        command,
        input_text=prompt,
        log=log,
        cancel_event=cancel_event,
        emit_stderr=False,
    )
    if return_code != 0:
        message = _last_error(stderr or stdout, failure_message)
        lowered = message.lower()
        if "login" in lowered or "unauthorized" in lowered or "authenticat" in lowered:
            message += "\n연결과 업데이트 화면에서 Claude 로그인을 다시 진행해주세요."
        raise RuntimeError(message)
    final_text = _clean_output(stdout).strip()
    if not final_text:
        raise RuntimeError("Claude 실행은 끝났지만 결과가 비어 있습니다.")
    return final_text
