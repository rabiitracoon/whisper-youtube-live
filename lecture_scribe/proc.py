from __future__ import annotations

import subprocess
import sys


def kill_process_tree(process: subprocess.Popen, wait_timeout: float = 10) -> None:
    """Stop a child process and everything it spawned (cmd.exe wrappers on Windows)."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            pass
