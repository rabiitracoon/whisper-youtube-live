#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
cd "$PROJECT_ROOT"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "이 버전은 Apple Silicon(arm64) Mac 전용입니다."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))'; then
        PYTHON_BIN="$candidate"
        break
      fi
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  print -u2 "Python 3.11 또는 3.12가 필요합니다. Homebrew로 python@3.12를 설치해주세요."
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

if command -v npm >/dev/null 2>&1; then
  mkdir -p tools/codex-cli
  npm install --prefix tools/codex-cli @openai/codex@latest
  mkdir -p tools/claude-code
  npm install --prefix tools/claude-code @anthropic-ai/claude-code@latest
else
  print "Node.js/npm을 찾지 못해 Codex CLI와 Claude Code CLI 설치를 건너뜁니다. 전사 기능은 그대로 사용할 수 있습니다."
  print "AI 노트 기능도 쓰려면 'brew install node' 후 install.command를 다시 실행하세요."
fi

.venv/bin/python scripts/verify_install.py
print "설치 완료. run.command를 더블클릭해 실행하세요."
