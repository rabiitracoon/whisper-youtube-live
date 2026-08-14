#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  print "가상환경이 없어 설치를 먼저 시작합니다."
  "$PROJECT_ROOT/scripts/install.sh"
fi
exec .venv/bin/python app.py
