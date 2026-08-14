#!/bin/zsh
set -euo pipefail
PROJECT_ROOT="${0:A:h}"
"$PROJECT_ROOT/scripts/install.sh"
print ""
read "?Enter 키를 누르면 창을 닫습니다. "
