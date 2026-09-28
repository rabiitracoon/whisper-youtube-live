from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lecture_scribe.ai_providers import auth_status, generate_notes
from lecture_scribe.prompting import PromptContext, render_prompt


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    status = auth_status("openai")
    print(status.detail)
    if not status.logged_in:
        print("ChatGPT OAuth login is required.", file=sys.stderr)
        return 2
    template = (PROJECT_ROOT / "prompts" / "default_lecture_notes_ko.md").read_text(encoding="utf-8")
    prompt = render_prompt(
        template,
        PromptContext(
            title="머신러닝 수업 필기",
            channel="테스트",
            url="https://example.invalid",
            duration="00:00:15",
            language="ko",
            transcript=(
                "지도 학습은 입력과 정답 쌍에서 패턴을 학습한다. "
                "검증 데이터는 학습에 직접 쓰지 않고 일반화 성능을 확인한다. "
                "교수는 학습 데이터와 검증 데이터를 섞으면 성능 평가를 믿기 어렵다고 강조했다."
            ),
        ),
    )
    with tempfile.TemporaryDirectory(prefix="lecture-scribe-codex-") as directory:
        output = Path(directory) / "notes.md"
        generate_notes(
            prompt,
            output,
            model="gpt-5.6-sol",
            reasoning_effort="high",
            log=print,
        )
        content = output.read_text(encoding="utf-8")
        print("--- result ---")
        print(content)
        if "#" not in content or "지도 학습" not in content or "00:00" in content:
            print("Unexpected Markdown output.", file=sys.stderr)
            return 3
    print("Codex OAuth + GPT-5.6 Sol integration passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
