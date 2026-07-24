from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QScrollArea

from lecture_scribe.ui.main_window import MainWindow
from lecture_scribe.ui.theme import APP_STYLE, configure_fonts
from lecture_scribe.quality import TranscriptReview


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print(
            "Usage: python scripts/ui_smoke.py <screenshot.png> "
            "[--review|--advanced|--prompt|--tools|--compact]"
        )
        return 2
    app = QApplication([])
    app.setStyle("Fusion")
    configure_fonts(app)
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    if len(sys.argv) == 3 and sys.argv[2] == "--review":
        sample_path = PROJECT_ROOT / "artifacts" / "sample_transcript.txt"
        window._on_transcript_review_ready(
            TranscriptReview(
                text_path=sample_path,
                preview_text=(
                    "오늘은 운영체제의 가상 메모리를 살펴보겠습니다.\n"
                    "페이지는 가상 주소 공간을 고정된 크기로 나눈 단위입니다.\n"
                    "페이지 폴트가 발생하면 운영체제가 필요한 페이지를 디스크에서 메모리로 가져옵니다.\n"
                    "이 과정에서 지역성의 원리가 성능에 중요한 역할을 합니다."
                ),
                status="warning",
                headline="전문용어를 한 번 확인해주세요",
                metrics="선택한 강의 02:00:00 · 842문장 · 단어 12,430개\n찾은 언어 한국어 (94%) · 말소리가 있는 시간 71%",
                warnings=("고유명사와 전문용어를 특히 확인해주세요.",),
            )
        )
        window.review_preview.setFocus()
    elif len(sys.argv) == 3 and sys.argv[2] == "--advanced":
        window.advanced_mode_check.setChecked(True)
    elif len(sys.argv) == 3 and sys.argv[2] == "--prompt":
        window._switch_page(1)
    elif len(sys.argv) == 3 and sys.argv[2] == "--tools":
        window._switch_page(2)
    elif len(sys.argv) == 3 and sys.argv[2] == "--compact":
        window.resize(1080, 720)

    def capture() -> None:
        if len(sys.argv) == 3 and sys.argv[2] == "--review":
            for scroll_area in window.findChildren(QScrollArea):
                scroll_area.ensureWidgetVisible(window.review_card, 20, 20)
            app.processEvents()
        elif len(sys.argv) == 3 and sys.argv[2] == "--advanced":
            for scroll_area in window.findChildren(QScrollArea):
                scroll_area.ensureWidgetVisible(window.advanced_panel, 20, 20)
            app.processEvents()
        elif len(sys.argv) == 3 and sys.argv[2] in {"--prompt", "--tools"}:
            app.processEvents()
        elif len(sys.argv) == 3 and sys.argv[2] == "--compact":
            app.processEvents()
        if len(sys.argv) == 3 and sys.argv[2] in {"--prompt", "--tools"}:
            expected = 1 if sys.argv[2] == "--prompt" else 2
            if window.tabs.currentIndex() != expected or not window.nav_buttons[expected].isChecked():
                print("Sidebar navigation state did not match the visible page.", file=sys.stderr)
                app.exit(4)
                return
        target = Path(sys.argv[1]).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(target)):
            print(f"Failed to save {target}")
            app.exit(3)
            return
        print(f"Saved UI screenshot: {target}")
        window.close()
        app.quit()

    QTimer.singleShot(3500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
