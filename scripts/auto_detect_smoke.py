from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import av
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from lecture_scribe.audio_edit import AudioEditRequest
from lecture_scribe.ui.audio_editor import AudioEditorDialog
from lecture_scribe.ui.theme import APP_STYLE, configure_fonts
from lecture_scribe.youtube import format_duration


def main() -> int:
    if len(sys.argv) not in {3, 4}:
        print(
            "Usage: python scripts/auto_detect_smoke.py <audio-file> <screenshot.png> [--cancel|--fixture]"
        )
        return 2
    audio_path = Path(sys.argv[1]).resolve()
    screenshot = Path(sys.argv[2]).resolve()
    with av.open(str(audio_path)) as container:
        duration = float(container.duration / av.time_base) if container.duration else 1.0

    app = QApplication([])
    app.setStyle("Fusion")
    configure_fonts(app)
    app.setStyleSheet(APP_STYLE)
    dialog = AudioEditorDialog(
        AudioEditRequest(audio_path, duration, (0.35,) * 12_000, 0, duration)
    )
    dialog.show()
    verify_cancel = len(sys.argv) == 4 and sys.argv[3] == "--cancel"
    verify_fixture = len(sys.argv) == 4 and sys.argv[3] == "--fixture"
    if verify_fixture:
        dialog._auto_previous_state = ((), (), None, ())
        dialog._on_auto_detected(((61.784, 5225.320), (8663.0, 9423.368)))
    else:
        dialog._toggle_auto_detection()
    if verify_cancel:
        dialog.rejected.connect(lambda: (print("Automatic detection cancellation passed."), app.quit()))
        QTimer.singleShot(100, dialog.reject)
        QTimer.singleShot(15_000, lambda: app.exit(6))
        return app.exec()

    def poll() -> None:
        if dialog._auto_thread is not None:
            QTimer.singleShot(200, poll)
            return
        if not dialog.exclusions:
            print("Automatic detection returned no candidates.", file=sys.stderr)
            dialog.close()
            app.exit(3)
            return
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not dialog.grab().save(str(screenshot)):
            dialog.close()
            app.exit(4)
            return
        total = sum(end - start for start, end in dialog.exclusions)
        print(f"Detected {len(dialog.exclusions)} candidate(s), total {format_duration(total)}")
        for start, end in dialog.exclusions:
            print(f"  {format_duration(start)} ~ {format_duration(end)}")
        print(f"Saved automatic detection screenshot: {screenshot}")
        dialog.close()
        app.quit()

    def timeout() -> None:
        if dialog._auto_thread is not None:
            print("Automatic detection timed out.", file=sys.stderr)
            dialog._cancel_auto_detection()
            app.exit(5)

    QTimer.singleShot(200, poll)
    QTimer.singleShot(180_000, timeout)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
