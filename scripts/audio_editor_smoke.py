from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from lecture_scribe.audio_edit import AudioEditRequest
from lecture_scribe.ui.audio_editor import AudioEditorDialog
from lecture_scribe.ui.theme import APP_STYLE, configure_fonts


def main() -> int:
    if len(sys.argv) not in {3, 4}:
        print("Usage: python scripts/audio_editor_smoke.py <audio-file> <screenshot.png> [--playback]")
        return 2
    audio_path = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()
    peaks = tuple(
        0.08 if 3600 <= index / 12_000 * 10_800 <= 4200 else 0.2 + 0.7 * abs(math.sin(index / 17))
        for index in range(12_000)
    )
    app = QApplication([])
    app.setStyle("Fusion")
    configure_fonts(app)
    app.setStyleSheet(APP_STYLE)
    dialog = AudioEditorDialog(
        AudioEditRequest(audio_path, 10_800, peaks, 300, 10_200)
    )
    dialog.cuts = (3600, 4200, 6500, 7100)
    dialog.exclusions = ((3600, 4200),)
    dialog._select_segment(1)
    dialog._refresh_edit_state()
    dialog.show()

    def wheel(delta: int, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
        viewport = dialog.waveform_scroll.viewport()
        center = QPointF(viewport.width() / 2, viewport.height() / 2)
        event = QWheelEvent(
            center,
            center,
            QPoint(),
            QPoint(0, delta),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(viewport, event)

    def verify_wheel_navigation() -> None:
        dialog._scroll_to_seconds(3900)
        viewport_x = dialog.waveform_scroll.viewport().width() / 2
        bar = dialog.waveform_scroll.horizontalScrollBar()
        before_anchor = (bar.value() + viewport_x) / dialog.waveform.width()
        wheel(120)
        after_anchor = (bar.value() + viewport_x) / dialog.waveform.width()
        if dialog.zoom_slider.value() != 6 or abs(before_anchor - after_anchor) > 0.002:
            print("Mouse-wheel cursor-anchored zoom failed.", file=sys.stderr)
            app.exit(5)
            return
        before_pan = bar.value()
        wheel(-120, Qt.KeyboardModifier.ShiftModifier)
        if bar.value() <= before_pan:
            print("Shift + mouse-wheel horizontal navigation failed.", file=sys.stderr)
            app.exit(6)
            return
        wheel(-120)
        print("Wheel zoom and Shift + wheel horizontal navigation passed.")

    QTimer.singleShot(100, verify_wheel_navigation)
    verify_playback = len(sys.argv) == 4 and sys.argv[3] == "--playback"
    if verify_playback:
        dialog.audio_output.setVolume(0)
        dialog._seek_seconds(300)
        started = [False]

        def start_when_ready() -> None:
            if dialog._media_ready and not started[0]:
                started[0] = True
                dialog.player.play()

        dialog.player.mediaStatusChanged.connect(lambda _status: start_when_ready())
        QTimer.singleShot(500, start_when_ready)

    def capture() -> None:
        dialog._scroll_to_seconds(3900)
        app.processEvents()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not dialog.grab().save(str(target)):
            app.exit(3)
            return
        print(f"Saved audio editor screenshot: {target}")
        if verify_playback and dialog.player.position() <= 300_000:
            print("Audio preview did not advance.", file=sys.stderr)
            app.exit(4)
            return
        if verify_playback:
            print(f"Muted playback advanced to {dialog.player.position()} ms")
        dialog.close()
        app.processEvents()
        if not dialog.player.source().isEmpty():
            print("Media source handle was not released on close.", file=sys.stderr)
            app.exit(7)
            return
        print("Media source handle released on close.")
        app.quit()

    QTimer.singleShot(4000 if verify_playback else 2500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
