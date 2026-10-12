from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from lecture_scribe.ui.main_window import MainWindow
from lecture_scribe.ui.theme import APP_STYLE, configure_fonts


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Lecture Scribe")
    app.setOrganizationName("Local Tools")
    app.setStyle("Fusion")
    configure_fonts(app)
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


def exit_now(code: int) -> None:
    """End the process even if background work is still running.

    Model downloads and loads, yt-dlp and ffmpeg cannot always be interrupted, and
    the Hugging Face download pool is joined at interpreter shutdown, so a normal
    exit could leave an invisible process running long after the window closed.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(code)


if __name__ == "__main__":
    exit_now(main())
