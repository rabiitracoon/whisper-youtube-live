from __future__ import annotations

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


if __name__ == "__main__":
    raise SystemExit(main())
