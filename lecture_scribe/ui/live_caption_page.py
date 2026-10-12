from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QHideEvent, QMouseEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..live_caption import LiveCaption, LiveCaptionOptions
from .workers import LiveCaptionWorker


LIVE_MODELS = [
    ("빠름 · 권장 (large-v3-turbo)", "large-v3-turbo"),
    ("가장 정확 (large-v3, 느림)", "large-v3"),
    ("가벼움 (medium)", "medium"),
    ("Qwen3-ASR 1.7B (선택 엔진)", "qwen3-asr-1.7b"),
]
LANGUAGES = [("한국어", "ko"), ("자동 감지", "auto"), ("영어", "en"), ("일본어", "ja"), ("중국어", "zh")]


class CaptionOverlay(QWidget):
    """A small always-on-top caption strip to keep beside the lecture player."""

    hidden = Signal()

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setWindowTitle("실시간 자막")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(760, 150)
        self._drag_offset: QPoint | None = None
        self._lines: list[str] = []
        panel = QFrame(self)
        panel.setObjectName("CaptionOverlayPanel")
        panel.setStyleSheet(
            "QFrame#CaptionOverlayPanel { background: rgba(18, 20, 30, 215); border-radius: 14px; }"
            "QLabel { color: #F5F6FA; background: transparent; }"
            "QLabel#OverlayPartial { color: #A9AEC0; }"
            "QPushButton { color: #C7CBD7; background: transparent; border: 0; font-size: 15px; }"
            "QSizeGrip { background: transparent; width: 14px; height: 14px; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 8, 12, 12)
        layout.setSpacing(2)
        top = QHBoxLayout()
        hint = QLabel("드래그해서 옮기기 · 오른쪽 아래 모서리로 크기 조절")
        hint.setStyleSheet("color: #6F7487; font-size: 10px;")
        close_button = QPushButton("✕")
        close_button.setFixedSize(22, 22)
        close_button.clicked.connect(self.hide)
        top.addWidget(hint, 1)
        top.addWidget(close_button)
        layout.addLayout(top)
        self.final_label = QLabel("")
        self.final_label.setWordWrap(True)
        self.final_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.final_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.partial_label = QLabel("")
        self.partial_label.setObjectName("OverlayPartial")
        self.partial_label.setWordWrap(True)
        layout.addWidget(self.final_label, 1)
        layout.addWidget(self.partial_label)
        grip_row = QHBoxLayout()
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(panel))
        layout.addLayout(grip_row)
        self.set_font_size(20)

    def set_font_size(self, size: int) -> None:
        font = QFont(self.font())
        font.setPixelSize(size)
        font.setWeight(QFont.Weight.DemiBold)
        self.final_label.setFont(font)
        partial_font = QFont(font)
        partial_font.setWeight(QFont.Weight.Normal)
        partial_font.setPixelSize(max(12, size - 3))
        self.partial_label.setFont(partial_font)

    def add_line(self, text: str) -> None:
        self._lines = (self._lines + [text])[-2:]
        self.final_label.setText("\n".join(self._lines))

    def set_partial(self, text: str) -> None:
        self.partial_label.setText(text)

    def clear(self) -> None:
        self._lines = []
        self.final_label.setText("")
        self.partial_label.setText("")

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self.hidden.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._drag_offset = None


class LiveCaptionPage(QWidget):
    def __init__(
        self,
        settings: AppSettings,
        card_factory: Callable[[str, str | None], tuple[QFrame, QVBoxLayout]],
        field_label: Callable[[str], QLabel],
        launch_worker: Callable[[object], None],
        cookie_browser: Callable[[], str],
    ) -> None:
        super().__init__()
        self.settings = settings
        self._launch_worker = launch_worker
        self._cookie_browser = cookie_browser
        self.worker: LiveCaptionWorker | None = None
        self.output_folder: Path | None = None
        self.overlay = CaptionOverlay()
        self.overlay.hidden.connect(lambda: self.overlay_check.setChecked(False))
        self._unseen = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(14)

        setup_card, setup = card_factory(
            "라이브 수업 실시간 자막",
            "방송 오디오를 인터넷에서 직접 받아 자막을 만듭니다. 스피커나 스테레오 믹스가 "
            "없어도 되고, 브라우저 영상은 음소거해도 됩니다. 딴생각을 했다면 아래 기록을 "
            "거슬러 올라가 읽으세요.",
        )
        setup.addWidget(field_label("라이브 방송 링크"))
        self.url_input = QLineEdit(settings.live_url)
        self.url_input.setPlaceholderText(
            "YouTube 라이브 링크 (연습: 녹화 영상 링크나 파일 경로도 가능)"
        )
        self.url_input.setClearButtonEnabled(True)
        self.url_input.setAccessibleName("실시간 자막을 만들 라이브 방송 링크")
        paste_button = QPushButton("붙여넣기")
        paste_button.clicked.connect(
            lambda: self.url_input.setText(QApplication.clipboard().text().strip())
        )
        url_row = QHBoxLayout()
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(paste_button)
        setup.addLayout(url_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        self.model_combo = QComboBox()
        for label, value in LIVE_MODELS:
            self.model_combo.addItem(label, value)
        self._select(self.model_combo, settings.live_model)
        self.language_combo = QComboBox()
        for label, value in LANGUAGES:
            self.language_combo.addItem(label, value)
        self._select(self.language_combo, settings.live_language)
        self.terms_input = QLineEdit(settings.live_terms)
        self.terms_input.setPlaceholderText("예: 미분방정식, 라플라스 변환, 고유값")
        self.terms_input.setToolTip("쉼표로 구분한 단어를 Whisper 문맥에 넣어 잘못 듣는 일을 줄입니다.")
        grid.addWidget(field_label("음성 인식 모델"), 0, 0)
        grid.addWidget(self.model_combo, 1, 0)
        grid.addWidget(field_label("수업 언어"), 0, 1)
        grid.addWidget(self.language_combo, 1, 1)
        grid.addWidget(field_label("자주 나오는 전문용어 (선택)"), 2, 0, 1, 2)
        grid.addWidget(self.terms_input, 3, 0, 1, 2)
        setup.addLayout(grid)

        action_row = QHBoxLayout()
        self.status_label = QLabel("대기 중")
        self.status_label.setObjectName("HelperText")
        self.status_label.setWordWrap(True)
        self.overlay_check = QCheckBox("항상 위 자막 창 띄우기")
        self.overlay_check.setToolTip("강의 화면 위에 겹쳐 둘 수 있는 작은 자막 창을 엽니다.")
        self.overlay_check.toggled.connect(self._toggle_overlay)
        self.start_button = QPushButton("자막 시작")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("자막 중지")
        self.stop_button.setObjectName("DangerButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        action_row.addWidget(self.status_label, 1)
        action_row.addWidget(self.overlay_check)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.start_button)
        setup.addLayout(action_row)
        layout.addWidget(setup_card)

        caption_card, captions = card_factory("자막 기록", None)
        caption_card.setProperty("cardRole", "progress")
        tool_row = QHBoxLayout()
        tool_row.addWidget(QLabel("글자 크기"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(12, 40)
        self.font_spin.setValue(settings.live_font_size)
        self.font_spin.valueChanged.connect(self._apply_font_size)
        tool_row.addWidget(self.font_spin)
        tool_row.addStretch(1)
        self.jump_button = QPushButton("최신 자막으로 ↓")
        self.jump_button.setVisible(False)
        self.jump_button.clicked.connect(self._scroll_to_latest)
        self.copy_button = QPushButton("전체 복사")
        self.copy_button.clicked.connect(
            lambda: QApplication.clipboard().setText(self.caption_view.toPlainText())
        )
        self.folder_button = QPushButton("저장 폴더 열기")
        self.folder_button.setEnabled(False)
        self.folder_button.clicked.connect(self._open_folder)
        tool_row.addWidget(self.jump_button)
        tool_row.addWidget(self.copy_button)
        tool_row.addWidget(self.folder_button)
        captions.addLayout(tool_row)
        self.caption_view = QPlainTextEdit()
        self.caption_view.setReadOnly(True)
        self.caption_view.setAccessibleName("지금까지의 실시간 자막")
        self.caption_view.setPlaceholderText(
            "자막이 시간과 함께 여기에 쌓입니다. 위로 올려 읽는 동안에는 자동으로 내려가지 않아요."
        )
        self.caption_view.verticalScrollBar().valueChanged.connect(self._on_scroll)
        captions.addWidget(self.caption_view, 1)
        self.partial_label = QLabel("")
        self.partial_label.setStyleSheet("color: #6E7385; font-size: 13px;")
        self.partial_label.setWordWrap(True)
        captions.addWidget(self.partial_label)
        layout.addWidget(caption_card, 1)
        self._apply_font_size(self.font_spin.value())

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @property
    def running(self) -> bool:
        return self.worker is not None

    def start(self) -> None:
        source = self.url_input.text().strip()
        if not source:
            QMessageBox.warning(self, "링크 필요", "실시간 자막을 만들 라이브 방송 링크를 입력해주세요.")
            return
        terms = tuple(
            item.strip() for item in self.terms_input.text().replace("\n", ",").split(",")
            if item.strip()
        )
        self.settings.live_url = source
        self.settings.live_model = str(self.model_combo.currentData())
        self.settings.live_language = str(self.language_combo.currentData())
        self.settings.live_terms = self.terms_input.text().strip()
        self.settings.live_font_size = self.font_spin.value()
        self.settings.save()
        options = LiveCaptionOptions(
            model_name=self.settings.live_model,
            language=self.settings.live_language,
            compute_type=self.settings.compute_type,
            prompt_terms=terms,
        )
        output_root = Path(self.settings.output_dir).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        if self.caption_view.toPlainText().strip():
            self.caption_view.appendPlainText("\n──────── 새 자막 세션 ────────\n")
        self.overlay.clear()
        worker = LiveCaptionWorker(source, options, output_root, self._cookie_browser())
        self.worker = worker
        worker.caption.connect(self._on_caption)
        worker.partial.connect(self._on_partial)
        worker.status.connect(self.status_label.setText)
        worker.log.connect(self._on_log)
        worker.result.connect(self._on_result)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        for widget in (self.url_input, self.model_combo, self.language_combo, self.terms_input):
            widget.setEnabled(False)
        self.status_label.setText("시작하는 중")
        self._launch_worker(worker)

    def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.status_label.setText("남은 문장을 정리하고 멈추는 중")

    def _on_caption(self, caption: LiveCaption) -> None:
        scrollbar = self.caption_view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        self.caption_view.appendPlainText(f"[{caption.wall_time:%H:%M:%S}]  {caption.text}")
        if at_bottom:
            self._scroll_to_latest()
        else:
            # The user is reading back; do not yank the view down.
            self._unseen += 1
            self.jump_button.setText(f"최신 자막으로 ↓ ({self._unseen})")
            self.jump_button.setVisible(True)
        self.overlay.add_line(caption.text)

    def _on_partial(self, text: str) -> None:
        self.partial_label.setText(f"듣는 중… {text}" if text else "")
        self.overlay.set_partial(text)

    def _on_scroll(self, value: int) -> None:
        if value >= self.caption_view.verticalScrollBar().maximum() - 4:
            self._unseen = 0
            self.jump_button.setVisible(False)

    def _scroll_to_latest(self) -> None:
        self.caption_view.moveCursor(QTextCursor.MoveOperation.End)
        scrollbar = self.caption_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_log(self, message: str) -> None:
        if message.startswith("자막 저장 위치: "):
            self.output_folder = Path(message.removeprefix("자막 저장 위치: "))
            self.folder_button.setEnabled(True)
        self.status_label.setToolTip(message)

    def _on_result(self, folder: object) -> None:
        self.output_folder = Path(str(folder))
        self.folder_button.setEnabled(True)
        self.status_label.setText(f"자막을 멈췄습니다 · 저장됨: {self.output_folder.name}")

    def _on_error(self, message: str) -> None:
        self.status_label.setText("오류로 멈췄습니다")
        QMessageBox.critical(self, "실시간 자막을 시작하지 못했습니다", message)

    def _on_finished(self) -> None:
        self.worker = None
        self.partial_label.setText("")
        self.overlay.set_partial("")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for widget in (self.url_input, self.model_combo, self.language_combo, self.terms_input):
            widget.setEnabled(True)

    def _apply_font_size(self, size: int) -> None:
        font = QFont(self.caption_view.font())
        font.setPixelSize(size)
        self.caption_view.setFont(font)
        self.overlay.set_font_size(size + 4)
        if self.settings.live_font_size != size:
            self.settings.live_font_size = size
            self.settings.save()

    def _toggle_overlay(self, visible: bool) -> None:
        if visible:
            screen = QApplication.primaryScreen()
            if screen is not None and not self.overlay.property("placed"):
                area = screen.availableGeometry()
                self.overlay.move(
                    area.center().x() - self.overlay.width() // 2,
                    area.bottom() - self.overlay.height() - 40,
                )
                self.overlay.setProperty("placed", True)
            self.overlay.show()
        else:
            self.overlay.hide()

    def _open_folder(self) -> None:
        if self.output_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_folder)))

    def shutdown(self) -> None:
        self.overlay.close()
