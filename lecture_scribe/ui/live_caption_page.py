from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QGuiApplication,
    QHideEvent,
    QMouseEvent,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings
from ..live_caption import LiveCaption, LiveCaptionOptions
from .theme import COLORS, FONT_STACK, MONO_STACK
from .workers import LiveCaptionWorker


LIVE_MODELS = [
    ("빠름 · 권장 (large-v3-turbo)", "large-v3-turbo"),
    ("가장 정확 (large-v3, 느림)", "large-v3"),
    ("가벼움 (medium)", "medium"),
    ("Qwen3-ASR 1.7B (선택 엔진)", "qwen3-asr-1.7b"),
]
LANGUAGES = [("한국어", "ko"), ("자동 감지", "auto"), ("영어", "en"), ("일본어", "ja"), ("중국어", "zh")]

DEFAULT_FONT_SIZE = 20
MAX_CAPTION_BLOCKS = 20000


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
        self.resize(760, 160)
        self._drag_offset: QPoint | None = None
        self._lines: list[str] = []
        panel = QFrame(self)
        panel.setObjectName("CaptionOverlayPanel")
        panel.setStyleSheet(
            f"QFrame#CaptionOverlayPanel {{ background: {COLORS['overlay_bg']}; border-radius: 12px; }}"
            f"QLabel {{ color: {COLORS['editor_text']}; background: transparent; }}"
            f"QLabel#OverlayHint {{ color: {COLORS['editor_text_tertiary']}; font-size: 11px; }}"
            f"QPushButton {{ color: {COLORS['editor_text_secondary']}; background: transparent;"
            f" border: 0; border-radius: 6px; font-size: 13px; padding: 0; min-height: 0; }}"
            f"QPushButton:hover {{ background: {COLORS['editor_elevated']};"
            f" color: {COLORS['editor_text']}; }}"
            "QSizeGrip { background: transparent; width: 14px; height: 14px; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 8, 10, 10)
        layout.setSpacing(2)
        top = QHBoxLayout()
        hint = QLabel("드래그해서 옮기기 · 오른쪽 아래 모서리로 크기 조절")
        hint.setObjectName("OverlayHint")
        close_button = QPushButton("✕")
        close_button.setProperty("compact", True)
        close_button.setFixedSize(24, 24)
        close_button.setToolTip("자막 창 닫기")
        close_button.clicked.connect(self.hide)
        top.addWidget(hint, 1)
        top.addWidget(close_button)
        layout.addLayout(top)
        self.final_label = QLabel("")
        self.final_label.setWordWrap(True)
        self.final_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.final_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.partial_label = QLabel("")
        self.partial_label.setWordWrap(True)
        layout.addWidget(self.final_label, 1)
        layout.addWidget(self.partial_label)
        grip_row = QHBoxLayout()
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(panel))
        layout.addLayout(grip_row)
        self.set_font_size(DEFAULT_FONT_SIZE + 4)

    def set_font_size(self, size: int) -> None:
        # The app-wide stylesheet wins over QWidget.setFont, so size goes in a per-widget QSS.
        self.final_label.setStyleSheet(
            f"font-size: {size}px; font-weight: 600; color: {COLORS['editor_text']};"
        )
        self.partial_label.setStyleSheet(
            f"font-size: {max(12, size - 4)}px; font-style: italic;"
            f" color: {COLORS['editor_text_secondary']};"
        )

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


def _button(text: str, name: str | None = None, compact: bool = False) -> QPushButton:
    button = QPushButton(text)
    if name:
        button.setObjectName(name)
    if compact:
        button.setProperty("compact", True)
    return button


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
        self._field_label = field_label
        self.worker: LiveCaptionWorker | None = None
        self.output_folder: Path | None = None
        self.overlay = CaptionOverlay()
        self.overlay.hidden.connect(self._on_overlay_hidden)
        self._unseen = 0
        self._follow_latest = True
        self._received_captions = False
        self._font_size = DEFAULT_FONT_SIZE

        self._font_save_timer = QTimer(self)
        self._font_save_timer.setSingleShot(True)
        self._font_save_timer.setInterval(600)
        self._font_save_timer.timeout.connect(self._save_font_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_setup_card())
        layout.addWidget(self._build_reading_card(), 1)

        self._set_status("대기 중", "neutral")
        self._show_empty_state(running=False)
        self._apply_font_size(self.font_spin.value())

    # ---- construction -------------------------------------------------

    def _new_card(self) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)
        return card, layout

    def _build_setup_card(self) -> QFrame:
        settings = self.settings
        card, setup = self._new_card()
        title = QLabel("라이브 수업 실시간 자막")
        title.setObjectName("SectionTitle")
        helper = QLabel(
            "방송 오디오를 직접 받아 자막을 만들어요. 영상은 음소거해도 괜찮고, 딴생각했다면 아래 기록을 위로 올려 읽으세요."
        )
        helper.setObjectName("HelperText")
        helper.setWordWrap(True)
        setup.addWidget(title)
        setup.addWidget(helper)

        self.url_input = QLineEdit(settings.live_url)
        self.url_input.setPlaceholderText(
            "YouTube 라이브 링크 (연습: 녹화 영상 링크나 파일 경로도 가능)"
        )
        self.url_input.setClearButtonEnabled(True)
        self.url_input.setAccessibleName("실시간 자막을 만들 라이브 방송 링크")
        self.url_input.returnPressed.connect(self._start_if_idle)
        paste_button = _button("붙여넣기", "GhostButton")
        paste_button.clicked.connect(
            lambda: self.url_input.setText(QApplication.clipboard().text().strip())
        )
        self.start_button = _button("자막 시작", "PrimaryButton")
        self.start_button.clicked.connect(self.start)
        self.stop_button = _button("자막 중지", "DangerButton")
        self.stop_button.setEnabled(False)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop)
        url_row = QHBoxLayout()
        url_row.setSpacing(8)
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(paste_button)
        url_row.addWidget(self.start_button)
        url_row.addWidget(self.stop_button)
        setup.addLayout(url_row)

        self.model_combo = QComboBox()
        for label, value in LIVE_MODELS:
            self.model_combo.addItem(label, value)
        self._select(self.model_combo, settings.live_model)
        self.language_combo = QComboBox()
        for label, value in LANGUAGES:
            self.language_combo.addItem(label, value)
        self._select(self.language_combo, settings.live_language)
        self.language_combo.setMinimumWidth(130)
        option_row = QHBoxLayout()
        option_row.setSpacing(12)
        model_col = QVBoxLayout()
        model_col.setSpacing(4)
        model_col.addWidget(self._field_label("음성 인식 모델"))
        model_col.addWidget(self.model_combo)
        language_col = QVBoxLayout()
        language_col.setSpacing(4)
        language_col.addWidget(self._field_label("수업 언어"))
        language_col.addWidget(self.language_combo)
        option_row.addLayout(model_col, 3)
        option_row.addLayout(language_col, 1)
        self.advanced_button = _button("고급 설정 ▸", "GhostButton", compact=True)
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        option_row.addWidget(self.advanced_button, 0, Qt.AlignmentFlag.AlignBottom)
        setup.addLayout(option_row)

        self.advanced_box = QWidget()
        advanced = QVBoxLayout(self.advanced_box)
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setSpacing(4)
        self.terms_input = QLineEdit(settings.live_terms)
        self.terms_input.setPlaceholderText("예: 미분방정식, 라플라스 변환, 고유값")
        self.terms_input.setToolTip("쉼표로 구분한 단어를 Whisper 문맥에 넣어 잘못 듣는 일을 줄입니다.")
        advanced.addWidget(self._field_label("자주 나오는 전문용어 (선택)"))
        advanced.addWidget(self.terms_input)
        self.advanced_box.setVisible(False)
        setup.addWidget(self.advanced_box)
        if settings.live_terms.strip():
            self.advanced_button.setChecked(True)
        return card

    def _build_reading_card(self) -> QFrame:
        card, captions = self._new_card()
        captions.setSpacing(8)

        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        self.status_pill = QLabel("대기 중")
        self.status_pill.setObjectName("StatusPill")
        self.status_label = QLabel("")
        self.status_label.setObjectName("HelperText")
        status_row.addWidget(self.status_pill)
        status_row.addWidget(self.status_label, 1)
        captions.addLayout(status_row)

        # Slim inline banner for the model download.
        self.model_banner = QWidget()
        banner = QHBoxLayout(self.model_banner)
        banner.setContentsMargins(0, 0, 0, 0)
        banner.setSpacing(10)
        self.model_progress_label = QLabel("모델 준비 중")
        self.model_progress_label.setObjectName("HelperText")
        self.model_progress = QProgressBar()
        self.model_progress.setAccessibleName("모델 준비 진행률")
        self.model_progress.setTextVisible(False)
        self.model_progress.setFixedHeight(6)
        banner.addWidget(self.model_progress_label)
        banner.addWidget(self.model_progress, 1)
        self.model_banner.setVisible(False)
        captions.addWidget(self.model_banner)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        font_caption = QLabel("글자 크기")
        font_caption.setObjectName("HelperText")
        self.font_spin = QSpinBox()
        self.font_spin.setRange(12, 48)
        self.font_spin.setValue(max(12, min(48, settings_font(self.settings))))
        self.font_spin.setSuffix(" px")
        self.font_spin.setAccessibleName("자막 글자 크기")
        self.font_spin.valueChanged.connect(self._apply_font_size)
        self.copy_button = _button("전체 복사", "GhostButton")
        self.copy_button.clicked.connect(
            lambda: QApplication.clipboard().setText(self.caption_view.toPlainText())
        )
        self.folder_button = _button("저장 폴더 열기", "GhostButton")
        self.folder_button.setEnabled(False)
        self.folder_button.clicked.connect(self._open_folder)
        self.overlay_button = _button("자막 창 띄우기")
        self.overlay_button.setCheckable(True)
        self.overlay_button.setToolTip("강의 화면 위에 겹쳐 둘 수 있는 작은 자막 창을 엽니다.")
        self.overlay_button.toggled.connect(self._toggle_overlay)
        tool_row.addWidget(font_caption)
        tool_row.addWidget(self.font_spin)
        tool_row.addStretch(1)
        tool_row.addWidget(self.copy_button)
        tool_row.addWidget(self.folder_button)
        tool_row.addWidget(self.overlay_button)
        captions.addLayout(tool_row)

        self.view_stack = QStackedWidget()
        self.view_stack.setMinimumHeight(180)
        self.empty_state = QWidget()
        empty = QVBoxLayout(self.empty_state)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setSpacing(6)
        self.empty_title = QLabel("")
        self.empty_title.setObjectName("SectionTitle")
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_helper = QLabel("")
        self.empty_helper.setObjectName("HelperText")
        self.empty_helper.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_helper.setWordWrap(True)
        empty.addWidget(self.empty_title)
        empty.addWidget(self.empty_helper)

        self.caption_view = QPlainTextEdit()
        self.caption_view.setReadOnly(True)
        self.caption_view.setFrameShape(QFrame.Shape.NoFrame)
        self.caption_view.setMaximumBlockCount(MAX_CAPTION_BLOCKS)
        self.caption_view.setAccessibleName("지금까지의 실시간 자막")
        self.caption_view.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.view_stack.addWidget(self.empty_state)
        self.view_stack.addWidget(self.caption_view)
        captions.addWidget(self.view_stack, 1)

        # Floating chip, shown only when the reader has scrolled up.
        self.jump_button = _button("최신 자막으로 ↓", "SecondaryButton")
        self.jump_button.setParent(self.view_stack)
        self.jump_button.setVisible(False)
        self.jump_button.clicked.connect(self._jump_to_latest)
        self.view_stack.installEventFilter(self)

        self.partial_label = QLabel("")
        self.partial_label.setWordWrap(True)
        self.partial_label.setStyleSheet(
            f"color: {COLORS['text_tertiary']}; font-style: italic; font-size: 14px;"
            " background: transparent;"
        )
        self.partial_label.setVisible(False)
        captions.addWidget(self.partial_label)
        return card

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @property
    def running(self) -> bool:
        return self.worker is not None

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self.view_stack and event.type() == QEvent.Type.Resize:
            self._place_jump_button()
        return super().eventFilter(watched, event)

    def _place_jump_button(self) -> None:
        button = self.jump_button
        button.adjustSize()
        x = (self.view_stack.width() - button.width()) // 2
        y = self.view_stack.height() - button.height() - 14
        button.move(max(0, x), max(0, y))
        button.raise_()

    def _toggle_advanced(self, shown: bool) -> None:
        self.advanced_box.setVisible(shown)
        self.advanced_button.setText("고급 설정 ▾" if shown else "고급 설정 ▸")

    def _set_status(self, text: str, tone: str) -> None:
        self.status_pill.setText(text)
        self.status_pill.setProperty("tone", tone)
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)

    def _show_empty_state(self, running: bool) -> None:
        if running:
            self.empty_title.setText("자막을 기다리고 있어요")
            self.empty_helper.setText(
                "방송 소리를 듣고 있어요. 첫 문장이 끝나면 여기에 시간과 함께 나타납니다."
            )
        else:
            self.empty_title.setText("아직 시작된 자막이 없어요")
            self.empty_helper.setText(
                "위에 라이브 방송 링크를 넣고 ‘자막 시작’을 누르면, 말한 내용이 여기에 차곡차곡 쌓입니다."
            )

    # ---- run control ----------------------------------------------------

    def _start_if_idle(self) -> None:
        if self.worker is None:
            self.start()

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
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._set_status("오류", "danger")
            self.status_label.setText("저장 폴더를 만들지 못했어요")
            QMessageBox.critical(
                self,
                "저장 폴더를 만들지 못했습니다",
                f"자막을 저장할 폴더를 만들 수 없습니다. 설정에서 저장 위치를 확인해주세요.\n{exc}",
            )
            return
        if self.caption_view.toPlainText().strip():
            self._append_separator()
        self.overlay.clear()
        self._received_captions = False
        worker = LiveCaptionWorker(source, options, output_root, self._cookie_browser())
        self.worker = worker
        worker.caption.connect(self._on_caption)
        worker.partial.connect(self._on_partial)
        worker.status.connect(self._on_status)
        worker.progress.connect(self._on_model_progress)
        worker.log.connect(self._on_log)
        worker.result.connect(self._on_result)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        self.start_button.setEnabled(False)
        self.start_button.setVisible(False)
        self.stop_button.setVisible(True)
        self.stop_button.setEnabled(True)
        for widget in (
            self.url_input,
            self.model_combo,
            self.language_combo,
            self.terms_input,
            self.advanced_button,
        ):
            widget.setEnabled(False)
        self._set_status("연결 중", "accent")
        self.status_label.setText("시작하는 중")
        if self.view_stack.currentWidget() is self.empty_state:
            self._show_empty_state(running=True)
        self._launch_worker(worker)

    def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.status_label.setText("남은 문장을 정리하고 멈추는 중")

    # ---- worker slots ---------------------------------------------------

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)
        if self.worker is None:
            return
        if text == "자막 생성 중":
            self._set_status("● 실시간", "success")
        elif "다시 연결" in text:
            self._set_status("재연결 중", "warning")
        elif self.status_pill.property("tone") != "accent":
            self._set_status("연결 중", "accent")

    def _document_cursor(self) -> QTextCursor:
        cursor = QTextCursor(self.caption_view.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        return cursor

    def _stamp_font(self) -> QFont:
        font = QFont(MONO_STACK.split(",")[0].strip('" '))
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPixelSize(max(11, round(self._font_size * 0.62)))
        return font

    def _block_format(self) -> QTextBlockFormat:
        fmt = QTextBlockFormat()
        fmt.setLineHeight(150, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        fmt.setBottomMargin(10)
        return fmt

    def _begin_block(self, cursor: QTextCursor) -> None:
        fmt = self._block_format()
        if self.caption_view.document().isEmpty():
            cursor.setBlockFormat(fmt)
        else:
            cursor.insertBlock(fmt)

    def _append_caption_line(self, stamp: str, text: str) -> None:
        cursor = self._document_cursor()
        self._begin_block(cursor)
        stamp_format = QTextCharFormat()
        stamp_format.setForeground(QColor(COLORS["text_tertiary"]))
        stamp_format.setFont(self._stamp_font())
        text_format = QTextCharFormat()
        text_format.setForeground(QColor(COLORS["text"]))
        cursor.insertText(stamp, stamp_format)
        cursor.insertText(" ", text_format)
        cursor.insertText(text, text_format)

    def _append_separator(self) -> None:
        cursor = self._document_cursor()
        self._begin_block(cursor)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(COLORS["text_tertiary"]))
        cursor.insertText("──────── 새 자막 세션 ────────", fmt)

    def _on_caption(self, caption: LiveCaption) -> None:
        self._received_captions = True
        self.view_stack.setCurrentWidget(self.caption_view)
        self._append_caption_line(f"{caption.wall_time:%H:%M:%S}", caption.text)
        if self._follow_latest:
            self._scroll_to_latest()
        else:
            # The user is reading back; do not yank the view down.
            self._unseen += 1
            self.jump_button.setText(f"최신 자막으로 ↓ ({self._unseen})")
            self.jump_button.setVisible(True)
            self._place_jump_button()
        self.overlay.add_line(caption.text)

    def _on_model_progress(self, visible: bool, fraction: object) -> None:
        bar = self.model_progress
        self.model_banner.setVisible(visible)
        if fraction is None:
            # Busy indicator: the amount of work is unknown (checking, loading into memory).
            bar.setRange(0, 0)
            self.model_progress_label.setText("모델 준비 중")
        else:
            bar.setRange(0, 1000)
            bar.setValue(int(float(fraction) * 1000))
            self.model_progress_label.setText(f"모델 내려받는 중 · {float(fraction) * 100:.0f}%")

    def _on_partial(self, text: str) -> None:
        self.partial_label.setText(f"듣는 중…  {text}" if text else "")
        self.partial_label.setVisible(bool(text))
        self.overlay.set_partial(text)

    def _on_scroll(self, value: int) -> None:
        bar = self.caption_view.verticalScrollBar()
        self._follow_latest = value >= bar.maximum() - 4
        if self._follow_latest:
            self._unseen = 0
            self.jump_button.setVisible(False)

    def _scroll_to_latest(self) -> None:
        bar = self.caption_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _jump_to_latest(self) -> None:
        self._follow_latest = True
        self._scroll_to_latest()

    def _on_log(self, message: str) -> None:
        if message.startswith("자막 저장 위치: "):
            self.output_folder = Path(message.removeprefix("자막 저장 위치: "))
            self.folder_button.setEnabled(True)
        self.status_label.setToolTip(message)

    def _on_result(self, folder: object) -> None:
        self.output_folder = Path(str(folder))
        self.folder_button.setEnabled(True)
        self._set_status("대기 중", "neutral")
        self.status_label.setText(f"자막을 멈췄어요 · 저장됨: {self.output_folder.name}")

    def _on_error(self, message: str) -> None:
        self._set_status("오류", "danger")
        self.status_label.setText("오류로 멈췄어요")
        title = (
            "자막이 멈췄습니다" if self._received_captions else "실시간 자막을 시작하지 못했습니다"
        )
        QMessageBox.critical(self, title, message)

    def _on_finished(self) -> None:
        self.worker = None
        self.model_banner.setVisible(False)
        self.partial_label.setText("")
        self.partial_label.setVisible(False)
        self.overlay.set_partial("")
        self.start_button.setVisible(True)
        self.start_button.setEnabled(True)
        self.stop_button.setVisible(False)
        self.stop_button.setEnabled(False)
        for widget in (
            self.url_input,
            self.model_combo,
            self.language_combo,
            self.terms_input,
            self.advanced_button,
        ):
            widget.setEnabled(True)
        if self.status_pill.property("tone") not in {"danger", "neutral"}:
            self._set_status("대기 중", "neutral")
        if not self._received_captions:
            self._show_empty_state(running=False)

    # ---- font size / overlay -------------------------------------------

    def _apply_font_size(self, size: int) -> None:
        # Pixel size goes through a per-widget stylesheet: the app-wide QSS overrides setFont().
        self.caption_view.setStyleSheet(
            "QPlainTextEdit {"
            f" font-family: {FONT_STACK}; font-size: {size}px;"
            f" color: {COLORS['text']}; background: {COLORS['surface']};"
            f" selection-background-color: {COLORS['accent_subtle']};"
            f" selection-color: {COLORS['text']};"
            " border: 0; padding: 8px 6px; }"
        )
        self.overlay.set_font_size(size + 4)
        if size != self._font_size:
            self._font_size = size
        if self.settings.live_font_size != size:
            self.settings.live_font_size = size
            self._font_save_timer.start()
        if self._follow_latest:
            QTimer.singleShot(0, self._scroll_to_latest)

    def _save_font_size(self) -> None:
        self.settings.save()

    def _toggle_overlay(self, visible: bool) -> None:
        self.overlay_button.setText("자막 창 닫기" if visible else "자막 창 띄우기")
        if visible:
            window = self.window()
            screen = (window.screen() if window is not None else None) or (
                QGuiApplication.screenAt(self.mapToGlobal(self.rect().center()))
            ) or QApplication.primaryScreen()
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

    def _on_overlay_hidden(self) -> None:
        if self.overlay_button.isChecked():
            self.overlay_button.setChecked(False)

    def _open_folder(self) -> None:
        if self.output_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_folder)))

    def shutdown(self) -> None:
        if self._font_save_timer.isActive():
            self._font_save_timer.stop()
            self.settings.save()
        self.overlay.close()


def settings_font(settings: AppSettings) -> int:
    """Stored size, with the larger reading default for installs that never changed it."""
    size = int(settings.live_font_size or DEFAULT_FONT_SIZE)
    return DEFAULT_FONT_SIZE if size == 17 else size
