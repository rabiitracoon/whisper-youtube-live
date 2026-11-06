from __future__ import annotations

import copy
import platform
from functools import partial
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, QThread, QTimer, QUrl, Qt, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QIcon, QKeySequence, QPixmap, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from ..ai_providers import (
    PROVIDER_CHOICES,
    PROVIDER_NAMES,
    TOOL_NAMES,
    ModelCatalog,
    fallback_catalog,
    find_tool,
    normalize_provider,
)
from ..codex_client import AuthStatus
from ..config import AppSettings, load_prompt, save_prompt
from ..diagnostics import get_gpu_info
from ..paths import DEFAULT_PROMPT_PATH
from ..pipeline import PipelineResult
from ..prompting import validate_prompt
from ..quality import TranscriptReview
from ..time_range import format_timecode, parse_timecode, validate_clip_range
from ..updater import UpdateResult, current_version
from ..local_media import resolve_local_media
from ..youtube import is_youtube_url
from .audio_editor import AudioEditorDialog
from .live_caption_page import LiveCaptionPage
from .workers import AuthWorker, ModelListWorker, PipelineWorker, UpdateWorker


ASSET_DIR = Path(__file__).resolve().parent / "assets"
IS_WINDOWS = platform.system() == "Windows"
INSTALLER_NAME = "install.bat" if IS_WINDOWS else "install.command"
ACCELERATOR_DESCRIPTION = (
    "NVIDIA 그래픽카드로 강의 음성을 빠르게 글로 옮깁니다."
    if IS_WINDOWS
    else "Apple Silicon의 GPU로 강의 음성을 빠르게 글로 옮깁니다."
)
VIDEO_ENCODER_NAME = "NVIDIA NVENC" if IS_WINDOWS else "Apple VideoToolbox"

PAGES = (
    ("새 강의 노트", "영상 링크나 파일을 고르면 공부하기 좋은 노트로 정리해드려요."),
    ("실시간 자막", "듣고 있는 라이브 수업을 자막으로 따라가고, 놓친 부분은 기록에서 다시 읽으세요."),
    ("노트 작성 방식", "AI와 노트 양식을 정해두면 노트의 말투와 구성이 내 공부 방식에 맞춰져요."),
    ("연결과 업데이트", "컴퓨터와 AI 계정 연결 상태를 한곳에서 관리하세요."),
)
STEP_LABELS = ("영상 가져오기", "쉬는 시간 정리", "강의 글 확인", "내 노트 완성")
STAGE_TO_STEP = {"download": 0, "edit": 1, "transcribe": 2, "notes": 3}
STAGE_DONE_TEXT = {
    "download": ("영상을 가져왔어요", "다음에는 쉬는 시간 정리부터 이어갈 수 있어요."),
    "edit": ("쉬는 시간 정리를 마쳤어요", "다음에는 강의 글 만들기부터 이어갈 수 있어요."),
    "transcribe": ("강의 글을 저장했어요", "다음에는 노트 정리부터 이어갈 수 있어요."),
    "notes": ("노트가 준비됐어요", "바로 열어보거나 저장된 폴더에서 다른 파일과 함께 확인하세요."),
}
# Exact pipeline stage names -> (step index, headline, plain-language detail).
STAGE_INFO: dict[str, tuple[int, str, str]] = {
    "사전 점검": (0, "준비 중", "필요한 도구와 저장된 작업을 확인하고 있어요."),
    "영상 확인": (0, "영상 가져오는 중", "영상 정보를 읽고 있어요."),
    "파일 확인": (0, "파일 확인 중", "선택한 파일을 읽고 있어요."),
    "전문용어 준비": (0, "전문용어 준비 중", "강의명에 맞는 단어를 AI로 만들고 있어요."),
    "파일 가져오기": (0, "파일 가져오는 중", "선택한 파일을 작업 폴더로 복사하고 있어요."),
    "원본 다운로드": (0, "영상 가져오는 중", "영상의 소리를 가져오고 있어요."),
    "원본 영상 다운로드": (0, "영상 가져오는 중", "영상 파일을 저장하고 있어요."),
    "원본 다운로드 완료": (0, "영상 가져오기 완료", "다음에는 쉬는 시간 정리부터 이어갈 수 있어요."),
    "파형 준비": (1, "쉬는 시간 정리 준비", "소리의 크기를 분석해 편집 화면을 준비하고 있어요."),
    "Razor 편집": (1, "쉬는 시간 정리", "소리를 들으며 수업이 아닌 구간을 빼주세요."),
    "편집본 저장": (1, "쉬는 시간 정리", "뺀 구간을 제외한 소리를 저장하고 있어요."),
    "영상 편집본 저장": (1, "쉬는 시간 정리", "뺀 구간을 영상에도 똑같이 적용하고 있어요."),
    "편집 완료": (1, "쉬는 시간 정리 완료", "다음에는 강의 글 만들기부터 이어갈 수 있어요."),
    "GPU 전사": (2, "강의 글 만드는 중", "강의를 문장으로 옮기고 있어요."),
    "전사 확인": (2, "강의 글 확인", "내용을 한 번 확인해주세요."),
    "전사 완료": (2, "강의 글 저장 완료", "다음에는 노트 정리부터 이어갈 수 있어요."),
    "AI 노트": (3, "노트 정리 중", "확인한 강의 글을 공부하기 좋은 노트로 정리하고 있어요."),
    "완료": (3, "마무리 중", "결과를 정리하고 있어요."),
}
STAGES_WITH_LIVE_DETAIL = {"원본 다운로드", "원본 영상 다운로드", "GPU 전사"}
STAGES_COMPLETING_STEP = {"원본 다운로드 완료", "편집 완료", "전사 완료", "완료"}
EFFORT_LABELS = {
    "low": "빠르게",
    "medium": "보통",
    "high": "깊게",
    "xhigh": "더 깊게",
    "max": "최대",
}
WHISPER_CHOICES = (
    ("가장 정확 · 기본", "large-v3"),
    ("정확 (이전 버전)", "large-v2"),
    ("가벼움 · 빠름", "medium"),
    ("Qwen3-ASR 1.7B (선택 엔진)", "qwen3-asr-1.7b"),
)


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class ElidedLabel(QLabel):
    """A label that elides instead of forcing the layout wider; full text in the tooltip."""

    def __init__(self, text: str = "", mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight) -> None:
        super().__init__("")
        self._full = ""
        self._mode = mode
        self._auto_tip = True
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full = text
        QLabel.setText(self, text)
        if self._auto_tip:
            QLabel.setToolTip(self, text)
        self._elide()

    def setToolTip(self, tip: str) -> None:  # noqa: N802 - Qt override
        self._auto_tip = False
        QLabel.setToolTip(self, tip)

    def fullText(self) -> str:  # noqa: N802
        return self._full

    def _pad(self) -> int:
        shown = QLabel.text(self)
        return max(0, QLabel.sizeHint(self).width() - self.fontMetrics().horizontalAdvance(shown))

    def sizeHint(self) -> QSize:  # noqa: N802
        base = QLabel.sizeHint(self)
        return QSize(self._pad() + self.fontMetrics().horizontalAdvance(self._full), base.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        base = QLabel.minimumSizeHint(self)
        return QSize(self._pad() + self.fontMetrics().horizontalAdvance("…"), base.height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        available = max(0, self.width() - self._pad())
        text = self.fontMetrics().elidedText(self._full, self._mode, available)
        if text != QLabel.text(self):
            QLabel.setText(self, text)


def make_pill(text: str = "", tone: str = "neutral") -> ElidedLabel:
    pill = ElidedLabel(text)
    pill.setObjectName("StatusPill")
    pill.setProperty("tone", tone)
    return pill


def set_pill(pill: QLabel, text: str, tone: str, tooltip: str | None = None) -> None:
    pill.setText(text)
    if tooltip is not None:
        pill.setToolTip(tooltip)
    pill.setProperty("tone", tone)
    repolish(pill)


class Stepper(QWidget):
    """Horizontal four-stage indicator. States: pending, active, done, error."""

    def __init__(self, labels: tuple[str, ...]) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._dots: list[QLabel] = []
        self._texts: list[QLabel] = []
        self._lines: list[QFrame] = []
        for index, label in enumerate(labels):
            dot = QLabel(str(index + 1))
            dot.setObjectName("StepDot")
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text = QLabel(label)
            text.setObjectName("StepLabel")
            layout.addWidget(dot)
            layout.addWidget(text)
            self._dots.append(dot)
            self._texts.append(text)
            if index < len(labels) - 1:
                line = QFrame()
                line.setObjectName("StepLine")
                line.setFixedHeight(2)
                line.setMinimumWidth(12)
                line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                layout.addWidget(line, 1)
                self._lines.append(line)
        self.set_states(["pending"] * len(labels))

    def set_states(self, states: list[str]) -> None:
        for index, state in enumerate(states):
            dot, text = self._dots[index], self._texts[index]
            if state == "done":
                dot.setPixmap(QIcon(str(ASSET_DIR / "step-check.svg")).pixmap(14, 14))
            else:
                dot.setPixmap(QPixmap())
                dot.setText("!" if state == "error" else str(index + 1))
            for widget in (dot, text):
                widget.setProperty("state", state)
                repolish(widget)
            dot.setAccessibleName(f"{index + 1}단계 {text.text()} {state}")
        for index, line in enumerate(self._lines):
            line.setProperty("done", "true" if states[index] == "done" else "false")
            repolish(line)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self._background: list[tuple[QThread, object]] = []
        self._finish_callbacks: dict[object, object] = {}
        self.pipeline_worker: PipelineWorker | None = None
        self.last_result: PipelineResult | None = None
        self.current_review: TranscriptReview | None = None
        self.audio_editor: AudioEditorDialog | None = None
        self._login_worker: AuthWorker | None = None
        self._login_provider = ""
        self._auth_pending = 0
        self._pipeline_step = 0
        self._status_mode = "idle"
        self._save_geometry_on_close = True
        self.setWindowTitle("Lecture Scribe · 강의를 내 노트로")
        self.setMinimumSize(960, 640)
        self._notification_tray: QSystemTrayIcon | None = None
        self._model_catalogs: dict[str, ModelCatalog] = {}
        self._loading_catalogs: set[str] = set()
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self._persist_ui_state)
        self._build_ui()
        self._setup_notifications()
        self._setup_shortcuts()
        self._connect_persistence()
        self._restore_geometry()
        self._refresh_local_status()
        self.url_input.setFocus(Qt.FocusReason.OtherFocusReason)
        QTimer.singleShot(400, self._startup_auth_status)
        QTimer.singleShot(600, self._startup_model_catalog)

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("AppCanvas")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        self.tabs = QStackedWidget()
        self.tabs.setObjectName("PageStack")
        self.tabs.addWidget(self._build_transcribe_tab())
        self.live_page = LiveCaptionPage(
            self.settings,
            self._card,
            self._field_label,
            self._launch_worker,
            cookie_browser=lambda: str(self.cookie_combo.currentData() or "none"),
        )
        live_scroll, live_column = self._page_shell(*PAGES[1])
        live_column.addWidget(self.live_page, 1)
        self.tabs.addWidget(live_scroll)
        self.tabs.addWidget(self._build_prompt_tab())
        self.tabs.addWidget(self._build_tools_tab())
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        # Buttons take keyboard focus only through Tab, so a mouse click never
        # leaves a focus ring behind.
        for button in self.findChildren(QPushButton):
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def _restore_geometry(self) -> None:
        available = self.screen().availableGeometry() if self.screen() else None
        default_w, default_h = 1240, 820
        if available is not None:
            default_w = min(default_w, available.width())
            default_h = min(default_h, available.height())
        self.resize(max(default_w, self.minimumWidth()), max(default_h, self.minimumHeight()))
        saved = QSettings("Local Tools", "Lecture Scribe").value("window/geometry")
        if saved is not None and self.restoreGeometry(saved):
            if available is not None and not available.intersects(self.frameGeometry()):
                self.resize(default_w, default_h)
                self.move(available.center() - self.rect().center())

    def _setup_shortcuts(self) -> None:
        self._shortcuts: list[QShortcut] = []
        for index in range(len(PAGES)):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self)
            shortcut.activated.connect(partial(self._switch_page, index))
            self._shortcuts.append(shortcut)

    def _page_shell(self, title: str, subtitle: str) -> tuple[QScrollArea, QVBoxLayout]:
        """A scrollable page with a centered content column (max 960px) and a header."""
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder_layout = QHBoxLayout(holder)
        holder_layout.setContentsMargins(32, 32, 32, 32)
        column = QWidget()
        column.setMaximumWidth(960)
        column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # Side stretches soak up whatever the 960px column cannot use, centering it.
        holder_layout.addStretch(1)
        holder_layout.addWidget(column, 1000)
        holder_layout.addStretch(1)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        heading = QLabel(title)
        heading.setObjectName("PageTitle")
        sub = QLabel(subtitle)
        sub.setObjectName("PageSubtitle")
        sub.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(sub)
        layout.addSpacing(8)
        scroll.setWidget(holder)
        return scroll, layout

    def _setup_notifications(self) -> None:
        icon = QIcon(str(ASSET_DIR / "app-icon.svg"))
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip("Lecture Scribe")
        tray.messageClicked.connect(self._restore_from_notification)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        self._notification_tray = tray

    def _notify_action_required(self, title: str, message: str) -> None:
        if self.isActiveWindow() and not self.isMinimized():
            return
        QApplication.alert(self, 0)
        if self._notification_tray and self._notification_tray.supportsMessages():
            self._notification_tray.showMessage(
                title,
                message,
                QSystemTrayIcon.MessageIcon.Information,
                15_000,
            )

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self._restore_from_notification()

    def _restore_from_notification(self) -> None:
        self._switch_page(0)
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if self.audio_editor is not None:
            self.audio_editor.showNormal()
            self.audio_editor.raise_()
            self.audio_editor.activateWindow()

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(232)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 24, 12, 16)
        layout.setSpacing(4)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(12, 0, 12, 0)
        brand_row.setSpacing(8)
        dot = QFrame()
        dot.setObjectName("BrandDot")
        dot.setFixedSize(10, 10)
        brand_name = QLabel("Lecture Scribe")
        brand_name.setObjectName("SidebarBrand")
        brand_row.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
        brand_row.addWidget(brand_name, 1)
        layout.addLayout(brand_row)
        layout.addSpacing(24)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: list[QPushButton] = []
        for index, (label, icon_name) in enumerate(
            zip(
                (page[0] for page in PAGES),
                ("nav-new.svg", "nav-live.svg", "nav-note.svg", "nav-connect.svg"),
            )
        ):
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setIcon(QIcon(str(ASSET_DIR / icon_name)))
            button.setIconSize(QSize(20, 20))
            button.setAccessibleName(label)
            button.setToolTip(f"{label} (Ctrl+{index + 1})")
            button.clicked.connect(lambda _checked=False, value=index: self._switch_page(value))
            self.nav_group.addButton(button)
            self.nav_buttons.append(button)
            layout.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        layout.addStretch(1)

        self.gpu_badge = make_pill("확인 중")
        self.ai_badge = make_pill("확인 중")
        for caption, pill in (("컴퓨터", self.gpu_badge), ("AI 계정", self.ai_badge)):
            row = QHBoxLayout()
            row.setContentsMargins(12, 0, 4, 0)
            row.setSpacing(8)
            name = QLabel(caption)
            name.setObjectName("SidebarMeta")
            name.setFixedWidth(52)
            row.addWidget(name)
            row.addWidget(pill, 1, Qt.AlignmentFlag.AlignLeft)
            layout.addLayout(row)
        return sidebar

    def _switch_page(self, index: int) -> None:
        self.tabs.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)

    # ------------------------------------------------------------ small helpers

    def _card(
        self, title: str | None, helper: str | None = None
    ) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)
        if title:
            label = QLabel(title)
            label.setObjectName("SectionTitle")
            layout.addWidget(label)
        if helper:
            description = QLabel(helper)
            description.setObjectName("HelperText")
            description.setWordWrap(True)
            layout.addWidget(description)
        return card, layout

    def _titled_card(
        self, title: str, description: str, pill_text: str, tone: str = "neutral"
    ) -> tuple[QFrame, QVBoxLayout, ElidedLabel]:
        card, layout = self._card(None)
        head = QHBoxLayout()
        head.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        pill = make_pill(pill_text, tone)
        head.addWidget(label)
        head.addStretch(1)
        head.addWidget(pill)
        layout.addLayout(head)
        text = QLabel(description)
        text.setObjectName("BodyText")
        text.setWordWrap(True)
        layout.addWidget(text)
        return card, layout, pill

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("FieldLabel")
        return label

    @staticmethod
    def _button(text: str, role: str = "SecondaryButton") -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(role)
        return button

    @staticmethod
    def _set_button_role(button: QPushButton, role: str) -> None:
        button.setObjectName(role)
        repolish(button)

    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    # ------------------------------------------------------------ new-note page

    def _build_transcribe_tab(self) -> QWidget:
        scroll, col = self._page_shell(*PAGES[0])

        # ---- source card: the one place to start a note
        source_card, source = self._card(None)
        source.addWidget(self._field_label("강의 영상 링크 또는 파일"))
        self.url_input = QLineEdit(self.settings.last_url)
        self.url_input.setObjectName("SourceInput")
        self.url_input.setPlaceholderText("YouTube 링크를 붙여넣거나 파일을 선택하세요")
        self.url_input.setClearButtonEnabled(True)
        self.url_input.setAccessibleName("YouTube 영상 링크 또는 로컬 영상·음성 파일")
        paste_button = self._button("붙여넣기")
        paste_button.setAccessibleName("클립보드에서 YouTube URL 붙여넣기")
        paste_button.clicked.connect(self._paste_url)
        choose_media_button = self._button("파일 선택")
        choose_media_button.setAccessibleName("전사할 영상 또는 음성 파일 선택")
        choose_media_button.clicked.connect(self._choose_media_file)
        for button in (paste_button, choose_media_button):
            button.setFixedHeight(44)
        url_row = QHBoxLayout()
        url_row.setSpacing(8)
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(paste_button)
        url_row.addWidget(choose_media_button)
        source.addLayout(url_row)

        fields = QGridLayout()
        fields.setHorizontalSpacing(16)
        fields.setVerticalSpacing(4)
        self.lecture_name_input = QLineEdit(self.settings.terminology_lecture_name)
        self.lecture_name_input.setPlaceholderText("예: 운영체제 3주차")
        self.lecture_name_input.setAccessibleName("강의명 (선택)")
        self.lecture_name_input.setToolTip(
            "‘전문용어 자동 입력’을 켰다면 강의명이 필요해요. "
            "강의명에 맞는 전문용어를 AI가 미리 만들어 음성 인식에 활용합니다."
        )
        self.folder_input = QLineEdit(self.settings.output_dir)
        self.folder_input.setCursorPosition(0)
        self.folder_input.setAccessibleName("노트를 저장할 폴더")
        choose_button = self._button("찾아보기")
        choose_button.clicked.connect(self._choose_output_folder)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        folder_row.addWidget(self.folder_input, 1)
        folder_row.addWidget(choose_button)
        fields.addWidget(self._field_label("강의명 (선택)"), 0, 0)
        fields.addWidget(self._field_label("노트를 저장할 곳"), 0, 1)
        fields.addWidget(self.lecture_name_input, 1, 0)
        fields.addLayout(folder_row, 1, 1)
        fields.setColumnStretch(0, 1)
        fields.setColumnStretch(1, 1)
        source.addLayout(fields)

        action_row = QHBoxLayout()
        action_row.setSpacing(12)
        self.action_helper = QLabel("중간에 멈춰도 끝난 단계부터 이어집니다.")
        self.action_helper.setObjectName("HelperText")
        self.action_helper.setWordWrap(True)
        self.start_button = self._button("강의 노트 만들기", "PrimaryButton")
        self.start_button.setMinimumHeight(44)
        self.start_button.setMinimumWidth(168)
        self.start_button.setAccessibleName("강의 노트 만들기 시작")
        self.start_button.clicked.connect(lambda _checked=False: self._start_pipeline("all"))
        action_row.addWidget(self.action_helper, 1)
        action_row.addWidget(self.start_button)
        source.addLayout(action_row)

        self.advanced_panel = QFrame()
        self.advanced_panel.setObjectName("Card")
        self.advanced_panel.setProperty("tone", "muted")
        advanced_layout = QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(16, 12, 16, 12)
        advanced_layout.setSpacing(8)
        advanced_hint = QLabel("필요한 단계만 골라 실행할 수 있어요. 이전에 끝낸 단계는 다시 하지 않습니다.")
        advanced_hint.setObjectName("HelperText")
        advanced_hint.setWordWrap(True)
        advanced_layout.addWidget(advanced_hint)
        stage_grid = QGridLayout()
        stage_grid.setHorizontalSpacing(8)
        stage_grid.setVerticalSpacing(8)
        self.stage_buttons: list[QPushButton] = []
        for index, (label, stage) in enumerate(
            [
                ("1. 영상 가져오기", "download"),
                ("2. 쉬는 시간 정리", "edit"),
                ("3. 강의 글 만들기", "transcribe"),
                ("4. 노트 정리하기", "notes"),
            ]
        ):
            button = self._button(label)
            button.clicked.connect(lambda _checked=False, value=stage: self._start_pipeline(value))
            stage_grid.addWidget(button, index // 2, index % 2)
            self.stage_buttons.append(button)
        advanced_layout.addLayout(stage_grid)
        self.advanced_panel.setVisible(False)
        source.addWidget(self.advanced_panel)
        col.addWidget(source_card)

        # ---- one status region: idle (hidden) / running / review / done
        col.addWidget(self._build_status_card())

        # ---- options
        option_card, options = self._card("처리 방식")
        toggles = QGridLayout()
        toggles.setHorizontalSpacing(24)
        toggles.setVerticalSpacing(0)
        self.vad_check = QCheckBox("말이 없는 긴 구간은 건너뛰기")
        self.vad_check.setChecked(self.settings.vad_filter)
        self.vad_check.setToolTip("강의 중 소리가 거의 없는 구간을 건너뛰어 반복 문장이 생길 가능성을 낮춥니다.")
        self.auto_terminology_check = QCheckBox("전문용어 자동 입력")
        self.auto_terminology_check.setChecked(self.settings.auto_terminology)
        self.auto_terminology_check.setToolTip(
            "강의명을 바탕으로 연결된 AI가 관련 전문용어를 만들어 "
            "음성 인식(Whisper)의 문맥으로 자동 적용합니다. 위의 강의명이 필요해요."
        )
        self.keep_video_check = QCheckBox("쉬는 시간을 뺀 영상도 보관하기")
        self.keep_video_check.setChecked(self.settings.keep_video)
        self.keep_video_check.setToolTip(
            "최대 1080p 영상을 받아, 제외한 구간을 똑같이 제거한 "
            f"edited_lecture.mp4를 {VIDEO_ENCODER_NAME} 하드웨어 가속으로 저장합니다. "
            "완료 후 원본 영상은 정리합니다."
        )
        self.keep_audio_check = QCheckBox("가져온 원본 오디오도 보관하기")
        self.keep_audio_check.setChecked(self.settings.keep_audio)
        self.use_time_range_check = QCheckBox("영상의 일부만 가져오기")
        self.live_snapshot_check = QCheckBox("진행 중인 라이브는 지금까지 부분만")
        self.live_snapshot_check.setChecked(self.settings.live_snapshot)
        self.live_snapshot_check.setToolTip(
            "아직 끝나지 않은 YouTube 라이브 링크를 넣으면 방송 시작부터 지금 시점까지만 "
            "받아 전사합니다. 끄면 방송이 끝난 뒤에만 가져올 수 있습니다."
        )
        for index, check in enumerate(
            [
                self.vad_check,
                self.auto_terminology_check,
                self.keep_video_check,
                self.keep_audio_check,
                self.use_time_range_check,
                self.live_snapshot_check,
            ]
        ):
            toggles.addWidget(check, index // 2, index % 2)
        toggles.setColumnStretch(0, 1)
        toggles.setColumnStretch(1, 1)
        options.addLayout(toggles)

        self.clip_start_input = QLineEdit(format_timecode(self.settings.clip_start_seconds))
        self.clip_start_input.setPlaceholderText("00:00:00")
        self.clip_start_input.setAccessibleName("가져올 시작 시간")
        self.clip_end_input = QLineEdit(format_timecode(self.settings.clip_end_seconds))
        self.clip_end_input.setPlaceholderText("비워두면 끝까지")
        self.clip_end_input.setAccessibleName("가져올 종료 시간")
        self.time_range_panel = QWidget()
        range_layout = QVBoxLayout(self.time_range_panel)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(4)
        range_row = QHBoxLayout()
        range_row.setSpacing(8)
        range_row.addWidget(self._field_label("시작"))
        range_row.addWidget(self.clip_start_input, 1)
        range_row.addWidget(self._field_label("종료"))
        range_row.addWidget(self.clip_end_input, 1)
        range_help = QLabel(
            "잘 모르겠다면 비워두세요. 다음 화면에서 소리를 들으며 수업 전 음악과 쉬는 시간을 쉽게 뺄 수 있어요."
        )
        range_help.setObjectName("HelperText")
        range_help.setWordWrap(True)
        range_layout.addLayout(range_row)
        range_layout.addWidget(range_help)
        has_range = self.settings.clip_start_seconds > 0 or self.settings.clip_end_seconds is not None
        self.use_time_range_check.setChecked(has_range)
        self.time_range_panel.setVisible(has_range)
        self.use_time_range_check.toggled.connect(self.time_range_panel.setVisible)
        options.addWidget(self.time_range_panel)

        # advanced disclosure
        self.advanced_toggle = self._button("고급 설정", "GhostButton")
        self.advanced_toggle.setProperty("disclosure", True)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setIcon(QIcon(str(ASSET_DIR / "chevron-right.svg")))
        self.advanced_toggle.setIconSize(QSize(12, 12))
        self.advanced_toggle.setAccessibleName("고급 설정 펼치기")
        options.addWidget(self.advanced_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.advanced_settings_panel = QFrame()
        self.advanced_settings_panel.setObjectName("Card")
        self.advanced_settings_panel.setProperty("tone", "muted")
        recognition_layout = QVBoxLayout(self.advanced_settings_panel)
        recognition_layout.setContentsMargins(16, 12, 16, 12)
        recognition_layout.setSpacing(12)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        self.whisper_combo = QComboBox()
        for label, value in WHISPER_CHOICES:
            self.whisper_combo.addItem(label, value)
            self.whisper_combo.setItemData(
                self.whisper_combo.count() - 1, value, Qt.ItemDataRole.ToolTipRole
            )
        if self.whisper_combo.findData(self.settings.whisper_model) < 0:
            self.whisper_combo.addItem(self.settings.whisper_model, self.settings.whisper_model)
        self._select_data(self.whisper_combo, self.settings.whisper_model)
        self.whisper_combo.setToolTip(
            "음성 인식에 쓰는 모델입니다. large-v3가 기본이며, "
            "Qwen3-ASR은 선택 엔진으로 첫 실행 때 약 5GB를 내려받습니다."
        )
        self.language_combo = QComboBox()
        for label, code in [
            ("자동 감지", "auto"),
            ("한국어", "ko"),
            ("영어", "en"),
            ("일본어", "ja"),
            ("중국어", "zh"),
        ]:
            self.language_combo.addItem(label, code)
        self._select_data(self.language_combo, self.settings.language)
        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(self.settings.beam_size)
        self.beam_spin.setToolTip(
            "높을수록 여러 후보를 더 꼼꼼히 비교해 정확해지지만 느려집니다. "
            "(Windows에서 쓰는 beam size 값)"
        )
        self.cookie_combo = QComboBox()
        for label, value in [
            ("사용 안 함", "none"),
            ("Chrome 로그인 정보 사용", "chrome"),
            ("Edge 로그인 정보 사용", "edge"),
            ("Firefox 로그인 정보 사용", "firefox"),
        ]:
            self.cookie_combo.addItem(label, value)
        self._select_data(self.cookie_combo, self.settings.cookie_browser)
        self.cookie_combo.setToolTip("로그인이 필요한 영상을 가져올 때 브라우저의 로그인 정보를 빌려 씁니다.")
        grid.addWidget(self._field_label("음성 인식 방식"), 0, 0)
        grid.addWidget(self._field_label("강의 언어"), 0, 1)
        grid.addWidget(self.whisper_combo, 1, 0)
        grid.addWidget(self.language_combo, 1, 1)
        if IS_WINDOWS:
            grid.addWidget(self._field_label("정확도 단계"), 2, 0)
            grid.addWidget(self._field_label("로그인이 필요한 영상"), 2, 1)
            grid.addWidget(self.beam_spin, 3, 0)
            grid.addWidget(self.cookie_combo, 3, 1)
        else:
            grid.addWidget(self._field_label("로그인이 필요한 영상"), 2, 0, 1, 2)
            grid.addWidget(self.cookie_combo, 3, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        recognition_layout.addLayout(grid)
        self.advanced_mode_check = QCheckBox("단계별로 직접 진행하기")
        self.advanced_mode_check.setToolTip(
            "영상 가져오기, 쉬는 시간 정리, 강의 글 만들기, 노트 정리를 원하는 단계만 실행합니다."
        )
        mode_help = QLabel("켜면 위쪽에 단계별 실행 버튼이 나타나요. 끝난 단계는 저장되며 같은 링크로 이어집니다.")
        mode_help.setObjectName("HelperText")
        mode_help.setWordWrap(True)
        recognition_layout.addWidget(self.advanced_mode_check)
        recognition_layout.addWidget(mode_help)
        self.advanced_settings_panel.setVisible(False)
        options.addWidget(self.advanced_settings_panel)
        self.advanced_toggle.toggled.connect(self._toggle_advanced_settings)
        self.advanced_mode_check.toggled.connect(self.advanced_panel.setVisible)
        self.advanced_mode_check.toggled.connect(self._set_advanced_mode_ui)
        col.addWidget(option_card)
        col.addStretch(1)
        return scroll

    @Slot(bool)
    def _toggle_advanced_settings(self, expanded: bool) -> None:
        self.advanced_settings_panel.setVisible(expanded)
        icon = "chevron-down.svg" if expanded else "chevron-right.svg"
        self.advanced_toggle.setIcon(QIcon(str(ASSET_DIR / icon)))

    @Slot(bool)
    def _set_advanced_mode_ui(self, enabled: bool) -> None:
        self.start_button.setVisible(not enabled)
        self.action_helper.setText(
            "아래에서 필요한 단계만 골라 실행하세요."
            if enabled
            else "중간에 멈춰도 끝난 단계부터 이어집니다."
        )

    @Slot()
    def _paste_url(self) -> None:
        self.url_input.setText(QApplication.clipboard().text().strip())

    def _build_status_card(self) -> QFrame:
        card, layout = self._card(None)
        self.status_card = card
        head = QHBoxLayout()
        head.setSpacing(8)
        self.status_title = QLabel("대기 중")
        self.status_title.setObjectName("SectionTitle")
        self.status_pill = make_pill("진행 중", "accent")
        head.addWidget(self.status_title, 1)
        head.addWidget(self.status_pill)
        layout.addLayout(head)
        self.stepper = Stepper(STEP_LABELS)
        layout.addWidget(self.stepper)

        self.progress_row = QWidget()
        progress_layout = QHBoxLayout(self.progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(12)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.percent_label = QLabel("0%")
        self.percent_label.setObjectName("FieldLabel")
        self.percent_label.setMinimumWidth(40)
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_layout.addWidget(self.progress_bar, 1, Qt.AlignmentFlag.AlignVCenter)
        progress_layout.addWidget(self.percent_label)
        layout.addWidget(self.progress_row)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("BodyText")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        # review panel
        self.review_panel = QWidget()
        review_layout = QVBoxLayout(self.review_panel)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.setSpacing(8)
        review_status_row = QHBoxLayout()
        review_status_row.setSpacing(8)
        self.review_badge = make_pill("", "neutral")
        self.review_headline = QLabel("")
        self.review_headline.setObjectName("InstructionText")
        self.review_headline.setWordWrap(True)
        review_status_row.addWidget(self.review_badge)
        review_status_row.addWidget(self.review_headline, 1)
        review_layout.addLayout(review_status_row)
        self.review_metrics = QLabel("")
        self.review_metrics.setObjectName("HelperText")
        self.review_metrics.setWordWrap(True)
        review_layout.addWidget(self.review_metrics)
        self.review_warnings = QLabel("")
        self.review_warnings.setObjectName("BodyText")
        self.review_warnings.setWordWrap(True)
        review_layout.addWidget(self.review_warnings)
        self.review_preview = QPlainTextEdit()
        self.review_preview.setReadOnly(True)
        self.review_preview.setMinimumHeight(220)
        self.review_preview.setAccessibleName("확인할 전체 강의 글")
        review_layout.addWidget(self.review_preview)
        review_actions = QHBoxLayout()
        review_actions.setSpacing(8)
        self.review_open_button = self._button("전체 글 파일 열기")
        self.review_open_button.clicked.connect(self._open_review_transcript)
        self.review_reject_button = self._button("여기서 멈추기", "DangerButton")
        self.review_reject_button.clicked.connect(lambda _checked=False: self._submit_transcript_review(False))
        self.review_approve_button = self._button("확인했어요 · 노트 만들기", "PrimaryButton")
        self.review_approve_button.clicked.connect(lambda _checked=False: self._submit_transcript_review(True))
        review_actions.addWidget(self.review_open_button)
        review_actions.addStretch(1)
        review_actions.addWidget(self.review_reject_button)
        review_actions.addWidget(self.review_approve_button)
        review_layout.addLayout(review_actions)
        self.review_panel.setVisible(False)
        layout.addWidget(self.review_panel)

        # result panel
        self.result_panel = QWidget()
        result_layout = QVBoxLayout(self.result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(8)
        self.result_title = QLabel("")
        self.result_title.setObjectName("ResultTitle")
        self.result_title.setWordWrap(True)
        self.result_message = QLabel("")
        self.result_message.setObjectName("BodyText")
        self.result_message.setWordWrap(True)
        self.result_path = ElidedLabel("", Qt.TextElideMode.ElideMiddle)
        self.result_path.setObjectName("HelperText")
        self.result_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_layout.addWidget(self.result_title)
        result_layout.addWidget(self.result_message)
        result_layout.addWidget(self.result_path)
        result_buttons = QHBoxLayout()
        result_buttons.setSpacing(8)
        self.open_notes_button = self._button("노트 열기", "PrimaryButton")
        self.open_notes_button.clicked.connect(self._open_last_notes)
        self.open_transcript_button = self._button("강의 글 열기")
        self.open_transcript_button.clicked.connect(self._open_last_transcript)
        self.open_video_button = self._button("편집 영상 열기")
        self.open_video_button.clicked.connect(self._open_last_video)
        self.open_folder_button = self._button("저장 폴더 열기")
        self.open_folder_button.clicked.connect(self._open_last_folder)
        for button in (
            self.open_notes_button,
            self.open_transcript_button,
            self.open_video_button,
            self.open_folder_button,
        ):
            result_buttons.addWidget(button)
        result_buttons.addStretch(1)
        result_layout.addLayout(result_buttons)
        self.result_panel.setVisible(False)
        layout.addWidget(self.result_panel)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.cancel_button = self._button("잠시 멈추기", "SecondaryButton")
        self.cancel_button.clicked.connect(self._cancel_pipeline)
        self.log_toggle = self._button("자세히 보기", "GhostButton")
        self.log_toggle.setCheckable(True)
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        controls.addWidget(self.log_toggle)
        layout.addLayout(controls)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1500)
        self.log_view.setMinimumHeight(140)
        self.log_view.setPlaceholderText("작업 과정의 자세한 내용이 여기에 표시됩니다.")
        self.log_view.setVisible(False)
        self.log_toggle.toggled.connect(self._toggle_log)
        layout.addWidget(self.log_view)

        card.setVisible(False)
        return card

    @Slot(bool)
    def _toggle_log(self, shown: bool) -> None:
        self.log_view.setVisible(shown)
        self.log_toggle.setText("간단히 보기" if shown else "자세히 보기")

    def _set_status_mode(self, mode: str, tone: str | None = None, pill: str | None = None) -> None:
        """Morph the single status region: idle / running / review / done / stopped."""
        self._status_mode = mode
        self.status_card.setVisible(mode != "idle")
        running = mode == "running"
        self.progress_row.setVisible(mode in {"running", "review"})
        self.detail_label.setVisible(mode in {"running", "review", "stopped"})
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        self.review_panel.setVisible(mode == "review")
        self.result_panel.setVisible(mode == "done")
        defaults = {
            "running": ("진행 중", "accent"),
            "review": ("확인 필요", "warning"),
            "done": ("완료", "success"),
            "stopped": ("중단됨", "warning"),
        }
        default_text, default_tone = defaults.get(mode, ("", "neutral"))
        set_pill(self.status_pill, pill or default_text, tone or default_tone)

    def _set_step_states(self, active: int | None, *, done_through: int = -1, error: bool = False) -> None:
        states = []
        for index in range(len(STEP_LABELS)):
            if index <= done_through:
                states.append("done")
            elif active is not None and index == active:
                states.append("error" if error else "active")
            elif active is not None and index < active:
                states.append("done")
            else:
                states.append("pending")
        self.stepper.set_states(states)

    # -------------------------------------------------------------- prompt page

    def _build_prompt_tab(self) -> QWidget:
        scroll, col = self._page_shell(*PAGES[2])

        model_card, model_layout = self._card(
            "노트를 만드는 AI",
            "ChatGPT 또는 Claude를 고르고, 연결된 계정에서 쓸 수 있는 최신 모델 중에 선택하세요.",
        )
        model_grid = QGridLayout()
        model_grid.setHorizontalSpacing(16)
        model_grid.setVerticalSpacing(4)
        self.provider_combo = QComboBox()
        self.provider_combo.setAccessibleName("노트 작성 AI 서비스")
        for provider, label in PROVIDER_CHOICES:
            self.provider_combo.addItem(label, provider)
        self._select_data(self.provider_combo, normalize_provider(self.settings.ai_provider))
        self._active_provider = self._current_provider()
        self.refresh_models_button = self._button("모델 목록 새로고침", "GhostButton")
        self.refresh_models_button.clicked.connect(self._refresh_model_catalog)
        self.notes_model_combo = QComboBox()
        self.notes_model_combo.setAccessibleName("노트 작성 모델")
        self.terminology_model_combo = QComboBox()
        self.terminology_model_combo.setAccessibleName("전문용어 생성 모델")
        self.reasoning_combo = QComboBox()
        self.reasoning_combo.setAccessibleName("생각 깊이")
        self.reasoning_combo.setToolTip(
            "깊게 생각할수록 노트 품질이 좋아지지만 오래 걸립니다. (reasoning effort)"
        )
        self.model_source_label = QLabel("")
        self.model_source_label.setObjectName("HelperText")
        self.model_source_label.setWordWrap(True)
        model_grid.addWidget(self._field_label("AI 서비스"), 0, 0)
        model_grid.addWidget(self.provider_combo, 1, 0)
        model_grid.addWidget(self.refresh_models_button, 1, 1)
        model_grid.addWidget(self._field_label("노트 작성 모델"), 2, 0)
        model_grid.addWidget(self._field_label("생각 깊이"), 2, 1)
        model_grid.addWidget(self.notes_model_combo, 3, 0)
        model_grid.addWidget(self.reasoning_combo, 3, 1)
        model_grid.addWidget(self._field_label("전문용어 생성 모델"), 4, 0)
        model_grid.addWidget(self.terminology_model_combo, 5, 0, 1, 2)
        model_grid.setColumnStretch(0, 2)
        model_grid.setColumnStretch(1, 1)
        model_layout.addLayout(model_grid)
        model_layout.addWidget(self.model_source_label)
        self._populate_model_combos()
        self.provider_combo.currentIndexChanged.connect(self._change_provider)
        self.notes_model_combo.currentIndexChanged.connect(self._on_notes_model_changed)
        col.addWidget(model_card)

        prompt_card, prompt_layout = self._card(
            "노트 양식",
            "강의용, 주식방송용처럼 원하는 만큼 양식을 저장해두고 작업마다 골라 쓸 수 있어요. "
            "{{transcript}}는 강의 글이 들어갈 자리이니 지우지 마세요.",
        )
        slot_row = QHBoxLayout()
        slot_row.setSpacing(8)
        self.prompt_slot_combo = QComboBox()
        self.prompt_slot_combo.setAccessibleName("사용할 노트 양식")
        self._refresh_prompt_slot_combo()
        self.prompt_slot_combo.currentIndexChanged.connect(self._change_prompt_slot)
        add_slot_button = self._button("새로 만들기")
        add_slot_button.clicked.connect(self._add_prompt_slot)
        rename_slot_button = self._button("이름 바꾸기", "GhostButton")
        rename_slot_button.clicked.connect(self._rename_prompt_slot)
        delete_slot_button = self._button("삭제", "DangerButton")
        delete_slot_button.clicked.connect(self._delete_prompt_slot)
        slot_row.addWidget(self.prompt_slot_combo, 1)
        slot_row.addWidget(add_slot_button)
        slot_row.addWidget(rename_slot_button)
        slot_row.addWidget(delete_slot_button)
        prompt_layout.addLayout(slot_row)
        self.prompt_editor = QPlainTextEdit()
        self.prompt_editor.setObjectName("PromptEditor")
        self.prompt_editor.setAccessibleName("노트 양식 편집기")
        self.prompt_editor.setMinimumHeight(240)
        self.prompt_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.prompt_editor.setPlainText(load_prompt(self.settings))
        prompt_layout.addWidget(self.prompt_editor, 1)
        prompt_actions = QHBoxLayout()
        prompt_actions.setSpacing(8)
        reset_button = self._button("되돌리기", "GhostButton")
        reset_button.setToolTip("처음 제공된 양식으로 되돌립니다. 저장하기 전에는 반영되지 않아요.")
        reset_button.clicked.connect(self._reset_prompt)
        save_button = self._button("저장", "PrimaryButton")
        save_button.setAccessibleName("노트 양식과 AI 설정 저장")
        save_button.clicked.connect(self._save_prompt_settings)
        prompt_actions.addWidget(reset_button)
        prompt_actions.addStretch(1)
        prompt_actions.addWidget(save_button)
        prompt_layout.addLayout(prompt_actions)
        col.addWidget(prompt_card, 1)
        return scroll

    # ----------------------------------------------------------- connections page

    def _build_tools_tab(self) -> QWidget:
        scroll, col = self._page_shell(*PAGES[3])

        gpu_card, gpu_layout, self.gpu_pill = self._titled_card(
            "내 컴퓨터", ACCELERATOR_DESCRIPTION, "확인 중"
        )
        self.gpu_detail = QLabel("확인 중")
        self.gpu_detail.setObjectName("HelperText")
        self.gpu_detail.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_detail)
        col.addWidget(gpu_card)

        auth_card, auth_layout, self.auth_pill = self._titled_card(
            "AI 계정 연결",
            "로그인은 공식 연결 창에서 진행합니다. 이 앱은 비밀번호를 보거나 저장하지 않아요.",
            "확인 중",
        )
        auth_provider_row = QHBoxLayout()
        auth_provider_row.setSpacing(8)
        self.auth_provider_combo = QComboBox()
        self.auth_provider_combo.setAccessibleName("연결을 관리할 AI 서비스")
        for provider, label in PROVIDER_CHOICES:
            self.auth_provider_combo.addItem(label, provider)
        self._select_data(self.auth_provider_combo, self._current_provider())
        self.auth_provider_combo.currentIndexChanged.connect(self._on_auth_provider_changed)
        auth_provider_row.addWidget(self._field_label("관리할 서비스"))
        auth_provider_row.addWidget(self.auth_provider_combo, 1)
        auth_layout.addLayout(auth_provider_row)
        self.auth_detail = QLabel("확인 중")
        self.auth_detail.setObjectName("HelperText")
        self.auth_detail.setWordWrap(True)
        self.auth_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        auth_layout.addWidget(self.auth_detail)
        auth_actions = QHBoxLayout()
        auth_actions.setSpacing(8)
        self.login_button = self._button("연결하기", "PrimaryButton")
        self.login_button.clicked.connect(self._on_login_button)
        self.logout_button = self._button("로그아웃")
        self.logout_button.clicked.connect(lambda _checked=False: self._run_auth_action("logout"))
        self.refresh_auth_button = self._button("상태 다시 확인", "GhostButton")
        self.refresh_auth_button.clicked.connect(lambda _checked=False: self._run_auth_action("status"))
        auth_actions.addWidget(self.login_button)
        auth_actions.addWidget(self.logout_button)
        auth_actions.addWidget(self.refresh_auth_button)
        auth_actions.addStretch(1)
        auth_layout.addLayout(auth_actions)
        self.auth_code_row = QWidget()
        code_layout = QVBoxLayout(self.auth_code_row)
        code_layout.setContentsMargins(0, 0, 0, 0)
        code_layout.setSpacing(4)
        code_instruction = QLabel("브라우저에서 로그인한 뒤 표시된 코드를 붙여넣으세요")
        code_instruction.setObjectName("InstructionText")
        code_instruction.setWordWrap(True)
        code_layout.addWidget(code_instruction)
        code_input_row = QHBoxLayout()
        code_input_row.setSpacing(8)
        self.auth_code_input = QLineEdit()
        self.auth_code_input.setPlaceholderText("인증 코드")
        self.auth_code_input.setAccessibleName("로그인 인증 코드")
        self.auth_code_input.returnPressed.connect(self._submit_auth_code)
        auth_code_button = self._button("코드 제출")
        auth_code_button.clicked.connect(self._submit_auth_code)
        code_input_row.addWidget(self.auth_code_input, 1)
        code_input_row.addWidget(auth_code_button)
        code_layout.addLayout(code_input_row)
        self.auth_code_row.setVisible(False)
        auth_layout.addWidget(self.auth_code_row)
        col.addWidget(auth_card)

        update_card, update_layout, self.update_pill = self._titled_card(
            "업데이트",
            "YouTube가 바뀌어 영상을 가져오지 못할 때 가장 먼저 업데이트해보세요.",
            f"버전 {current_version()}",
        )
        update_row = QHBoxLayout()
        update_row.setSpacing(8)
        self.update_channel_combo = QComboBox()
        self.update_channel_combo.addItem("새 기능 빠르게 받기 · 권장", "nightly")
        self.update_channel_combo.addItem("검증된 버전만 받기", "stable")
        self._select_data(self.update_channel_combo, self.settings.yt_dlp_channel)
        self.update_channel_combo.setAccessibleName("업데이트 방식")
        self.update_button = self._button("확인하고 업데이트")
        self.update_button.clicked.connect(self._run_ytdlp_update)
        update_row.addWidget(self.update_channel_combo, 1)
        update_row.addWidget(self.update_button)
        update_layout.addLayout(update_row)
        col.addWidget(update_card)

        log_card, log_layout = self._card("자세한 기록", "문제가 생겼을 때 원인을 확인할 수 있는 기록입니다.")
        self.tool_log = QPlainTextEdit()
        self.tool_log.setObjectName("ToolLog")
        self.tool_log.setReadOnly(True)
        self.tool_log.setMaximumBlockCount(1000)
        self.tool_log.setMinimumHeight(140)
        log_layout.addWidget(self.tool_log)
        col.addWidget(log_card)
        col.addStretch(1)
        return scroll

    # ----------------------------------------------------------- local status

    def _refresh_local_status(self) -> None:
        gpu = get_gpu_info()
        if gpu.available:
            memory_gb = gpu.memory_mb / 1024
            tip = f"{gpu.name} · 드라이버 {gpu.driver}"
            set_pill(self.gpu_badge, f"준비됨 · {memory_gb:.0f}GB", "success", tip)
            set_pill(self.gpu_pill, "준비됨", "success", tip)
            memory_name = "그래픽 메모리" if IS_WINDOWS else "통합 메모리"
            self.gpu_detail.setText(f"{gpu.name} · {memory_name} {memory_gb:.1f}GB")
            self.gpu_detail.setToolTip(tip)
        else:
            set_pill(self.gpu_badge, "확인 필요", "warning", gpu.detail)
            set_pill(self.gpu_pill, "확인 필요", "warning", gpu.detail)
            self.gpu_detail.setText("그래픽 가속 환경을 확인하지 못했어요. 자세한 내용은 마우스를 올려 확인하세요.")
            self.gpu_detail.setToolTip(str(gpu.detail))
        provider = self._auth_provider()
        self.auth_detail.setText(f"{PROVIDER_NAMES[provider]} 로그인 상태를 확인하고 있어요.")
        self._update_auth_tool_tip(provider)
        set_pill(self.ai_badge, "확인 중", "neutral")

    def _update_auth_tool_tip(self, provider: str) -> None:
        tool = find_tool(provider)
        text = (
            f"연결 도구: {tool}"
            if tool
            else f"{TOOL_NAMES[provider]}를 찾지 못했어요. {INSTALLER_NAME}을 먼저 실행해주세요."
        )
        self.auth_detail.setToolTip(text)

    def _choose_output_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "결과 저장 폴더 선택", self.folder_input.text()
        )
        if chosen:
            self.folder_input.setText(chosen)
            self._schedule_persist()

    def _choose_media_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "전사할 영상 또는 음성 파일 선택",
            "",
            "미디어 파일 (*.mp4 *.m4v *.mov *.mkv *.webm *.avi *.mp3 *.m4a *.wav *.flac *.aac *.ogg *.opus);;모든 파일 (*)",
        )
        if chosen:
            self.url_input.setText(chosen)

    # -------------------------------------------------------------- prompt slots

    def _refresh_prompt_slot_combo(self) -> None:
        self.settings.ensure_prompt_slots()
        combo = getattr(self, "prompt_slot_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        for slot in self.settings.prompt_slots:
            combo.addItem(slot["name"], slot["id"])
        combo.setCurrentIndex(combo.findData(self.settings.active_prompt_slot_id))
        combo.blockSignals(False)

    def _save_prompt_draft(self) -> None:
        if hasattr(self, "prompt_editor"):
            self.settings.active_prompt_slot()["content"] = self.prompt_editor.toPlainText()

    def _change_prompt_slot(self) -> None:
        if not hasattr(self, "prompt_editor"):
            return
        selected_id = str(self.prompt_slot_combo.currentData() or "")
        if not selected_id or selected_id == self.settings.active_prompt_slot_id:
            return
        self._save_prompt_draft()
        self.settings.active_prompt_slot_id = selected_id
        self.prompt_editor.setPlainText(load_prompt(self.settings))
        self._schedule_persist()

    def _add_prompt_slot(self) -> None:
        self._save_prompt_draft()
        name, accepted = QInputDialog.getText(self, "새 노트 양식", "양식 이름")
        name = " ".join(name.split()).strip()
        if not accepted or not name:
            return
        self.settings.add_prompt_slot(name, DEFAULT_PROMPT_PATH.read_text(encoding="utf-8"))
        self._refresh_prompt_slot_combo()
        self.prompt_editor.setPlainText(load_prompt(self.settings))

    def _rename_prompt_slot(self) -> None:
        slot = self.settings.active_prompt_slot()
        name, accepted = QInputDialog.getText(self, "노트 양식 이름", "양식 이름", text=slot["name"])
        name = " ".join(name.split()).strip()
        if accepted and name:
            slot["name"] = name
            self._refresh_prompt_slot_combo()

    def _delete_prompt_slot(self) -> None:
        if len(self.settings.prompt_slots) == 1:
            self._show_problem("노트 양식은 하나 이상 남겨야 해요.")
            return
        slot = self.settings.active_prompt_slot()
        answer = QMessageBox.question(self, "노트 양식 삭제", f"‘{slot['name']}’ 양식을 삭제할까요?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.settings.remove_active_prompt_slot()
        self._refresh_prompt_slot_combo()
        self.prompt_editor.setPlainText(load_prompt(self.settings))

    def _reset_prompt(self) -> None:
        answer = QMessageBox.question(
            self,
            "처음 양식으로 되돌리기",
            "현재 내용을 처음 제공된 노트 양식으로 바꿀까요? 저장하기 전까지는 실제 설정에 반영되지 않습니다.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.prompt_editor.setPlainText(DEFAULT_PROMPT_PATH.read_text(encoding="utf-8"))

    # ----------------------------------------------------------------- settings

    def _apply_ai_and_prompt_settings(self) -> None:
        """Validate and store the AI model choice and the note template (no run settings)."""
        provider = self._current_provider()
        model = str(self.notes_model_combo.currentData() or "")
        if not model:
            raise ValueError("노트 작성에 사용할 AI 모델을 선택해주세요.")
        terminology_model = str(self.terminology_model_combo.currentData() or "")
        if not terminology_model:
            raise ValueError("전문용어 생성에 사용할 AI 모델을 선택해주세요.")
        self._save_prompt_draft()
        template = self.prompt_editor.toPlainText()
        validate_prompt(template)
        self.settings.ai_provider = provider
        self.settings.set_ai_selection(
            provider, model, str(self.reasoning_combo.currentData() or ""), terminology_model
        )
        save_prompt(self.settings, template)

    def _apply_run_settings(self) -> None:
        """Validate and store the options that only matter for a run."""
        output_text = self.folder_input.text().strip()
        if not output_text:
            raise ValueError("노트를 저장할 폴더를 선택해주세요.")
        output_dir = Path(output_text).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.use_time_range_check.isChecked():
            clip_start, clip_end = validate_clip_range(
                parse_timecode(self.clip_start_input.text()),
                parse_timecode(self.clip_end_input.text()),
            )
        else:
            clip_start, clip_end = 0.0, None
        s = self.settings
        s.output_dir = str(output_dir.resolve())
        s.whisper_model = str(self.whisper_combo.currentData() or s.whisper_model)
        s.language = str(self.language_combo.currentData())
        s.vad_filter = self.vad_check.isChecked()
        s.clip_start_seconds = clip_start
        s.clip_end_seconds = clip_end
        s.keep_audio = self.keep_audio_check.isChecked()
        s.keep_video = self.keep_video_check.isChecked()
        s.live_snapshot = self.live_snapshot_check.isChecked()
        s.auto_terminology = self.auto_terminology_check.isChecked()
        s.beam_size = self.beam_spin.value()
        s.cookie_browser = str(self.cookie_combo.currentData())
        s.yt_dlp_channel = str(self.update_channel_combo.currentData())

    def _persist_ui_state(self) -> None:
        """Save what the controls currently show, without validating or raising."""
        s = self.settings
        try:
            folder = self.folder_input.text().strip()
            if folder:
                s.output_dir = folder
            s.last_url = self.url_input.text().strip()
            s.terminology_lecture_name = " ".join(self.lecture_name_input.text().split())
            s.whisper_model = str(self.whisper_combo.currentData() or s.whisper_model)
            s.language = str(self.language_combo.currentData() or s.language)
            s.vad_filter = self.vad_check.isChecked()
            s.keep_audio = self.keep_audio_check.isChecked()
            s.keep_video = self.keep_video_check.isChecked()
            s.live_snapshot = self.live_snapshot_check.isChecked()
            s.auto_terminology = self.auto_terminology_check.isChecked()
            s.beam_size = self.beam_spin.value()
            s.cookie_browser = str(self.cookie_combo.currentData() or s.cookie_browser)
            s.yt_dlp_channel = str(self.update_channel_combo.currentData() or s.yt_dlp_channel)
            if self.use_time_range_check.isChecked():
                try:
                    start, end = validate_clip_range(
                        parse_timecode(self.clip_start_input.text()),
                        parse_timecode(self.clip_end_input.text()),
                    )
                    s.clip_start_seconds, s.clip_end_seconds = start, end
                except (ValueError, TypeError):
                    pass
            else:
                s.clip_start_seconds, s.clip_end_seconds = 0.0, None
            s.ai_provider = self._current_provider()
            self._remember_model_selection()
            s.save()
        except (OSError, ValueError, TypeError) as exc:
            if hasattr(self, "tool_log"):
                self._append_tool_log(f"설정을 저장하지 못했습니다: {exc}")

    def _schedule_persist(self, *_args: object) -> None:
        self._persist_timer.start()

    def _connect_persistence(self) -> None:
        for edit in (
            self.url_input,
            self.folder_input,
            self.lecture_name_input,
            self.clip_start_input,
            self.clip_end_input,
        ):
            edit.editingFinished.connect(self._schedule_persist)
        for check in (
            self.vad_check,
            self.auto_terminology_check,
            self.keep_video_check,
            self.keep_audio_check,
            self.use_time_range_check,
            self.live_snapshot_check,
        ):
            check.toggled.connect(self._schedule_persist)
        for combo in (
            self.whisper_combo,
            self.language_combo,
            self.cookie_combo,
            self.provider_combo,
            self.notes_model_combo,
            self.terminology_model_combo,
            self.reasoning_combo,
            self.update_channel_combo,
        ):
            combo.currentIndexChanged.connect(self._schedule_persist)
        self.beam_spin.valueChanged.connect(self._schedule_persist)

    @Slot()
    def _save_prompt_settings(self) -> None:
        try:
            self._apply_ai_and_prompt_settings()
            self.settings.save()
        except (OSError, ValueError) as exc:
            self._show_problem(str(exc))
            return
        QMessageBox.information(self, "저장 완료", "노트 양식과 AI 설정을 저장했어요.")

    # ------------------------------------------------------------ pipeline run

    def _start_pipeline(self, target_stage: str = "all") -> None:
        if self.pipeline_worker is not None:
            return
        source = self.url_input.text().strip()
        if not is_youtube_url(source):
            try:
                resolve_local_media(source)
            except ValueError as exc:
                self._show_problem(
                    "YouTube 링크를 붙여넣거나 영상·음성 파일을 선택해주세요.", raw=str(exc)
                )
                self.url_input.setFocus()
                return
        try:
            self._apply_run_settings()
            self._apply_ai_and_prompt_settings()
        except (OSError, ValueError) as exc:
            self._show_problem(str(exc))
            self.url_input.setFocus()
            return
        settings = self.settings
        lecture_name = " ".join(self.lecture_name_input.text().split())
        if settings.auto_terminology and target_stage in {"all", "transcribe"} and not lecture_name:
            self._show_problem("전문용어 자동 입력을 사용하려면 강의명을 입력해주세요.")
            self.lecture_name_input.setFocus()
            return
        settings.terminology_lecture_name = lecture_name
        settings.last_url = source
        try:
            settings.save()
        except OSError as exc:
            self._append_tool_log(f"설정을 저장하지 못했습니다: {exc}")
        self.log_view.clear()
        self.current_review = None
        self.progress_bar.setValue(0)
        self.percent_label.setText("0%")
        self._pipeline_step = STAGE_TO_STEP.get(target_stage, 0)
        self._set_step_states(self._pipeline_step if target_stage != "all" else 0)
        self.start_button.setEnabled(False)
        for button in self.stage_buttons:
            button.setEnabled(False)
        self.status_title.setText("준비 중")
        self.detail_label.setText("필요한 도구와 저장된 작업을 확인하고 있어요.")
        self._set_status_mode("running")
        worker = PipelineWorker(source, copy.deepcopy(settings), target_stage=target_stage)
        self.pipeline_worker = worker
        worker.progress.connect(self._on_pipeline_progress)
        worker.log.connect(self._append_pipeline_log)
        worker.result.connect(self._on_pipeline_result)
        worker.audio_edit_ready.connect(self._on_audio_edit_ready)
        worker.review_ready.connect(self._on_transcript_review_ready)
        worker.cancelled.connect(self._on_pipeline_cancelled)
        worker.error.connect(self._on_pipeline_error)
        self._launch_worker(worker, on_finished=self._on_pipeline_finished)

    @Slot()
    def _cancel_pipeline(self) -> None:
        if self.pipeline_worker:
            self.pipeline_worker.cancel()
            self.cancel_button.setEnabled(False)
            self.detail_label.setText("현재 작업을 안전하게 멈추고 있어요.")
            self._append_pipeline_log("작업 중지를 요청했습니다.")

    @Slot(int, str, str)
    def _on_pipeline_progress(self, percent: int, stage: str, detail: str) -> None:
        self.progress_bar.setValue(percent)
        self.percent_label.setText(f"{percent}%")
        info = STAGE_INFO.get(stage)
        if info is None:
            # An unknown stage name keeps the current step instead of guessing from words.
            self.status_title.setText("작업 중")
            self.detail_label.setText(detail.strip()[:120] or "작업을 진행하고 있어요.")
            self._set_step_states(self._pipeline_step)
            return
        step, headline, line = info
        self._pipeline_step = step
        if stage in STAGES_WITH_LIVE_DETAIL and detail.strip() and len(detail.strip()) <= 80:
            line = f"{line} {detail.strip()}"
        self.status_title.setText(headline)
        self.detail_label.setText(line)
        if stage in STAGES_COMPLETING_STEP:
            self._set_step_states(step, done_through=step)
        else:
            self._set_step_states(step)

    def _append_pipeline_log(self, message: str) -> None:
        if message.strip():
            self.log_view.appendPlainText(message.strip())
            scrollbar = self.log_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @Slot(object)
    def _on_pipeline_result(self, result: PipelineResult) -> None:
        self.last_result = result
        stage = result.completed_stage
        headline, message = STAGE_DONE_TEXT.get(
            stage, ("작업이 끝났어요", "결과를 저장했어요.")
        )
        step = STAGE_TO_STEP.get(stage, 3)
        self._set_step_states(None, done_through=step)
        self.status_title.setText(headline)
        self.result_title.setText(result.video_title)
        self.result_message.setText(message)
        self.result_path.setText(str(result.output_dir))
        buttons = (
            (self.open_notes_button, result.notes_path),
            (self.open_transcript_button, result.transcript_path),
            (self.open_video_button, result.video_path),
        )
        for button, path in buttons:
            button.setVisible(path is not None)
        # The folder button is the main action when there is no note to open.
        self._set_button_role(
            self.open_folder_button,
            "SecondaryButton" if result.notes_path is not None else "PrimaryButton",
        )
        self.progress_bar.setValue(100)
        self.percent_label.setText("100%")
        pill = "모두 완료" if stage == "notes" else f"{STEP_LABELS[step]} 완료"
        self._set_status_mode("done", pill=pill)

    @Slot(object)
    def _on_audio_edit_ready(self, request: object) -> None:
        worker = self.pipeline_worker
        if not worker:
            return
        try:
            self.status_title.setText("쉬는 시간 정리")
            self.detail_label.setText("소리를 들으며 수업이 아닌 구간을 빼주세요.")
            self._append_pipeline_log("오디오 파형이 준비됐습니다. 편집 창에서 쉬는 시간과 음악 구간을 확인해주세요.")
            dialog = AudioEditorDialog(request, self)
            self.audio_editor = dialog
            self._notify_action_required(
                "쉬는 시간 확인이 필요합니다",
                "강의 파형이 준비됐습니다. 제외할 쉬는 시간과 음악 구간을 확인해주세요.",
            )
            result = dialog.exec()
            exclusions = dialog.exclusions if result == QDialog.DialogCode.Accepted else None
            dialog.release_media()
            dialog.deleteLater()
            self.audio_editor = None
            worker.submit_audio_edit(exclusions)
            if exclusions is not None:
                self._append_pipeline_log(
                    f"오디오 정리 적용: 수업이 아닌 구간 {len(exclusions)}개를 강의 글에서 뺍니다."
                )
        except Exception as exc:  # never leave the worker waiting on a dead dialog
            self.audio_editor = None
            worker.cancel()
            self._append_pipeline_log(f"오디오 편집 창에서 오류가 났습니다: {exc!r}")
            self._show_problem("쉬는 시간 정리 화면을 열지 못해 작업을 멈췄어요.", raw=repr(exc))

    @Slot(object)
    def _on_transcript_review_ready(self, review: TranscriptReview) -> None:
        try:
            self.current_review = review
            self.review_preview.setPlainText(review.preview_text)
            self.review_preview.moveCursor(QTextCursor.MoveOperation.Start)
            self.review_metrics.setText(review.metrics)
            if review.warnings:
                self.review_warnings.setText("\n".join(f"• {item}" for item in review.warnings))
                set_pill(self.review_badge, "확인 필요", "warning")
            else:
                self.review_warnings.setText(
                    "자동 확인은 통과했어요. 앞부분과 끝부분, 전문용어만 한 번 살펴보세요."
                )
                set_pill(self.review_badge, "자연스러워요", "success")
            self.review_headline.setText(review.headline)
            self.review_approve_button.setEnabled(True)
            self.review_reject_button.setEnabled(True)
            self.status_title.setText("강의 글을 한 번 확인해주세요")
            self._pipeline_step = 2
            self._set_step_states(2)
            target = self.pipeline_worker.target_stage if self.pipeline_worker else "all"
            if target == "transcribe":
                self.review_approve_button.setText("확인했어요 · 강의 글 저장")
                self.detail_label.setText("노트를 만들기 전에 반복된 문장이나 잘못 들은 전문용어가 없는지 살펴보세요. 확인하면 강의 글을 저장합니다.")
                self._append_pipeline_log("강의 글 만들기가 끝났습니다. 내용을 확인한 뒤 저장해주세요.")
            else:
                self.review_approve_button.setText("확인했어요 · 노트 만들기")
                self.detail_label.setText("노트를 만들기 전에 반복된 문장이나 잘못 들은 전문용어가 없는지 살펴보세요. 확인하기 전에는 AI 노트 작성이 시작되지 않아요.")
                self._append_pipeline_log("강의 글 만들기가 끝났습니다. 화면에서 내용을 확인한 뒤 노트를 만들어주세요.")
            self._set_status_mode("review")
            self._notify_action_required(
                "강의 글 확인이 필요합니다",
                "전사가 끝났습니다. 내용을 확인해야 다음 단계로 진행할 수 있습니다.",
            )
        except Exception as exc:
            self.current_review = None
            if self.pipeline_worker:
                self.pipeline_worker.cancel()
            self._append_pipeline_log(f"강의 글 확인 화면에서 오류가 났습니다: {exc!r}")
            self._show_problem("강의 글 확인 화면을 만들지 못해 작업을 멈췄어요.", raw=repr(exc))

    def _submit_transcript_review(self, approved: bool) -> None:
        if not self.pipeline_worker or not self.current_review:
            return
        self.review_approve_button.setEnabled(False)
        self.review_reject_button.setEnabled(False)
        worker = self.pipeline_worker
        worker.submit_review(approved)
        if approved:
            if worker.target_stage == "transcribe":
                self.status_title.setText("강의 글 저장 중")
                self.detail_label.setText("확인한 강의 글을 저장하고 있어요.")
                self._set_step_states(2)
                self._append_pipeline_log("강의 글 확인 완료: 이 단계의 결과를 저장합니다.")
            else:
                self.status_title.setText("노트 정리 중")
                self.detail_label.setText("확인한 강의 글로 노트를 만들고 있어요.")
                self._set_step_states(3)
                self._pipeline_step = 3
                self._append_pipeline_log("강의 글 확인 완료: 노트 정리를 시작합니다.")
            self._set_status_mode("running")
            self.cancel_button.setEnabled(True)
        else:
            self._append_pipeline_log("강의 글 확인 단계에서 작업을 멈췄습니다.")

    def _open_review_transcript(self) -> None:
        if self.current_review:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_review.text_path)))

    @Slot(str)
    def _on_pipeline_cancelled(self, message: str) -> None:
        self.status_title.setText("작업을 멈췄어요")
        self.detail_label.setText("끝난 단계는 저장됐어요. 같은 링크로 다시 시작하면 이어집니다.")
        self.stepper.set_states(
            [
                "done" if i < self._pipeline_step else "pending"
                for i in range(len(STEP_LABELS))
            ]
        )
        self._append_pipeline_log(message)
        self._set_status_mode("stopped")

    @Slot(str)
    def _on_pipeline_error(self, message: str) -> None:
        self.status_title.setText("문제가 생겨 멈췄어요")
        first_line = message.strip().splitlines()[0] if message.strip() else "알 수 없는 오류"
        self.detail_label.setText(first_line[:200])
        self._set_step_states(self._pipeline_step, error=True)
        self._append_pipeline_log(f"오류: {message}")
        self._set_status_mode("stopped", tone="danger", pill="오류")
        self._show_problem(first_line[:300], raw=message)

    def _on_pipeline_finished(self) -> None:
        self.start_button.setEnabled(True)
        for button in self.stage_buttons:
            button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.pipeline_worker = None
        if self._status_mode in {"running", "review"}:
            # No result/cancel/error arrived; never leave a live-looking panel behind.
            self.status_title.setText("작업이 끝났어요")
            self._set_status_mode("stopped")
        self.current_review = None
        self.review_panel.setVisible(False)
        self.review_approve_button.setEnabled(False)
        self.review_reject_button.setEnabled(False)

    # -------------------------------------------------------------- AI provider

    def _current_provider(self) -> str:
        return normalize_provider(str(self.provider_combo.currentData() or ""))

    def _auth_provider(self) -> str:
        combo = getattr(self, "auth_provider_combo", None)
        if combo is None:
            return self._current_provider()
        return normalize_provider(str(combo.currentData() or ""))

    def _remember_model_selection(self) -> None:
        """Keep the combos' choice for the provider they currently show."""
        notes_model = str(self.notes_model_combo.currentData() or "")
        terminology_model = str(self.terminology_model_combo.currentData() or "")
        if notes_model and terminology_model:
            self.settings.set_ai_selection(
                self._active_provider,
                notes_model,
                str(self.reasoning_combo.currentData() or ""),
                terminology_model,
            )

    @Slot()
    def _change_provider(self) -> None:
        self._remember_model_selection()
        provider = self._current_provider()
        self._active_provider = provider
        self._populate_model_combos()
        self._load_model_catalog(provider)
        if self._auth_provider() != provider:
            # Changing the combo also checks the new provider's login status.
            self._select_data(self.auth_provider_combo, provider)
        self._sync_ai_badge()

    @Slot()
    def _on_notes_model_changed(self) -> None:
        self._populate_reasoning_combo()

    @Slot()
    def _refresh_model_catalog(self) -> None:
        self._load_model_catalog(self._current_provider(), force=True)

    def _startup_model_catalog(self) -> None:
        self._load_model_catalog(self._current_provider())

    def _load_model_catalog(self, provider: str, force: bool = False) -> None:
        catalog = self._model_catalogs.get(provider)
        if provider in self._loading_catalogs or (catalog and catalog.live and not force):
            return
        self._loading_catalogs.add(provider)
        if provider == self._current_provider():
            self.refresh_models_button.setEnabled(False)
            self.model_source_label.setText(
                f"{TOOL_NAMES[provider]}에서 최신 모델 목록을 불러오는 중입니다…"
            )
        worker = ModelListWorker(provider)
        worker.result.connect(self._on_model_catalog)
        self._launch_worker(worker, on_finished=partial(self._on_model_catalog_finished, provider))

    @Slot(object)
    def _on_model_catalog(self, catalog: ModelCatalog) -> None:
        self._model_catalogs[catalog.provider] = catalog
        if not catalog.live and catalog.detail:
            self._append_tool_log(
                f"{TOOL_NAMES[catalog.provider]} 모델 목록을 불러오지 못해 기본 목록을 사용합니다: "
                f"{catalog.detail}"
            )
        if catalog.provider == self._current_provider():
            self._remember_model_selection()
            self._populate_model_combos()

    def _on_model_catalog_finished(self, provider: str) -> None:
        self._loading_catalogs.discard(provider)
        if provider == self._current_provider():
            self.refresh_models_button.setEnabled(True)
            self._update_model_source_label()

    def _catalog(self) -> ModelCatalog:
        provider = self._current_provider()
        return self._model_catalogs.get(provider) or fallback_catalog(provider)

    def _populate_model_combos(self) -> None:
        provider = self._current_provider()
        catalog = self._catalog()
        notes_model, reasoning_effort, terminology_model = self.settings.ai_selection(provider)
        for combo, selected in (
            (self.notes_model_combo, notes_model),
            (self.terminology_model_combo, terminology_model),
        ):
            combo.blockSignals(True)
            combo.clear()
            for model in catalog.models:
                combo.addItem(model.label, model.value)
                combo.setItemData(
                    combo.count() - 1,
                    f"{model.value}\n{model.description}".strip(),
                    Qt.ItemDataRole.ToolTipRole,
                )
            if selected and combo.findData(selected) < 0:
                # A model saved earlier that the current catalog no longer lists.
                combo.insertItem(0, f"{selected} · 목록에 없음", selected)
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)
        self._populate_reasoning_combo(reasoning_effort)
        self._update_model_source_label()

    def _update_model_source_label(self) -> None:
        provider = self._current_provider()
        if provider in self._loading_catalogs:
            return
        if self._catalog().live:
            text = f"{PROVIDER_NAMES[provider]}가 알려준 현재 계정의 최신 모델 목록입니다."
        else:
            text = (
                f"기본 모델 목록입니다. {TOOL_NAMES[provider]}를 설치하면 "
                "계정에서 쓸 수 있는 최신 목록을 불러옵니다."
            )
        self.model_source_label.setText(text)

    def _populate_reasoning_combo(self, preferred: str | None = None) -> None:
        if preferred is None:
            preferred = str(self.reasoning_combo.currentData() or "")
        selected = str(self.notes_model_combo.currentData() or "")
        model = next((item for item in self._catalog().models if item.value == selected), None)
        efforts = model.efforts if model else ("low", "medium", "high", "xhigh", "max")
        default = model.default_effort if model else ""
        self.reasoning_combo.blockSignals(True)
        self.reasoning_combo.clear()
        for effort in efforts:
            label = EFFORT_LABELS.get(effort, effort)
            self.reasoning_combo.addItem(f"{label} · 기본" if effort == default else label, effort)
            self.reasoning_combo.setItemData(
                self.reasoning_combo.count() - 1, effort, Qt.ItemDataRole.ToolTipRole
            )
        if not efforts:
            self.reasoning_combo.addItem("이 모델은 조절할 수 없어요", "")
        self.reasoning_combo.setEnabled(bool(efforts))
        index = self.reasoning_combo.findData(preferred)
        if index < 0:
            index = max(0, self.reasoning_combo.findData(default))
        self.reasoning_combo.setCurrentIndex(index)
        self.reasoning_combo.blockSignals(False)

    # ------------------------------------------------------------ auth / login

    def _startup_auth_status(self) -> None:
        self._run_auth_action("status")

    @Slot()
    def _on_auth_provider_changed(self) -> None:
        self._update_auth_tool_tip(self._auth_provider())
        self._run_auth_action("status")

    @Slot()
    def _on_login_button(self) -> None:
        if self._login_worker is not None:
            self._login_worker.cancel()
            self.login_button.setEnabled(False)
            self.login_button.setText("취소하는 중…")
            self._append_tool_log("연결을 취소하고 있습니다…")
            return
        self._run_auth_action("login")

    def _run_auth_action(self, action: str) -> None:
        if self._login_worker is not None:
            return  # a running login owns the auth controls; status checks must not clobber it
        provider = self._auth_provider()
        worker = AuthWorker(action, provider)
        worker.log.connect(self._append_tool_log)
        worker.result.connect(self._on_auth_result)
        worker.error.connect(self._on_auth_error)
        if action == "login":
            self._login_worker = worker
            self._login_provider = provider
            self._append_tool_log(f"{PROVIDER_NAMES[provider]} 연결을 시작합니다.")
            self.auth_detail.setText("브라우저에서 로그인을 진행해주세요.")
        else:
            self._auth_pending += 1
            if action == "status":
                self.auth_detail.setText(f"{PROVIDER_NAMES[provider]} 로그인 상태를 확인하고 있어요.")
        self._update_auth_controls()
        self._launch_worker(worker, on_finished=partial(self._on_auth_finished, worker))

    def _update_auth_controls(self) -> None:
        logging_in = self._login_worker is not None
        busy = logging_in or self._auth_pending > 0
        self.login_button.setEnabled(not busy or (logging_in and not self._login_worker.cancel_event.is_set()))
        self.login_button.setText("취소" if logging_in else "연결하기")
        self._set_button_role(self.login_button, "SecondaryButton" if logging_in else "PrimaryButton")
        self.logout_button.setEnabled(not busy)
        self.refresh_auth_button.setEnabled(not busy)
        self.auth_provider_combo.setEnabled(not logging_in)
        self.provider_combo.setEnabled(not logging_in)
        self.auth_code_row.setVisible(logging_in and self._login_provider == "claude")

    def _submit_auth_code(self) -> None:
        code = self.auth_code_input.text().strip()
        if not code or self._login_worker is None:
            return
        self._login_worker.send_input(code)
        self.auth_code_input.clear()
        self._append_tool_log("인증 코드를 제출했습니다. 확인 중입니다…")

    @Slot(object)
    def _on_auth_result(self, value: tuple[str, AuthStatus]) -> None:
        provider, status = value
        name = PROVIDER_NAMES[provider]
        tool = find_tool(provider)
        if status.logged_in:
            text, tone = f"{name} 연결됨", "success"
        elif status.available:
            text, tone = f"{name} 로그인 필요", "warning"
        else:
            text, tone = "연결 도구 설치 필요", "warning"
        if provider == self._auth_provider():
            self.auth_detail.setText(status.detail)
            self.auth_detail.setToolTip(f"연결 도구: {tool or '찾지 못함'}")
            set_pill(self.auth_pill, text, tone)
        if provider == self._current_provider():
            set_pill(self.ai_badge, text, tone, status.detail)

    def _sync_ai_badge(self) -> None:
        set_pill(self.ai_badge, "확인 중", "neutral")
        if self._login_worker is None:
            self._run_auth_action("status")

    @Slot(str)
    def _on_auth_error(self, message: str) -> None:
        worker = self.sender()
        action = getattr(worker, "action", "status")
        self._append_tool_log(f"오류: {message}")
        if action == "login" and getattr(worker, "cancel_event", None) is not None and worker.cancel_event.is_set():
            self._append_tool_log("연결을 취소했습니다.")
            return
        if action == "status":
            first = message.strip().splitlines()[0] if message.strip() else "알 수 없는 오류"
            self.auth_detail.setText(f"연결 상태를 확인하지 못했어요. {first[:160]}")
            set_pill(self.auth_pill, "확인 실패", "danger", message)
            if worker is not None and getattr(worker, "provider", "") == self._current_provider():
                set_pill(self.ai_badge, "확인 실패", "danger", message)
            return
        self._show_problem("AI 계정 연결 작업을 마치지 못했어요.", raw=message)

    def _on_auth_finished(self, worker: AuthWorker) -> None:
        if worker is self._login_worker:
            self._login_worker = None
            self._login_provider = ""
            self.auth_code_input.clear()
            # Re-read the login state so the pill reflects the finished login.
            self._update_auth_controls()
            self._run_auth_action("status")
            return
        self._auth_pending = max(0, self._auth_pending - 1)
        self._update_auth_controls()

    # ------------------------------------------------------------------ updater

    def _run_ytdlp_update(self) -> None:
        channel = str(self.update_channel_combo.currentData())
        self.settings.yt_dlp_channel = channel
        self._persist_ui_state()
        self.update_button.setEnabled(False)
        set_pill(self.update_pill, "확인 중", "accent")
        self._append_tool_log(f"영상 가져오기 도구({channel})의 업데이트를 확인합니다.")
        worker = UpdateWorker(channel)
        worker.log.connect(self._append_tool_log)
        worker.result.connect(self._on_update_result)
        worker.error.connect(self._on_tool_error)
        self._launch_worker(worker, on_finished=self._on_update_finished)

    def _on_update_finished(self) -> None:
        self.update_button.setEnabled(True)

    @Slot(object)
    def _on_update_result(self, result: UpdateResult) -> None:
        if result.updated:
            message = f"영상 가져오기 도구를 {result.previous}에서 {result.current}(으)로 업데이트했습니다."
            set_pill(self.update_pill, f"업데이트됨 · {result.current}", "success")
        else:
            message = f"이미 최신 버전입니다: {result.current}"
            set_pill(self.update_pill, f"최신 버전 · {result.current}", "success")
        self._append_tool_log(message)
        QMessageBox.information(self, "업데이트 완료", message)

    def _append_tool_log(self, message: str) -> None:
        if message.strip():
            self.tool_log.appendPlainText(message.strip())
            scrollbar = self.tool_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @Slot(str)
    def _on_tool_error(self, message: str) -> None:
        self._append_tool_log(f"오류: {message}")
        set_pill(self.update_pill, "업데이트 실패", "danger", message)
        self._show_problem("업데이트를 마치지 못했어요. 인터넷 연결을 확인하고 다시 시도해주세요.", raw=message)

    # ------------------------------------------------------------------ workers

    def _launch_worker(self, worker: object, on_finished: object = None) -> None:
        """Run a worker on its own thread.

        Completion is reported through bound slots only (``completed`` carries the
        worker itself), so GUI code never runs on the worker thread.
        """
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        self._finish_callbacks[worker] = on_finished
        completed = getattr(worker, "completed", None)
        if completed is not None:
            completed.connect(self._on_worker_completed)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._background.append((thread, worker))
        thread.start()

    @Slot(object)
    def _on_worker_completed(self, worker: object) -> None:
        callback = self._finish_callbacks.pop(worker, None)
        if callable(callback):
            callback()

    @Slot()
    def _on_thread_finished(self) -> None:
        thread = self.sender()
        self._background = [pair for pair in self._background if pair[0] is not thread]
        if thread is not None:
            thread.deleteLater()

    # ------------------------------------------------------------ open results

    def _open_last_notes(self) -> None:
        if self.last_result and self.last_result.notes_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_result.notes_path)))

    def _open_last_transcript(self) -> None:
        if self.last_result and self.last_result.transcript_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_result.transcript_path)))

    def _open_last_video(self) -> None:
        if self.last_result and self.last_result.video_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_result.video_path)))

    def _open_last_folder(self) -> None:
        if self.last_result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_result.output_dir)))

    def _show_problem(self, message: str, raw: str | None = None) -> None:
        """Friendly dialog; the raw technical detail goes to the log instead."""
        if raw and raw.strip() and raw.strip() != message.strip():
            self._append_tool_log(f"자세한 내용: {raw.strip()}")
            if hasattr(self, "log_view"):
                self._append_pipeline_log(f"자세한 내용: {raw.strip()}")
        QMessageBox.warning(self, "잠깐, 확인이 필요해요", message)

    # ------------------------------------------------------------------- close

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.pipeline_worker:
            answer = QMessageBox.question(
                self,
                "작업을 멈추고 닫을까요?",
                "진행 중인 작업을 멈추고 앱을 닫습니다. 끝난 단계는 저장되어 있어요.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.pipeline_worker.cancel()
        if self.live_page.running:
            answer = QMessageBox.question(
                self,
                "실시간 자막 진행 중",
                "실시간 자막을 멈추고 창을 닫을까요? 지금까지의 자막은 저장되어 있습니다.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._persist_timer.stop()
        self._persist_ui_state()
        if self._save_geometry_on_close:
            QSettings("Local Tools", "Lecture Scribe").setValue("window/geometry", self.saveGeometry())
        self.live_page.shutdown()
        if self._login_worker is not None:
            self._login_worker.cancel()
        for thread, worker in list(self._background):
            try:
                cancel = getattr(worker, "cancel", None)
                if callable(cancel):
                    cancel()
                thread.quit()
                thread.wait(5000)
            except RuntimeError:
                continue  # the C++ object is already gone
        if self._notification_tray:
            self._notification_tray.hide()
        event.accept()
