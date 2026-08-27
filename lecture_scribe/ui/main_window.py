from __future__ import annotations

import platform
from pathlib import Path

from PySide6.QtCore import QSize, QThread, QTimer, QUrl, Qt
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
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

from ..codex_client import AuthStatus, find_codex
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
from .workers import AuthWorker, PipelineWorker, UpdateWorker


ASSET_DIR = Path(__file__).resolve().parent / "assets"
IS_WINDOWS = platform.system() == "Windows"
INSTALLER_NAME = "install.bat" if IS_WINDOWS else "install.command"
ACCELERATOR_DESCRIPTION = (
    "빠르고 정확한 음성 인식을 위해 NVIDIA CUDA 가속을 사용합니다."
    if IS_WINDOWS
    else "빠르고 정확한 음성 인식을 위해 Apple Silicon의 MLX Metal 가속을 사용합니다."
)
VIDEO_ENCODER_NAME = "NVIDIA NVENC" if IS_WINDOWS else "Apple VideoToolbox"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self._background: list[tuple[QThread, object]] = []
        self.pipeline_worker: PipelineWorker | None = None
        self.last_result: PipelineResult | None = None
        self.current_review: TranscriptReview | None = None
        self.audio_editor: AudioEditorDialog | None = None
        self.setWindowTitle("Lecture Scribe · 강의를 내 노트로")
        self.setMinimumSize(1080, 720)
        self.resize(1280, 860)
        self._notification_tray: QSystemTrayIcon | None = None
        self._build_ui()
        self._setup_notifications()
        self._refresh_local_status()
        QTimer.singleShot(400, lambda: self._run_auth_action("status"))

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("AppCanvas")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())

        workspace = QFrame()
        workspace.setObjectName("Workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(30, 22, 30, 24)
        workspace_layout.setSpacing(18)
        workspace_layout.addWidget(self._build_workspace_header())
        self.tabs = QStackedWidget()
        self.tabs.setObjectName("PageStack")
        self.tabs.addWidget(self._build_transcribe_tab())
        self.tabs.addWidget(self._build_prompt_tab())
        self.tabs.addWidget(self._build_tools_tab())
        workspace_layout.addWidget(self.tabs, 1)
        layout.addWidget(workspace, 1)
        self.setCentralWidget(root)

    def _setup_notifications(self) -> None:
        icon = QIcon(str(ASSET_DIR / "nav-new.svg"))
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

    def _on_tray_activated(
        self, reason: QSystemTrayIcon.ActivationReason
    ) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self._restore_from_notification()

    def _restore_from_notification(self) -> None:
        self.tabs.setCurrentIndex(0)
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
        sidebar.setFixedWidth(238)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(8)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(11)
        mark = QFrame()
        mark.setObjectName("SidebarBrandMark")
        mark.setFixedSize(40, 40)
        mark_layout = QVBoxLayout(mark)
        mark_layout.setContentsMargins(0, 0, 0, 0)
        mark_letter = QLabel("L")
        mark_letter.setObjectName("SidebarBrandLetter")
        mark_letter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark_layout.addWidget(mark_letter)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand_name = QLabel("Lecture Scribe")
        brand_name.setObjectName("SidebarBrand")
        brand_meta = QLabel("STUDY STUDIO")
        brand_meta.setObjectName("SidebarEyebrow")
        brand_text.addWidget(brand_name)
        brand_text.addWidget(brand_meta)
        brand_row.addWidget(mark)
        brand_row.addLayout(brand_text, 1)
        layout.addLayout(brand_row)
        layout.addSpacing(28)

        section = QLabel("작업 공간")
        section.setObjectName("NavSection")
        layout.addWidget(section)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: list[QPushButton] = []
        for index, (label, icon_name) in enumerate(
            [
                ("새 강의 노트", "nav-new.svg"),
                ("노트 작성 방식", "nav-note.svg"),
                ("연결과 업데이트", "nav-connect.svg"),
            ]
        ):
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setIcon(QIcon(str(ASSET_DIR / icon_name)))
            button.setIconSize(QSize(20, 20))
            button.setAccessibleName(label)
            button.clicked.connect(lambda _checked, value=index: self._switch_page(value))
            self.nav_group.addButton(button)
            self.nav_buttons.append(button)
            layout.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        layout.addStretch(1)

        environment = QFrame()
        environment.setObjectName("SidebarStatusCard")
        environment_layout = QVBoxLayout(environment)
        environment_layout.setContentsMargins(13, 13, 13, 13)
        environment_layout.setSpacing(8)
        environment_title = QLabel("작업 환경")
        environment_title.setObjectName("SidebarCardTitle")
        environment_layout.addWidget(environment_title)
        self.gpu_badge = QLabel("컴퓨터 확인 중")
        self.ai_badge = QLabel("ChatGPT 확인 중")
        self._set_badge(self.gpu_badge, "컴퓨터 확인 중", "neutral")
        self._set_badge(self.ai_badge, "ChatGPT 확인 중", "neutral")
        environment_layout.addWidget(self.gpu_badge)
        environment_layout.addWidget(self.ai_badge)
        layout.addWidget(environment)
        privacy = QLabel("로컬 저장 · 파일은 내 컴퓨터에")
        privacy.setObjectName("SidebarFooter")
        privacy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(privacy)
        return sidebar

    def _build_workspace_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("WorkspaceHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.page_title = QLabel("새 강의 노트")
        self.page_title.setObjectName("PageTitle")
        self.page_subtitle = QLabel("영상 링크 하나로 공부하기 좋은 노트를 만드세요.")
        self.page_subtitle.setObjectName("PageSubtitle")
        titles.addWidget(self.page_title)
        titles.addWidget(self.page_subtitle)
        layout.addLayout(titles, 1)
        private_chip = QLabel("내 컴퓨터에서 안전하게 처리")
        private_chip.setObjectName("PrivacyChip")
        layout.addWidget(private_chip, 0, Qt.AlignmentFlag.AlignVCenter)
        return header

    def _switch_page(self, index: int) -> None:
        page_copy = [
            ("새 강의 노트", "영상 링크 하나로 공부하기 좋은 노트를 만드세요."),
            ("노트 작성 방식", "내가 공부하는 방식에 맞게 노트의 말투와 구성을 정하세요."),
            ("연결과 업데이트", "컴퓨터와 ChatGPT 연결 상태를 한곳에서 관리하세요."),
        ]
        self.tabs.setCurrentIndex(index)
        self.page_title.setText(page_copy[index][0])
        self.page_subtitle.setText(page_copy[index][1])
        self.nav_buttons[index].setChecked(True)
        self.nav_buttons[index].setFocus(Qt.FocusReason.OtherFocusReason)

    def _set_advanced_mode_ui(self, enabled: bool) -> None:
        self.start_button.setVisible(not enabled)
        if enabled:
            self.dock_title.setText("원하는 단계를 선택해 시작하세요")
            self.dock_helper.setText("위의 네 단계 중 필요한 작업만 실행할 수 있습니다.")
        else:
            self.dock_title.setText("준비되면 바로 시작하세요")
            self.dock_helper.setText("중간에 멈춰도 완료한 단계부터 이어집니다.")

    def _build_header(self) -> QFrame:
        card = QFrame()
        card.setObjectName("HeaderCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(16)
        top = QHBoxLayout()
        top.setSpacing(16)
        brand = QFrame()
        brand.setObjectName("BrandMark")
        brand.setFixedSize(50, 50)
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_initial = QLabel("L")
        brand_initial.setObjectName("BrandInitial")
        brand_initial.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_layout.addWidget(brand_initial)
        top.addWidget(brand, 0, Qt.AlignmentFlag.AlignVCenter)
        text = QVBoxLayout()
        text.setSpacing(3)
        eyebrow = QLabel("LECTURE SCRIBE")
        eyebrow.setObjectName("Eyebrow")
        title = QLabel("영상 한 편이, 공부하기 좋은 노트로")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("링크만 넣어주세요. 강의를 듣고 글로 옮긴 뒤 핵심 노트까지 정리해드려요.")
        subtitle.setObjectName("HeroSubtitle")
        text.addWidget(eyebrow)
        text.addWidget(title)
        text.addWidget(subtitle)
        top.addLayout(text, 1)
        badges = QVBoxLayout()
        badges.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        badges.setSpacing(7)
        self.gpu_badge = QLabel("컴퓨터 확인 중")
        self.ai_badge = QLabel("ChatGPT 확인 중")
        self._set_badge(self.gpu_badge, "컴퓨터 확인 중", "neutral")
        self._set_badge(self.ai_badge, "ChatGPT 확인 중", "neutral")
        badges.addWidget(self.gpu_badge, 0, Qt.AlignmentFlag.AlignRight)
        badges.addWidget(self.ai_badge, 0, Qt.AlignmentFlag.AlignRight)
        top.addLayout(badges)
        layout.addLayout(top)

        flow = QFrame()
        flow.setObjectName("FlowStrip")
        flow_layout = QHBoxLayout(flow)
        flow_layout.setContentsMargins(14, 10, 14, 10)
        flow_layout.setSpacing(9)
        for index, label in enumerate(
            ["영상 가져오기", "쉬는 시간 정리", "강의 글 확인", "내 노트 완성"],
            start=1,
        ):
            number = QLabel(str(index))
            number.setObjectName("FlowNumber")
            step = QLabel(label)
            step.setObjectName("FlowStep")
            flow_layout.addWidget(number)
            flow_layout.addWidget(step)
            if index < 4:
                arrow = QLabel("›")
                arrow.setObjectName("FlowArrow")
                flow_layout.addWidget(arrow)
        flow_layout.addStretch(1)
        layout.addWidget(flow)
        self._apply_shadow(card, blur=26, y_offset=7, alpha=24)
        return card

    def _build_transcribe_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 12, 0, 0)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 6)
        layout.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("ProjectHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 20)
        hero_layout.setSpacing(18)
        hero_top = QHBoxLayout()
        hero_copy = QVBoxLayout()
        hero_copy.setSpacing(5)
        hero_kicker = QLabel("새 프로젝트")
        hero_kicker.setObjectName("HeroKicker")
        hero_title = QLabel("강의 한 편을 온전한 공부 자료로")
        hero_title.setObjectName("ProjectHeroTitle")
        hero_subtitle = QLabel(
            "영상만 골라주세요. 쉬는 시간을 걷어내고, 내용을 확인한 뒤 내 노트로 정리합니다."
        )
        hero_subtitle.setObjectName("ProjectHeroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_copy.addWidget(hero_kicker)
        hero_copy.addWidget(hero_title)
        hero_copy.addWidget(hero_subtitle)
        hero_top.addLayout(hero_copy, 1)
        hero_badge = QFrame()
        hero_badge.setObjectName("HeroBadge")
        hero_badge_layout = QVBoxLayout(hero_badge)
        hero_badge_layout.setContentsMargins(14, 10, 14, 10)
        hero_badge_layout.setSpacing(1)
        hero_badge_number = QLabel("4")
        hero_badge_number.setObjectName("HeroBadgeNumber")
        hero_badge_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_badge_label = QLabel("단계 자동 진행")
        hero_badge_label.setObjectName("HeroBadgeLabel")
        hero_badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_badge_layout.addWidget(hero_badge_number)
        hero_badge_layout.addWidget(hero_badge_label)
        hero_top.addWidget(hero_badge)
        hero_layout.addLayout(hero_top)
        journey = QHBoxLayout()
        journey.setSpacing(8)
        for index, label in enumerate(
            ["영상 가져오기", "쉬는 시간 정리", "강의 글 확인", "내 노트 완성"],
            start=1,
        ):
            step = QFrame()
            step.setProperty("journeyStep", True)
            step_layout = QHBoxLayout(step)
            step_layout.setContentsMargins(10, 8, 10, 8)
            step_layout.setSpacing(7)
            number = QLabel(str(index))
            number.setObjectName("JourneyNumber")
            number.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text = QLabel(label)
            text.setObjectName("JourneyText")
            step_layout.addWidget(number)
            step_layout.addWidget(text)
            journey.addWidget(step, 1)
        hero_layout.addLayout(journey)
        layout.addWidget(hero)

        source_card, source = self._card(
            "어떤 강의를 정리할까요?",
            "YouTube 링크를 붙여넣거나 내 컴퓨터의 영상·음성 파일을 골라주세요.",
        )
        url_label = self._field_label("YouTube 링크 또는 영상·음성 파일")
        self.url_input = QLineEdit(self.settings.last_url)
        self.url_input.setPlaceholderText("YouTube 링크를 붙여넣거나 파일을 선택하세요")
        self.url_input.setClearButtonEnabled(True)
        self.url_input.setAccessibleName("YouTube 영상 링크 또는 로컬 영상·음성 파일")
        paste_button = QPushButton("붙여넣기")
        paste_button.setAccessibleName("클립보드에서 YouTube URL 붙여넣기")
        paste_button.clicked.connect(
            lambda: self.url_input.setText(QApplication.clipboard().text().strip())
        )
        url_row = QHBoxLayout()
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(paste_button)
        choose_media_button = QPushButton("파일 선택")
        choose_media_button.setAccessibleName("전사할 영상 또는 음성 파일 선택")
        choose_media_button.clicked.connect(self._choose_media_file)
        url_row.addWidget(choose_media_button)
        source.addWidget(url_label)
        source.addLayout(url_row)

        self.use_time_range_check = QCheckBox("영상의 일부만 가져오기")
        self.use_time_range_check.setChecked(False)
        self.clip_start_input = QLineEdit(format_timecode(self.settings.clip_start_seconds))
        self.clip_start_input.setPlaceholderText("00:00:00")
        self.clip_start_input.setAccessibleName("가져올 시작 시간")
        self.clip_end_input = QLineEdit(format_timecode(self.settings.clip_end_seconds))
        self.clip_end_input.setPlaceholderText("비워두면 영상 끝까지")
        self.clip_end_input.setAccessibleName("가져올 종료 시간")
        self.time_range_panel = QWidget()
        range_row = QHBoxLayout(self.time_range_panel)
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.addWidget(QLabel("시작"))
        range_row.addWidget(self.clip_start_input, 1)
        range_row.addWidget(QLabel("종료"))
        range_row.addWidget(self.clip_end_input, 1)
        self.time_range_panel.setVisible(False)
        self.use_time_range_check.toggled.connect(self.time_range_panel.setVisible)
        range_help = QLabel(
            "잘 모르겠다면 비워두세요. 다음 화면에서 소리를 들으며 수업 전 음악과 쉬는 시간을 쉽게 뺄 수 있어요."
        )
        range_help.setObjectName("HelperText")
        range_help.setWordWrap(True)
        source.addWidget(self.use_time_range_check)
        source.addWidget(self.time_range_panel)
        source.addWidget(range_help)

        folder_label = self._field_label("노트를 저장할 곳")
        self.folder_input = QLineEdit(self.settings.output_dir)
        self.folder_input.setAccessibleName("노트를 저장할 폴더")
        choose_button = QPushButton("찾아보기")
        choose_button.clicked.connect(self._choose_output_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_input, 1)
        folder_row.addWidget(choose_button)
        source.addWidget(folder_label)
        source.addLayout(folder_row)
        source_card.setProperty("cardRole", "source")

        option_card, options = self._card(
            "어떻게 진행할까요?",
            "처음이라면 기본 설정 그대로 시작해도 좋습니다. 정확도를 우선해 준비해두었어요.",
        )

        self.vad_check = QCheckBox("말이 없는 긴 구간은 건너뛰기")
        self.vad_check.setChecked(self.settings.vad_filter)
        self.vad_check.setToolTip("강의 중 소리가 거의 없는 구간을 건너뛰어 반복 문장이 생길 가능성을 낮춥니다.")
        options.addWidget(self.vad_check)

        self.auto_terminology_check = QCheckBox("전문용어 자동 입력")
        self.auto_terminology_check.setChecked(self.settings.auto_terminology)
        self.auto_terminology_check.setToolTip(
            "시작할 때 강의명을 입력하면 연결된 GPT가 관련 전문용어를 만들고 "
            "Whisper 전사 문맥에 자동 적용합니다."
        )
        options.addWidget(self.auto_terminology_check)

        self.keep_video_check = QCheckBox("쉬는 시간을 뺀 영상도 보관하기")
        self.keep_video_check.setChecked(self.settings.keep_video)
        self.keep_video_check.setToolTip(
            "최대 1080p 영상을 받아 Razor에서 제외한 구간을 똑같이 제거한 "
            f"edited_lecture.mp4를 {VIDEO_ENCODER_NAME} 하드웨어 가속으로 저장합니다. "
            "완료 후 원본 영상은 정리합니다."
        )
        options.addWidget(self.keep_video_check)

        self.recognition_settings_check = QCheckBox("음성 인식 세부 설정 보기")
        options.addWidget(self.recognition_settings_check)
        recognition_panel = QFrame()
        recognition_panel.setObjectName("StagePanel")
        recognition_layout = QVBoxLayout(recognition_panel)
        recognition_layout.setContentsMargins(14, 13, 14, 13)
        recognition_layout.setSpacing(11)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(["large-v3", "large-v2", "medium"])
        self.whisper_combo.setCurrentText(self.settings.whisper_model)
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
        self.beam_spin.setToolTip("Windows CUDA 전사에서 높을수록 더 꼼꼼히 비교합니다.")
        self.cookie_combo = QComboBox()
        for label, value in [
            ("사용 안 함", "none"),
            ("Chrome 쿠키", "chrome"),
            ("Edge 쿠키", "edge"),
            ("Firefox 쿠키", "firefox"),
        ]:
            self.cookie_combo.addItem(label, value)
        self._select_data(self.cookie_combo, self.settings.cookie_browser)
        grid.addWidget(self._field_label("음성 인식 방식"), 0, 0)
        grid.addWidget(self.whisper_combo, 1, 0)
        grid.addWidget(self._field_label("강의 언어"), 0, 1)
        grid.addWidget(self.language_combo, 1, 1)
        if IS_WINDOWS:
            grid.addWidget(self._field_label("정확도 단계"), 2, 0)
            grid.addWidget(self.beam_spin, 3, 0)
            grid.addWidget(self._field_label("로그인이 필요한 영상"), 2, 1)
            grid.addWidget(self.cookie_combo, 3, 1)
        else:
            grid.addWidget(self._field_label("로그인이 필요한 영상"), 2, 0)
            grid.addWidget(self.cookie_combo, 3, 0, 1, 2)
        recognition_layout.addLayout(grid)
        checks = QHBoxLayout()
        self.keep_audio_check = QCheckBox("가져온 원본 오디오도 보관하기")
        self.keep_audio_check.setChecked(self.settings.keep_audio)
        checks.addWidget(self.keep_audio_check)
        checks.addStretch(1)
        recognition_layout.addLayout(checks)
        recognition_panel.setVisible(False)
        self.recognition_settings_check.toggled.connect(recognition_panel.setVisible)
        options.addWidget(recognition_panel)

        mode_row = QHBoxLayout()
        self.advanced_mode_check = QCheckBox("단계별로 직접 진행하기")
        self.advanced_mode_check.setToolTip(
            "영상 가져오기, 쉬는 시간 정리, 강의 글 만들기, 노트 정리를 원하는 단계만 실행합니다."
        )
        mode_help = QLabel("중간에 멈춰도 끝난 단계는 저장되며, 같은 링크로 다시 시작하면 이어집니다.")
        mode_help.setObjectName("HelperText")
        mode_help.setWordWrap(True)
        mode_row.addWidget(self.advanced_mode_check)
        mode_row.addStretch(1)
        options.addLayout(mode_row)
        options.addWidget(mode_help)

        self.start_button = QPushButton("강의 노트 만들기")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setAccessibleName("YouTube 강의 노트 만들기 시작")
        self.start_button.clicked.connect(lambda: self._start_pipeline("all"))
        self.cancel_button = QPushButton("잠시 멈추기")
        self.cancel_button.setObjectName("DangerButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel_pipeline)

        self.advanced_panel = QFrame()
        self.advanced_panel.setObjectName("StagePanel")
        advanced_layout = QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(14, 12, 14, 12)
        advanced_hint = QLabel("필요한 단계만 골라 실행할 수 있어요. 이전에 끝낸 단계는 다시 하지 않습니다.")
        advanced_hint.setObjectName("HelperText")
        advanced_layout.addWidget(advanced_hint)
        stage_grid = QGridLayout()
        stage_grid.setHorizontalSpacing(8)
        stage_grid.setVerticalSpacing(8)
        self.stage_buttons: list[QPushButton] = []
        for index, (label, stage) in enumerate([
            ("1. 영상 가져오기", "download"),
            ("2. 쉬는 시간 정리", "edit"),
            ("3. 강의 글 만들기", "transcribe"),
            ("4. 노트 정리하기", "notes"),
        ]):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked, value=stage: self._start_pipeline(value))
            stage_grid.addWidget(button, index // 2, index % 2)
            self.stage_buttons.append(button)
        advanced_layout.addLayout(stage_grid)
        self.advanced_panel.setVisible(False)
        options.addWidget(self.advanced_panel)
        self.advanced_mode_check.toggled.connect(self.advanced_panel.setVisible)
        option_card.setProperty("cardRole", "settings")

        bento = QHBoxLayout()
        bento.setSpacing(16)
        bento.addWidget(source_card, 3)
        bento.addWidget(option_card, 2)
        layout.addLayout(bento)

        self.action_dock = QFrame()
        self.action_dock.setObjectName("ActionDock")
        dock_layout = QHBoxLayout(self.action_dock)
        dock_layout.setContentsMargins(18, 13, 14, 13)
        dock_layout.setSpacing(12)
        dock_copy = QVBoxLayout()
        dock_copy.setSpacing(1)
        self.dock_title = QLabel("준비되면 바로 시작하세요")
        self.dock_title.setObjectName("DockTitle")
        self.dock_helper = QLabel("중간에 멈춰도 완료한 단계부터 이어집니다.")
        self.dock_helper.setObjectName("DockHelper")
        dock_copy.addWidget(self.dock_title)
        dock_copy.addWidget(self.dock_helper)
        dock_layout.addLayout(dock_copy, 1)
        dock_layout.addWidget(self.cancel_button)
        dock_layout.addWidget(self.start_button)
        layout.addWidget(self.action_dock)
        self.advanced_mode_check.toggled.connect(self._set_advanced_mode_ui)

        self.progress_card, progress_layout = self._card(
            "작업 진행",
            "현재 단계와 다음에 할 일을 여기에서 확인할 수 있어요.",
        )
        self.progress_card.setProperty("cardRole", "progress")
        status_row = QHBoxLayout()
        self.stage_label = QLabel("대기 중")
        self.stage_label.setObjectName("SectionTitle")
        self.detail_label = QLabel("영상 링크를 넣으면 시작할 수 있어요.")
        self.detail_label.setObjectName("HelperText")
        status_row.addWidget(self.stage_label)
        status_row.addStretch(1)
        status_row.addWidget(self.detail_label)
        progress_layout.addLayout(status_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.log_toggle = QCheckBox("자세한 진행 내용 보기")
        progress_layout.addWidget(self.log_toggle)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1500)
        self.log_view.setMinimumHeight(120)
        self.log_view.setPlaceholderText("작업 과정의 자세한 내용이 여기에 표시됩니다.")
        self.log_view.setVisible(False)
        self.log_toggle.toggled.connect(self.log_view.setVisible)
        progress_layout.addWidget(self.log_view)
        self.progress_card.setVisible(False)
        layout.addWidget(self.progress_card)

        self.review_card, review_layout = self._card(
            "강의 글을 한 번 확인해주세요",
            "노트를 만들기 전에 반복된 문장이나 잘못 들은 전문용어가 없는지 가볍게 살펴보세요.",
        )
        review_status_row = QHBoxLayout()
        self.review_badge = QLabel("")
        self.review_headline = QLabel("")
        self.review_headline.setWordWrap(True)
        review_status_row.addWidget(self.review_badge)
        review_status_row.addWidget(self.review_headline, 1)
        review_layout.addLayout(review_status_row)
        self.review_metrics = QLabel("")
        self.review_metrics.setObjectName("HelperText")
        self.review_metrics.setWordWrap(True)
        review_layout.addWidget(self.review_metrics)
        self.review_warnings = QLabel("")
        self.review_warnings.setWordWrap(True)
        review_layout.addWidget(self.review_warnings)
        self.review_preview = QPlainTextEdit()
        self.review_preview.setReadOnly(True)
        self.review_preview.setMinimumHeight(260)
        self.review_preview.setAccessibleName("확인할 전체 강의 글")
        review_layout.addWidget(self.review_preview)
        review_actions = QHBoxLayout()
        self.review_open_button = QPushButton("전체 글 파일 열기")
        self.review_open_button.clicked.connect(self._open_review_transcript)
        self.review_reject_button = QPushButton("여기서 멈추기")
        self.review_reject_button.setObjectName("DangerButton")
        self.review_reject_button.clicked.connect(lambda: self._submit_transcript_review(False))
        self.review_approve_button = QPushButton("확인했어요 · 노트 만들기")
        self.review_approve_button.setObjectName("PrimaryButton")
        self.review_approve_button.clicked.connect(lambda: self._submit_transcript_review(True))
        review_actions.addWidget(self.review_open_button)
        review_actions.addStretch(1)
        review_actions.addWidget(self.review_reject_button)
        review_actions.addWidget(self.review_approve_button)
        review_layout.addLayout(review_actions)
        self.review_card.setVisible(False)
        layout.addWidget(self.review_card)

        self.result_card, result_layout = self._card(
            "노트가 준비됐어요",
            "바로 열어보거나 저장된 폴더에서 다른 파일과 함께 확인할 수 있어요.",
        )
        self.result_title = QLabel("")
        self.result_title.setObjectName("ResultTitle")
        self.result_title.setWordWrap(True)
        self.result_path = QLabel("")
        self.result_path.setObjectName("HelperText")
        self.result_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_layout.addWidget(self.result_title)
        result_layout.addWidget(self.result_path)
        result_buttons = QHBoxLayout()
        self.open_notes_button = QPushButton("강의 노트 열기")
        self.open_notes_button.clicked.connect(self._open_last_notes)
        self.open_transcript_button = QPushButton("강의 글 열기")
        self.open_transcript_button.clicked.connect(self._open_last_transcript)
        self.open_video_button = QPushButton("편집 영상 열기")
        self.open_video_button.clicked.connect(self._open_last_video)
        self.open_folder_button = QPushButton("저장 폴더 열기")
        self.open_folder_button.clicked.connect(self._open_last_folder)
        result_buttons.addWidget(self.open_notes_button)
        result_buttons.addWidget(self.open_transcript_button)
        result_buttons.addWidget(self.open_video_button)
        result_buttons.addWidget(self.open_folder_button)
        result_buttons.addStretch(1)
        result_layout.addLayout(result_buttons)
        self.result_card.setVisible(False)
        layout.addWidget(self.result_card)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return page

    def _build_prompt_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(14)
        model_card, model_layout = self._card(
            "노트를 만드는 ChatGPT",
            "ChatGPT Plus 로그인으로 사용합니다. 별도의 API 키나 추가 결제는 필요하지 않아요.",
        )
        model_grid = QGridLayout()
        self.codex_model_input = QLineEdit(self.settings.codex_model)
        self.codex_model_input.setAccessibleName("사용할 ChatGPT 모델")
        self.reasoning_combo = QComboBox()
        for label, value in [
            ("빠르게", "low"),
            ("보통", "medium"),
            ("꼼꼼하게", "high"),
            ("아주 꼼꼼하게", "xhigh"),
            ("가장 깊게", "max"),
        ]:
            self.reasoning_combo.addItem(label, value)
        self._select_data(self.reasoning_combo, self.settings.reasoning_effort)
        model_grid.addWidget(self._field_label("사용할 모델"), 0, 0)
        model_grid.addWidget(self._field_label("생각 깊이"), 0, 1)
        model_grid.addWidget(self.codex_model_input, 1, 0)
        model_grid.addWidget(self.reasoning_combo, 1, 1)
        model_grid.setColumnStretch(0, 2)
        model_grid.setColumnStretch(1, 1)
        model_layout.addLayout(model_grid)
        model_card.setProperty("cardRole", "model")

        profile = QFrame()
        profile.setObjectName("NoteProfile")
        profile_layout = QVBoxLayout(profile)
        profile_layout.setContentsMargins(18, 16, 18, 16)
        profile_layout.setSpacing(7)
        profile_kicker = QLabel("현재 노트 성격")
        profile_kicker.setObjectName("ProfileKicker")
        profile_title = QLabel("요약문이 아닌, 실제 수업 필기")
        profile_title.setObjectName("ProfileTitle")
        profile_text = QLabel("강의 흐름 유지  ·  개념 중심  ·  타임스탬프 없음")
        profile_text.setObjectName("ProfileText")
        profile_text.setWordWrap(True)
        profile_layout.addWidget(profile_kicker)
        profile_layout.addWidget(profile_title)
        profile_layout.addWidget(profile_text)
        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        top_row.addWidget(model_card, 3)
        top_row.addWidget(profile, 2)
        layout.addLayout(top_row)

        prompt_card, prompt_layout = self._card(
            "노트 작성 방식 · 프롬프트 슬롯",
            "강의용, 주식방송용처럼 원하는 개수만큼 저장해두고 작업마다 골라 쓸 수 있어요. {{transcript}}는 지우지 마세요.",
        )
        slot_row = QHBoxLayout()
        self.prompt_slot_combo = QComboBox()
        self.prompt_slot_combo.setAccessibleName("사용할 프롬프트 슬롯")
        self._refresh_prompt_slot_combo()
        self.prompt_slot_combo.currentIndexChanged.connect(self._change_prompt_slot)
        add_slot_button = QPushButton("새 슬롯")
        add_slot_button.clicked.connect(self._add_prompt_slot)
        rename_slot_button = QPushButton("이름 바꾸기")
        rename_slot_button.clicked.connect(self._rename_prompt_slot)
        delete_slot_button = QPushButton("삭제")
        delete_slot_button.setObjectName("DangerButton")
        delete_slot_button.clicked.connect(self._delete_prompt_slot)
        slot_row.addWidget(self._field_label("현재 프롬프트"))
        slot_row.addWidget(self.prompt_slot_combo, 1)
        slot_row.addWidget(add_slot_button)
        slot_row.addWidget(rename_slot_button)
        slot_row.addWidget(delete_slot_button)
        prompt_layout.addLayout(slot_row)
        self.prompt_editor = QPlainTextEdit()
        self.prompt_editor.setObjectName("PromptEditor")
        self.prompt_editor.setAccessibleName("강의 노트 작성 방식 편집기")
        self.prompt_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.prompt_editor.setPlainText(load_prompt(self.settings))
        prompt_layout.addWidget(self.prompt_editor, 1)
        prompt_actions = QHBoxLayout()
        reset_button = QPushButton("처음 설정으로 되돌리기")
        reset_button.clicked.connect(self._reset_prompt)
        save_button = QPushButton("이 방식 저장하기")
        save_button.setObjectName("PrimaryButton")
        save_button.clicked.connect(self._save_prompt_settings)
        prompt_actions.addWidget(reset_button)
        prompt_actions.addStretch(1)
        prompt_actions.addWidget(save_button)
        prompt_layout.addLayout(prompt_actions)
        prompt_card.setProperty("cardRole", "prompt")
        layout.addWidget(prompt_card, 1)
        return page

    def _build_tools_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 6)
        layout.setSpacing(16)

        gpu_card, gpu_layout = self._card(
            "내 컴퓨터 확인",
            ACCELERATOR_DESCRIPTION,
        )
        self.gpu_detail = QLabel("확인 중")
        self.gpu_detail.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_detail)
        gpu_card.setProperty("cardRole", "system")

        auth_card, auth_layout = self._card(
            "ChatGPT 연결",
            "로그인은 공식 연결 창에서 진행합니다. 이 앱은 비밀번호를 보거나 저장하지 않아요.",
        )
        self.auth_detail = QLabel("확인 중")
        self.auth_detail.setWordWrap(True)
        self.auth_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        auth_layout.addWidget(self.auth_detail)
        auth_actions = QHBoxLayout()
        self.login_button = QPushButton("ChatGPT 연결하기")
        self.login_button.setObjectName("PrimaryButton")
        self.login_button.clicked.connect(lambda: self._run_auth_action("login"))
        self.logout_button = QPushButton("로그아웃")
        self.logout_button.clicked.connect(lambda: self._run_auth_action("logout"))
        refresh_auth = QPushButton("연결 상태 다시 확인")
        refresh_auth.clicked.connect(lambda: self._run_auth_action("status"))
        auth_actions.addWidget(self.login_button)
        auth_actions.addWidget(self.logout_button)
        auth_actions.addWidget(refresh_auth)
        auth_actions.addStretch(1)
        auth_layout.addLayout(auth_actions)
        auth_card.setProperty("cardRole", "connection")
        system_row = QHBoxLayout()
        system_row.setSpacing(16)
        system_row.addWidget(gpu_card, 2)
        system_row.addWidget(auth_card, 3)
        layout.addLayout(system_row)

        update_card, update_layout = self._card(
            "영상 가져오기 도구 업데이트",
            "YouTube 변경 때문에 영상을 가져오지 못할 때 가장 먼저 업데이트해보세요.",
        )
        update_row = QHBoxLayout()
        self.ytdlp_version_label = QLabel(f"현재 버전: {current_version()}")
        self.update_channel_combo = QComboBox()
        self.update_channel_combo.addItem("새 기능 빠르게 받기 · 권장", "nightly")
        self.update_channel_combo.addItem("검증된 버전만 받기", "stable")
        self._select_data(self.update_channel_combo, self.settings.yt_dlp_channel)
        self.update_button = QPushButton("한 번에 확인하고 업데이트")
        self.update_button.setObjectName("PrimaryButton")
        self.update_button.clicked.connect(self._run_ytdlp_update)
        update_row.addWidget(self.ytdlp_version_label)
        update_row.addStretch(1)
        update_row.addWidget(self.update_channel_combo)
        update_row.addWidget(self.update_button)
        update_layout.addLayout(update_row)
        update_card.setProperty("cardRole", "update")
        layout.addWidget(update_card)

        log_card, log_layout = self._card("자세한 기록", "문제가 생겼을 때 원인을 확인할 수 있는 기록입니다.")
        self.tool_log = QPlainTextEdit()
        self.tool_log.setReadOnly(True)
        self.tool_log.setMaximumBlockCount(1000)
        self.tool_log.setMinimumHeight(150)
        log_layout.addWidget(self.tool_log)
        log_card.setProperty("cardRole", "log")
        layout.addWidget(log_card)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return page

    def _card(self, title: str, helper: str | None = None) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        if helper:
            description = QLabel(helper)
            description.setObjectName("HelperText")
            description.setWordWrap(True)
            layout.addWidget(description)
        return card, layout

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("FieldLabel")
        return label

    @staticmethod
    def _apply_shadow(
        widget: QWidget, *, blur: int = 22, y_offset: int = 5, alpha: int = 20
    ) -> None:
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, y_offset)
        shadow.setColor(QColor(30, 35, 55, alpha))
        widget.setGraphicsEffect(shadow)

    def _refresh_local_status(self) -> None:
        gpu = get_gpu_info()
        if gpu.available:
            memory_gb = gpu.memory_mb / 1024
            self._set_badge(self.gpu_badge, f"컴퓨터 준비됨 · {memory_gb:.0f}GB", "good")
            if IS_WINDOWS:
                detail = (
                    f"{gpu.name} · VRAM {memory_gb:.1f}GB · 드라이버 {gpu.driver}\n"
                    "CUDA 음성 인식을 사용할 수 있습니다."
                )
            else:
                detail = (
                    f"{gpu.name} · 통합 메모리 {memory_gb:.1f}GB · macOS {gpu.driver}\n"
                    "MLX Metal 음성 인식을 사용할 수 있습니다."
                )
            self.gpu_detail.setText(detail)
        else:
            self._set_badge(self.gpu_badge, "컴퓨터 확인 필요", "warn")
            self.gpu_detail.setText(f"GPU 가속 환경을 확인하지 못했습니다. {gpu.detail}")
        codex = find_codex()
        codex_text = (
            str(codex)
            if codex
            else f"연결 도구를 찾지 못했습니다 · {INSTALLER_NAME}을 먼저 실행해주세요."
        )
        self.auth_detail.setText(f"연결 도구: {codex_text}\nChatGPT 로그인 상태를 확인하고 있습니다.")

    def _set_badge(self, label: QLabel, text: str, state: str) -> None:
        object_name = {"good": "StatusGood", "warn": "StatusWarn"}.get(
            state, "StatusNeutral"
        )
        label.setObjectName(object_name)
        label.setText(text)
        label.style().unpolish(label)
        label.style().polish(label)

    def _choose_output_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "결과 저장 폴더 선택", self.folder_input.text()
        )
        if chosen:
            self.folder_input.setText(chosen)

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

    def _add_prompt_slot(self) -> None:
        self._save_prompt_draft()
        name, accepted = QInputDialog.getText(self, "새 프롬프트 슬롯", "슬롯 이름")
        name = " ".join(name.split()).strip()
        if not accepted or not name:
            return
        self.settings.add_prompt_slot(name, DEFAULT_PROMPT_PATH.read_text(encoding="utf-8"))
        self._refresh_prompt_slot_combo()
        self.prompt_editor.setPlainText(load_prompt(self.settings))

    def _rename_prompt_slot(self) -> None:
        slot = self.settings.active_prompt_slot()
        name, accepted = QInputDialog.getText(
            self, "프롬프트 슬롯 이름", "슬롯 이름", text=slot["name"]
        )
        name = " ".join(name.split()).strip()
        if accepted and name:
            slot["name"] = name
            self._refresh_prompt_slot_combo()

    def _delete_prompt_slot(self) -> None:
        if len(self.settings.prompt_slots) == 1:
            self._show_error("프롬프트 슬롯은 하나 이상 남겨야 합니다.")
            return
        slot = self.settings.active_prompt_slot()
        answer = QMessageBox.question(
            self, "프롬프트 슬롯 삭제", f"‘{slot['name']}’ 슬롯을 삭제할까요?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.settings.remove_active_prompt_slot()
        self._refresh_prompt_slot_combo()
        self.prompt_editor.setPlainText(load_prompt(self.settings))

    def _choose_media_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "전사할 영상 또는 음성 파일 선택",
            "",
            "미디어 파일 (*.mp4 *.m4v *.mov *.mkv *.webm *.avi *.mp3 *.m4a *.wav *.flac *.aac *.ogg *.opus);;모든 파일 (*)",
        )
        if chosen:
            self.url_input.setText(chosen)

    def _collect_settings(self) -> AppSettings:
        output_text = self.folder_input.text().strip()
        if not output_text:
            raise ValueError("결과 저장 폴더를 선택해주세요.")
        output_dir = Path(output_text).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self.codex_model_input.text().strip()
        if not model:
            raise ValueError("사용할 ChatGPT 모델을 입력해주세요.")
        self._save_prompt_draft()
        template = self.prompt_editor.toPlainText()
        validate_prompt(template)
        self.settings.output_dir = str(output_dir.resolve())
        self.settings.whisper_model = self.whisper_combo.currentText()
        self.settings.language = str(self.language_combo.currentData())
        self.settings.vad_filter = self.vad_check.isChecked()
        if self.use_time_range_check.isChecked():
            clip_start, clip_end = validate_clip_range(
                parse_timecode(self.clip_start_input.text()),
                parse_timecode(self.clip_end_input.text()),
            )
        else:
            clip_start, clip_end = 0.0, None
        self.settings.clip_start_seconds = clip_start
        self.settings.clip_end_seconds = clip_end
        self.settings.keep_audio = self.keep_audio_check.isChecked()
        self.settings.keep_video = self.keep_video_check.isChecked()
        self.settings.auto_terminology = self.auto_terminology_check.isChecked()
        self.settings.beam_size = self.beam_spin.value()
        self.settings.cookie_browser = str(self.cookie_combo.currentData())
        self.settings.codex_model = model
        self.settings.reasoning_effort = str(self.reasoning_combo.currentData())
        self.settings.yt_dlp_channel = str(self.update_channel_combo.currentData())
        save_prompt(self.settings, template)
        self.settings.save()
        return self.settings

    def _start_pipeline(self, target_stage: str = "all") -> None:
        source = self.url_input.text().strip()
        if not is_youtube_url(source):
            try:
                resolve_local_media(source)
            except ValueError as exc:
                self._show_error(f"YouTube 링크 또는 영상·음성 파일을 선택해주세요.\n{exc}")
                self.url_input.setFocus()
                return
        try:
            settings = self._collect_settings()
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            self.url_input.setFocus()
            return
        if settings.auto_terminology and target_stage in {"all", "transcribe"}:
            lecture_name, accepted = QInputDialog.getText(
                self,
                "전문용어 자동 입력",
                "전사 정확도를 높일 강의명을 입력하세요.",
                text=settings.terminology_lecture_name,
            )
            if not accepted:
                return
            lecture_name = " ".join(lecture_name.split()).strip()
            if not lecture_name:
                self._show_error("전문용어 자동 입력을 사용하려면 강의명을 입력해주세요.")
                return
            settings.terminology_lecture_name = lecture_name
        settings.last_url = source
        settings.save()
        self.log_view.clear()
        self.result_card.setVisible(False)
        self.review_card.setVisible(False)
        self.progress_card.setVisible(True)
        self.current_review = None
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        for button in self.stage_buttons:
            button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.stage_label.setText("준비 중")
        self.detail_label.setText("필요한 도구와 저장된 작업을 확인하고 있어요.")
        worker = PipelineWorker(source, settings, target_stage=target_stage)
        self.pipeline_worker = worker
        worker.progress.connect(self._on_pipeline_progress)
        worker.log.connect(self._append_pipeline_log)
        worker.result.connect(self._on_pipeline_result)
        worker.audio_edit_ready.connect(self._on_audio_edit_ready)
        worker.review_ready.connect(self._on_transcript_review_ready)
        worker.cancelled.connect(self._on_pipeline_cancelled)
        worker.error.connect(self._on_pipeline_error)
        worker.finished.connect(self._on_pipeline_finished)
        self._launch_worker(worker)

    def _cancel_pipeline(self) -> None:
        if self.pipeline_worker:
            self.pipeline_worker.cancel()
            self.cancel_button.setEnabled(False)
            self.detail_label.setText("현재 작업을 안전하게 멈추고 있어요.")
            self._append_pipeline_log("작업 중지를 요청했습니다.")

    def _on_pipeline_progress(self, percent: int, stage: str, detail: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(percent)
        friendly_stage, friendly_detail = self._friendly_progress(stage, detail)
        self.stage_label.setText(friendly_stage)
        self.detail_label.setText(f"{percent}% · {friendly_detail}")

    @staticmethod
    def _friendly_progress(stage: str, detail: str) -> tuple[str, str]:
        combined = f"{stage} {detail}".lower()
        if "전문용어" in combined or "terminology" in combined:
            return "전문용어 준비 중", "강의명에 맞는 단어를 GPT로 만들고 있어요."
        if any(word in combined for word in ("download", "다운로드", "원본")):
            return "영상 가져오는 중", "영상의 소리를 안전하게 가져오고 있어요."
        if any(word in combined for word in ("waveform", "파형", "edit", "편집")):
            return "쉬는 시간 정리", "말이 없는 구간과 음악 구간을 정리하고 있어요."
        if any(word in combined for word in ("transcrib", "whisper", "전사")):
            return "강의 글 만드는 중", "강의를 문장으로 옮기고 있어요."
        if any(word in combined for word in ("review", "검토", "확인")):
            return "강의 글 확인", "노트를 만들기 전에 내용을 점검하고 있어요."
        if any(word in combined for word in ("note", "gpt", "ai", "노트")):
            return "노트 정리 중", "확인한 강의 글을 공부하기 좋은 노트로 정리하고 있어요."
        if any(word in combined for word in ("완료", "complete")):
            return "모두 완료", "노트가 준비됐어요."
        return "준비 중", "작업에 필요한 내용을 확인하고 있어요."

    def _append_pipeline_log(self, message: str) -> None:
        if message.strip():
            self.log_view.appendPlainText(message.strip())
            scrollbar = self.log_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _on_pipeline_result(self, result: PipelineResult) -> None:
        self.last_result = result
        stage_labels = {
            "download": "영상을 가져왔어요. 다음에는 쉬는 시간 정리부터 이어갈 수 있습니다.",
            "edit": "정리한 오디오와 선택한 영상 파일을 저장했어요. 다음에는 강의 글 만들기부터 이어갈 수 있습니다.",
            "transcribe": "강의 글을 저장했어요. 다음에는 노트 정리부터 이어갈 수 있습니다.",
            "notes": "강의 노트가 모두 완성됐습니다.",
        }
        self.result_title.setText(
            f"{result.video_title}\n{stage_labels.get(result.completed_stage, '작업 단계가 완료되었습니다.')}"
        )
        self.result_path.setText(str(result.output_dir))
        self.open_notes_button.setEnabled(result.notes_path is not None)
        self.open_transcript_button.setEnabled(result.transcript_path is not None)
        self.open_video_button.setEnabled(result.video_path is not None)
        self.result_card.setVisible(True)
        self.stage_label.setText("모두 완료")
        self.detail_label.setText(f"100% · {stage_labels.get(result.completed_stage, '단계 저장 완료')}")
        self.progress_bar.setValue(100)

    def _on_audio_edit_ready(self, request: object) -> None:
        if not self.pipeline_worker:
            return
        self.stage_label.setText("쉬는 시간 정리")
        self.detail_label.setText("28% · 소리를 들으며 수업이 아닌 구간을 빼주세요.")
        self._append_pipeline_log("오디오 파형이 준비됐습니다. 편집 창에서 쉬는 시간과 음악 구간을 확인해주세요.")
        dialog = AudioEditorDialog(request, self)
        self.audio_editor = dialog
        self._notify_action_required(
            "쉬는 시간 확인이 필요합니다",
            "강의 파형이 준비됐습니다. 제외할 쉬는 시간과 음악 구간을 확인해주세요.",
        )
        result = dialog.exec()
        worker = self.pipeline_worker
        exclusions = dialog.exclusions if result == QDialog.DialogCode.Accepted else None
        dialog.release_media()
        dialog.deleteLater()
        self.audio_editor = None
        if worker:
            worker.submit_audio_edit(exclusions)
            if exclusions is not None:
                self._append_pipeline_log(
                    f"오디오 정리 적용: 수업이 아닌 구간 {len(exclusions)}개를 강의 글에서 뺍니다."
                )

    def _on_transcript_review_ready(self, review: TranscriptReview) -> None:
        self.current_review = review
        self.review_preview.setPlainText(review.preview_text)
        self.review_preview.moveCursor(QTextCursor.MoveOperation.Start)
        self.review_metrics.setText(review.metrics)
        if review.warnings:
            self.review_warnings.setText("\n".join(f"• {item}" for item in review.warnings))
            self._set_badge(self.review_badge, "확인 필요", "warn")
        else:
            self.review_warnings.setText("자동 확인은 통과했어요. 앞부분과 끝부분, 전문용어만 한 번 살펴보세요.")
            self._set_badge(self.review_badge, "내용이 자연스러워요", "good")
        self.review_headline.setText(review.headline)
        self.review_approve_button.setEnabled(True)
        self.review_reject_button.setEnabled(True)
        self.review_card.setVisible(True)
        self.stage_label.setText("강의 글 확인")
        target = self.pipeline_worker.target_stage if self.pipeline_worker else "all"
        if target == "transcribe":
            self.review_approve_button.setText("확인했어요 · 강의 글 저장")
            self.detail_label.setText("81% · 확인하면 강의 글을 저장합니다.")
            self._append_pipeline_log(
                "강의 글 만들기가 끝났습니다. 내용을 확인한 뒤 저장해주세요."
            )
        else:
            self.review_approve_button.setText("확인했어요 · 노트 만들기")
            self.detail_label.setText("81% · 확인하기 전에는 ChatGPT가 시작되지 않습니다.")
            self._append_pipeline_log(
                "강의 글 만들기가 끝났습니다. 화면에서 내용을 확인한 뒤 노트를 만들어주세요."
            )
        self._notify_action_required(
            "강의 글 확인이 필요합니다",
            "전사가 끝났습니다. 내용을 확인해야 다음 단계로 진행할 수 있습니다.",
        )

    def _submit_transcript_review(self, approved: bool) -> None:
        if not self.pipeline_worker or not self.current_review:
            return
        self.review_approve_button.setEnabled(False)
        self.review_reject_button.setEnabled(False)
        self.pipeline_worker.submit_review(approved)
        if approved:
            if self.pipeline_worker.target_stage == "transcribe":
                self.stage_label.setText("강의 글 저장 중")
                self.detail_label.setText("확인한 강의 글을 저장하고 있어요.")
                self._append_pipeline_log("강의 글 확인 완료: 이 단계의 결과를 저장합니다.")
            else:
                self.stage_label.setText("노트 정리 중")
                self.detail_label.setText("84% · 확인한 강의 글로 노트를 만들고 있어요.")
                self._append_pipeline_log("강의 글 확인 완료: 노트 정리를 시작합니다.")
        else:
            self._append_pipeline_log("강의 글 확인 단계에서 작업을 멈췄습니다.")

    def _open_review_transcript(self) -> None:
        if self.current_review:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_review.text_path)))

    def _on_pipeline_cancelled(self, message: str) -> None:
        self.stage_label.setText("중단됨")
        self.detail_label.setText("끝난 단계는 저장됐어요. 같은 링크로 다시 시작하면 이어집니다.")
        self._append_pipeline_log(message)

    def _on_pipeline_error(self, message: str) -> None:
        self.stage_label.setText("중단됨")
        self.detail_label.setText("자세한 진행 내용과 안내 메시지를 확인해주세요.")
        self._append_pipeline_log(f"오류: {message}")
        self._show_error(message)

    def _on_pipeline_finished(self) -> None:
        self.start_button.setEnabled(True)
        for button in self.stage_buttons:
            button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.pipeline_worker = None

    def _save_prompt_settings(self) -> None:
        try:
            self._collect_settings()
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        QMessageBox.information(self, "저장 완료", "노트 작성 방식과 앱 설정을 저장했습니다.")

    def _reset_prompt(self) -> None:
        answer = QMessageBox.question(
            self,
            "처음 설정으로 되돌리기",
            "현재 내용을 처음 제공된 노트 작성 방식으로 바꿀까요? 저장 버튼을 누르기 전까지는 실제 설정에 반영되지 않습니다.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.prompt_editor.setPlainText(DEFAULT_PROMPT_PATH.read_text(encoding="utf-8"))

    def _run_auth_action(self, action: str) -> None:
        self.login_button.setEnabled(False)
        self.logout_button.setEnabled(False)
        if action == "login":
            self.tool_log.appendPlainText("ChatGPT 연결을 시작합니다.")
        worker = AuthWorker(action)
        worker.log.connect(self._append_tool_log)
        worker.result.connect(self._on_auth_result)
        worker.error.connect(self._on_tool_error)
        worker.finished.connect(lambda: self._set_auth_buttons_enabled(True))
        self._launch_worker(worker)

    def _on_auth_result(self, status: AuthStatus) -> None:
        codex = find_codex()
        prefix = f"연결 도구: {codex or '찾지 못함'}"
        self.auth_detail.setText(f"{prefix}\n{status.detail}")
        if status.logged_in:
            self._set_badge(self.ai_badge, "ChatGPT 연결됨", "good")
        elif status.available:
            self._set_badge(self.ai_badge, "ChatGPT 로그인 필요", "warn")
        else:
            self._set_badge(self.ai_badge, "연결 도구 설치 필요", "warn")

    def _set_auth_buttons_enabled(self, enabled: bool) -> None:
        self.login_button.setEnabled(enabled)
        self.logout_button.setEnabled(enabled)

    def _run_ytdlp_update(self) -> None:
        channel = str(self.update_channel_combo.currentData())
        self.settings.yt_dlp_channel = channel
        self.settings.save()
        self.update_button.setEnabled(False)
        self.tool_log.appendPlainText(f"영상 가져오기 도구({channel})의 업데이트를 확인합니다.")
        worker = UpdateWorker(channel)
        worker.log.connect(self._append_tool_log)
        worker.result.connect(self._on_update_result)
        worker.error.connect(self._on_tool_error)
        worker.finished.connect(lambda: self.update_button.setEnabled(True))
        self._launch_worker(worker)

    def _on_update_result(self, result: UpdateResult) -> None:
        self.ytdlp_version_label.setText(f"현재 버전: {result.current}")
        if result.updated:
            message = f"영상 가져오기 도구를 {result.previous}에서 {result.current}(으)로 업데이트했습니다."
        else:
            message = f"이미 최신 버전입니다: {result.current}"
        self._append_tool_log(message)
        QMessageBox.information(self, "업데이트 완료", message)

    def _append_tool_log(self, message: str) -> None:
        if message.strip():
            self.tool_log.appendPlainText(message.strip())
            scrollbar = self.tool_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _on_tool_error(self, message: str) -> None:
        self._append_tool_log(f"오류: {message}")
        self._show_error(message)

    def _launch_worker(self, worker: object) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        pair = (thread, worker)
        self._background.append(pair)

        def cleanup() -> None:
            if pair in self._background:
                self._background.remove(pair)
            thread.deleteLater()

        thread.finished.connect(cleanup)
        thread.start()

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

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "작업을 완료하지 못했습니다", message)

    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.pipeline_worker:
            answer = QMessageBox.question(
                self,
                "작업 진행 중",
                "현재 작업을 취소할까요? 안전하게 중단된 뒤 창을 다시 닫아주세요.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._cancel_pipeline()
            event.ignore()
            return
        for thread, worker in list(self._background):
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()
            thread.quit()
            thread.wait(3000)
        if self._notification_tray:
            self._notification_tray.hide()
        event.accept()
