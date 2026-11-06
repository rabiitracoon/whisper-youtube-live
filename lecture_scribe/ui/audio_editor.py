from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, QRect, QRegularExpression, QRectF, QSize, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QCloseEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QRegularExpressionValidator,
    QWheelEvent,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..audio_edit import (
    AutoDetectionCancelled,
    AudioEditRequest,
    build_included_ranges,
    cuts_to_segments,
    detect_long_non_speech,
    normalize_exclusions,
)
from ..time_range import parse_timecode
from ..youtube import format_duration
from .editor_style import EDITOR_STYLE
from .theme import COLORS


def _precise_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class TimelineScrollArea(QScrollArea):
    """Premiere-style wheel navigation for the waveform timeline."""

    zoom_requested = Signal(int, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom_wheel_remainder = 0
        self.setAccessibleName("오디오 타임라인 탐색 영역")
        self.setAccessibleDescription(
            "마우스 휠은 커서 위치를 중심으로 확대하거나 축소하고, Shift와 마우스 휠은 좌우로 이동합니다."
        )
        self.setToolTip("휠: 확대/축소 · Shift + 휠: 좌우 탐색")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        widget = self.widget()
        if widget is not None:
            # The waveform is as wide as the zoom asks, but always as tall as the view.
            widget.setFixedHeight(max(widget.minimumHeight(), self.viewport().height()))

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        angle = event.angleDelta()
        pixel = event.pixelDelta()
        horizontal_only = abs(angle.x()) > abs(angle.y()) or (
            not angle.y() and abs(pixel.x()) > abs(pixel.y())
        )
        angle_delta = angle.x() if horizontal_only else angle.y()
        pixel_delta = pixel.x() if horizontal_only else pixel.y()
        if not angle_delta and not pixel_delta:
            event.ignore()
            return

        if horizontal_only or event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # A mouse-wheel notch moves roughly one comfortable viewport slice.
            # Precision touchpads keep their native pixel distance. Horizontal
            # trackpad swipes pan instead of zooming.
            distance = pixel_delta if pixel_delta else round(angle_delta * 1.5)
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - distance)
            event.accept()
            return

        # Most mouse wheels report 120 units per notch. Accumulation also makes
        # high-resolution wheels and touchpads predictable instead of too fast.
        self._zoom_wheel_remainder += angle_delta or pixel_delta * 8
        steps = int(self._zoom_wheel_remainder / 120)
        if steps:
            self._zoom_wheel_remainder -= steps * 120
            cursor_x = max(0.0, min(event.position().x(), float(self.viewport().width())))
            self.zoom_requested.emit(steps, cursor_x)
        event.accept()


class AutoDetectionWorker(QObject):
    progress = Signal(int, str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, request: AudioEditRequest, cancel_event: Event) -> None:
        super().__init__()
        self.request = request
        self.cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        try:
            candidates = detect_long_non_speech(
                self.request.audio_path,
                self.request.range_start,
                self.request.range_end,
                cancel_event=self.cancel_event,
                progress=lambda ratio, detail: self.progress.emit(
                    round(ratio * 100), detail
                ),
            )
            self.completed.emit(candidates)
        except AutoDetectionCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface worker errors in the dialog
            self.failed.emit(str(exc))




_TICK_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)


def _tick_label(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _rgba(key: str, alpha: int) -> QColor:
    color = QColor(COLORS[key])
    color.setAlpha(alpha)
    return color


class WaveformWidget(QWidget):
    seek_requested = Signal(float)
    razor_requested = Signal(float)
    segment_requested = Signal(int)
    toggle_playback_requested = Signal()

    def __init__(self, request: AudioEditRequest) -> None:
        super().__init__()
        self.duration = request.duration
        self.peaks = request.peaks
        self.range_start = request.range_start
        self.range_end = request.range_end
        self.tool = "razor"
        self.cuts: tuple[float, ...] = ()
        self.exclusions: tuple[tuple[float, float], ...] = ()
        self.selected_segment: int | None = None
        self.playhead = request.range_start
        self.setMinimumHeight(220)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setAccessibleName("강의 오디오 파형 편집 영역")
        self.setAccessibleDescription(
            "자르기 도구에서는 클릭한 위치가 잘립니다. 구간 선택 도구에서는 잘린 구간을 선택하고, 위치 이동 도구에서는 재생 위치를 옮깁니다."
        )

    @property
    def segments(self) -> tuple[tuple[float, float], ...]:
        return cuts_to_segments(self.cuts, self.range_start, self.range_end)

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        cursor = Qt.CursorShape.PointingHandCursor
        if tool == "razor":
            cursor = Qt.CursorShape.CrossCursor
        elif tool == "playhead":
            cursor = Qt.CursorShape.SplitHCursor
        self.setCursor(cursor)

    def set_edit_state(
        self,
        cuts: tuple[float, ...],
        exclusions: tuple[tuple[float, float], ...],
        selected_segment: int | None,
    ) -> None:
        self.cuts = cuts
        self.exclusions = exclusions
        self.selected_segment = selected_segment
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self.playhead = max(0.0, min(seconds, self.duration))
        self.update()

    def set_zoom_width(self, width: int) -> None:
        self.setFixedWidth(max(900, width))
        self.updateGeometry()

    def _tick_interval(self) -> float:
        if self.duration <= 0 or self.width() <= 0:
            return _TICK_STEPS[-1]
        pixels_per_second = self.width() / self.duration
        for step in _TICK_STEPS:
            if step * pixels_per_second >= 110:
                return step
        return _TICK_STEPS[-1]

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor(COLORS["editor_bg"]))
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0:
            return
        ruler = 26
        wave_height = height - ruler
        center = wave_height / 2
        visible = event.rect().adjusted(-2, 0, 2, 0)
        left_x = max(0, visible.left())
        right_x = min(width, visible.right() + 1)

        painter.setPen(QPen(QColor(COLORS["editor_border"]), 1))
        painter.drawLine(left_x, int(center), right_x, int(center))

        if self.peaks and self.duration > 0:
            painter.setPen(QPen(QColor(COLORS["wave_kept"]), 1))
            count = len(self.peaks)
            for x in range(left_x, right_x):
                index = min(count - 1, int(x / max(1, width - 1) * count))
                amplitude = max(1.0, self.peaks[index] * (wave_height * 0.42))
                painter.drawLine(x, int(center - amplitude), x, int(center + amplitude))

        if self.duration > 0:
            dim = _rgba("editor_bg", 190)
            self._shade(painter, 0.0, self.range_start, dim)
            self._shade(painter, self.range_end, self.duration, dim)
            for start, end in self.exclusions:
                self._shade(painter, start, end, _rgba("wave_removed", 70), wave_height)
            for start, end in self.exclusions:
                left, right = self._x_for_seconds(start), self._x_for_seconds(end)
                painter.fillRect(
                    QRectF(left, 0, max(1, right - left), 3), _rgba("wave_removed", 220)
                )

            segments = self.segments
            if self.selected_segment is not None and self.selected_segment < len(segments):
                start, end = segments[self.selected_segment]
                left, right = self._x_for_seconds(start), self._x_for_seconds(end)
                painter.fillRect(
                    QRectF(left, 0, max(1, right - left), wave_height),
                    _rgba("wave_selection", 38),
                )
                painter.setPen(QPen(QColor(COLORS["wave_selection"]), 2))
                painter.drawRect(
                    QRectF(left + 1, 1, max(2, right - left - 2), wave_height - 3)
                )

            painter.setPen(QPen(QColor(COLORS["wave_cut"]), 2))
            for cut in self.cuts:
                x = self._x_for_seconds(cut)
                if left_x - 2 <= x <= right_x + 2:
                    painter.drawLine(x, 0, x, height)

            self._draw_ruler(painter, left_x, right_x, height, ruler)

            playhead_x = self._x_for_seconds(self.playhead)
            painter.setPen(QPen(QColor(COLORS["wave_playhead"]), 2))
            painter.drawLine(playhead_x, 0, playhead_x, height)

    def _draw_ruler(
        self, painter: QPainter, left_x: int, right_x: int, height: int, ruler: int
    ) -> None:
        painter.setPen(QPen(QColor(COLORS["editor_border"]), 1))
        painter.drawLine(left_x, height - ruler, right_x, height - ruler)
        interval = self._tick_interval()
        pixels_per_second = self.width() / self.duration
        first = max(0, int((left_x - 90) / pixels_per_second / interval))
        last = int((right_x + 2) / pixels_per_second / interval) + 1
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        for index in range(first, last + 1):
            seconds = index * interval
            if seconds > self.duration:
                break
            x = self._x_for_seconds(seconds)
            painter.setPen(QPen(QColor(COLORS["editor_text_tertiary"]), 1))
            painter.drawLine(x, height - ruler, x, height - ruler + 6)
            painter.setPen(QPen(QColor(COLORS["editor_text_secondary"]), 1))
            painter.drawText(
                QRectF(x + 4, height - ruler + 5, 90, ruler - 6),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                _tick_label(seconds),
            )

    def _shade(
        self,
        painter: QPainter,
        start: float,
        end: float,
        color: QColor,
        height: float | None = None,
    ) -> None:
        if end <= start:
            return
        left = self._x_for_seconds(start)
        right = self._x_for_seconds(end)
        painter.fillRect(
            QRectF(left, 0, max(1, right - left), height or self.height()), color
        )

    def _seconds_for_x(self, x: float) -> float:
        if self.width() <= 0:
            return self.range_start
        return max(
            self.range_start,
            min(self.range_end, x / max(1, self.width()) * self.duration),
        )

    def _x_for_seconds(self, seconds: float) -> int:
        if self.duration <= 0:
            return 0
        return round(
            max(0.0, min(seconds, self.duration)) / self.duration * self.width()
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        seconds = self._seconds_for_x(event.position().x())
        self.setFocus()
        if self.tool == "razor":
            self.razor_requested.emit(seconds)
        elif self.tool == "select":
            for index, (start, end) in enumerate(self.segments):
                if start <= seconds <= end:
                    self.segment_requested.emit(index)
                    break
        else:
            self.seek_requested.emit(seconds)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Right}:
            step = 30 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 5
            direction = -1 if event.key() == Qt.Key.Key_Left else 1
            self.seek_requested.emit(self.playhead + step * direction)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            self.toggle_playback_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def _card(tone: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("EditorCard")
    if tone:
        frame.setProperty("tone", tone)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 14)
    layout.setSpacing(8)
    return frame, layout


def _label(text: str, name: str | None = None, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


class AudioEditorDialog(QDialog):
    NO_SELECTION_TEXT = "파형에서 두 곳을 잘라 구간을 만들고, 구간 선택 도구로 눌러보세요."

    def __init__(self, request: AudioEditRequest, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.request = request
        self.cuts: tuple[float, ...] = ()
        self.exclusions: tuple[tuple[float, float], ...] = ()
        self.selected_segment: int | None = None
        self._cut_history: list[float] = []
        self._skip_discard_confirm = False
        self._preview_stop_ms: int | None = None
        self._media_ready = False
        self._pending_seek_seconds = request.range_start
        self._media_released = False
        self._auto_thread: QThread | None = None
        self._auto_worker: AutoDetectionWorker | None = None
        self._auto_cancel_event: Event | None = None
        self._auto_previous_state: tuple[
            tuple[float, ...],
            tuple[tuple[float, float], ...],
            int | None,
            tuple[tuple[float, float], ...],
        ] | None = None
        self._auto_candidates: tuple[tuple[float, float], ...] = ()
        self._close_after_detection = False
        self.setObjectName("EditorDialog")
        self.setStyleSheet(EDITOR_STYLE)
        self.setWindowTitle("Lecture Scribe — 쉬는 시간 정리")
        screen = self.screen() or QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen else QRect(0, 0, 1366, 728)
        self.setMinimumSize(min(960, available.width()), min(620, available.height()))
        self.resize(
            min(1380, int(available.width() * 0.92)),
            min(860, int(available.height() * 0.92)),
        )
        self.setModal(True)

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.errorOccurred.connect(self._on_player_error)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(12)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(_label("2단계 · 수업이 아닌 구간 빼기", "EditorKicker"))
        header.addWidget(_label("쉬는 시간과 음악을 정리해주세요", "EditorTitle"))
        header.addWidget(
            _label("자동으로 찾은 뒤 소리를 들어 확인하고, 필요하면 파형을 클릭해 직접 자르세요.", "HelperText")
        )
        root.addLayout(header)

        work_area = QHBoxLayout()
        work_area.setSpacing(14)
        timeline_column = QVBoxLayout()
        timeline_column.setSpacing(10)

        toolbar = QFrame()
        toolbar.setObjectName("Toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(6)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for label, tool, shortcut in [
            ("구간 선택", "select", "V"),
            ("자르기", "razor", "C"),
            ("위치 이동", "playhead", "P"),
        ]:
            button = QPushButton(f"{label}  {shortcut}")
            button.setCheckable(True)
            button.setProperty("editorTool", True)
            button.clicked.connect(lambda _checked, value=tool: self._set_tool(value))
            self.tool_group.addButton(button)
            toolbar_layout.addWidget(button)
            if tool == "razor":
                button.setChecked(True)
        toolbar_layout.addSpacing(10)
        self.time_input = QLineEdit(_precise_time(request.range_start))
        self.time_input.setAccessibleName("정확한 현재 위치 시간")
        self.time_input.setPlaceholderText("HH:MM:SS.mmm")
        pattern = QRegularExpression(r"\d{1,3}:[0-5]\d:[0-5]\d(?:\.\d{1,3})?")
        self.time_input.setValidator(QRegularExpressionValidator(pattern, self.time_input))
        self.time_input.setFixedWidth(128)
        self.time_input.returnPressed.connect(self._seek_from_input)
        move_button = QPushButton("찾기")
        move_button.setObjectName("GhostButton")
        move_button.clicked.connect(self._seek_from_input)
        cut_at_head = QPushButton("여기서 자르기")
        cut_at_head.clicked.connect(lambda: self._add_cut(self.waveform.playhead))
        toolbar_layout.addWidget(self.time_input)
        toolbar_layout.addWidget(move_button)
        toolbar_layout.addWidget(cut_at_head)
        toolbar_layout.addStretch(1)
        zoom_caption = _label("확대", "MutedLabel")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 24)
        self.zoom_slider.setValue(5)
        self.zoom_slider.setFixedWidth(130)
        self.zoom_slider.setAccessibleName("파형 확대 배율")
        self.zoom_slider.setToolTip("휠: 확대/축소 · Shift + 휠: 좌우 이동")
        self.zoom_label = _label("5×", "MutedLabel")
        self.zoom_label.setMinimumWidth(26)
        toolbar_layout.addWidget(zoom_caption)
        toolbar_layout.addWidget(self.zoom_slider)
        toolbar_layout.addWidget(self.zoom_label)
        timeline_column.addWidget(toolbar)

        self.waveform = WaveformWidget(request)
        self.waveform.set_zoom_width(1000 * self.zoom_slider.value())
        self.waveform.seek_requested.connect(self._seek_seconds)
        self.waveform.razor_requested.connect(self._add_cut)
        self.waveform.segment_requested.connect(self._select_segment)
        self.waveform.toggle_playback_requested.connect(self._toggle_playback)
        self.zoom_slider.valueChanged.connect(self._set_zoom)
        self.waveform_scroll = TimelineScrollArea()
        self.waveform_scroll.setObjectName("TimelineScroll")
        self.waveform_scroll.setWidget(self.waveform)
        self.waveform_scroll.setWidgetResizable(False)
        self.waveform_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.waveform_scroll.setMinimumHeight(250)
        self.waveform_scroll.zoom_requested.connect(self._zoom_from_wheel)
        timeline_column.addWidget(self.waveform_scroll, 1)

        legend = QHBoxLayout()
        legend.setSpacing(6)
        for key, text in [
            ("wave_cut", "자른 곳"),
            ("wave_removed", "글에서 뺄 구간"),
            ("wave_selection", "선택한 구간"),
            ("wave_playhead", "현재 위치"),
        ]:
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background: {COLORS[key]}; border-radius: 5px;")
            legend.addWidget(dot)
            legend.addWidget(_label(text, "MutedLabel"))
            legend.addSpacing(10)
        legend.addStretch(1)
        legend.addWidget(_label("휠: 확대/축소 · Shift + 휠: 좌우 이동", "MutedLabel"))
        timeline_column.addLayout(legend)

        playback_row = QHBoxLayout()
        playback_row.setSpacing(8)
        self.play_button = QPushButton("재생")
        self.play_button.setObjectName("AccentButton")
        self.play_button.setMinimumWidth(92)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_clip_button = QPushButton("선택 구간 듣기")
        self.play_clip_button.setEnabled(False)
        self.play_clip_button.clicked.connect(self._play_selected_clip)
        stop_button = QPushButton("정지")
        stop_button.clicked.connect(self._stop)
        self.position_label = QLabel(
            f"{_precise_time(request.range_start)} / {format_duration(request.duration)}"
        )
        self.position_label.setObjectName("TimeLabel")
        self.audio_status_label = _label("소리 준비 중", "MutedLabel")
        self.audio_status_label.setMinimumWidth(110)
        volume = QSlider(Qt.Orientation.Horizontal)
        volume.setRange(0, 100)
        volume.setValue(80)
        volume.setFixedWidth(110)
        volume.setAccessibleName("미리듣기 음량")
        volume.valueChanged.connect(
            lambda value: self.audio_output.setVolume(value / 100)
        )
        playback_row.addWidget(self.play_button)
        playback_row.addWidget(self.play_clip_button)
        playback_row.addWidget(stop_button)
        playback_row.addSpacing(8)
        playback_row.addWidget(self.position_label)
        playback_row.addStretch(1)
        playback_row.addWidget(self.audio_status_label)
        playback_row.addSpacing(8)
        playback_row.addWidget(_label("음량", "MutedLabel"))
        playback_row.addWidget(volume)
        timeline_column.addLayout(playback_row)

        # Inspector column (scrolls when the window is short).
        inspector = QWidget()
        clip_layout = QVBoxLayout(inspector)
        clip_layout.setContentsMargins(0, 0, 6, 0)
        clip_layout.setSpacing(10)

        auto_card, auto_layout = _card()
        auto_layout.addWidget(_label("자동으로 찾기", "CardKicker"))
        auto_layout.addWidget(_label("쉬는 시간과 음악을 먼저 찾아볼까요?", "CardTitle", True))
        self.auto_detect_button = QPushButton("쉬는 시간 자동 찾기")
        self.auto_detect_button.setObjectName("AccentButton")
        self.auto_detect_button.setAccessibleName("음악과 쉬는 시간 자동 찾기")
        self.auto_detect_button.setToolTip(
            "수업 전과 종료 후의 음악, 중간의 긴 쉬는 시간을 찾아 제외할 구간으로 표시합니다."
        )
        self.auto_detect_button.clicked.connect(self._toggle_auto_detection)
        self.undo_auto_button = QPushButton("되돌리기")
        self.undo_auto_button.setObjectName("GhostButton")
        self.undo_auto_button.setToolTip("자동 찾기 전 상태로 되돌립니다.")
        self.undo_auto_button.setEnabled(False)
        self.undo_auto_button.clicked.connect(self._undo_auto_detection)
        self.auto_status_label = _label(
            "표시된 구간은 꼭 한 번 들어보고 확인하세요.", "HelperText", True
        )
        self.auto_progress = QProgressBar()
        self.auto_progress.setRange(0, 100)
        self.auto_progress.setAccessibleName("쉬는 시간 자동 찾기 진행률")
        self.auto_progress.setVisible(False)
        auto_buttons = QHBoxLayout()
        auto_buttons.setSpacing(6)
        auto_buttons.addWidget(self.auto_detect_button, 1)
        auto_buttons.addWidget(self.undo_auto_button)
        auto_layout.addLayout(auto_buttons)
        auto_layout.addWidget(self.auto_progress)
        auto_layout.addWidget(self.auto_status_label)
        clip_layout.addWidget(auto_card)

        sel_card, sel_layout = _card()
        sel_layout.addWidget(_label("선택한 구간", "CardKicker"))
        self.clip_label = _label(self.NO_SELECTION_TEXT, "HelperText", True)
        self.toggle_exclusion_button = QPushButton("선택 구간 빼기")
        self.toggle_exclusion_button.setEnabled(False)
        self.toggle_exclusion_button.clicked.connect(self._toggle_selected_exclusion)
        self.undo_cut_button = QPushButton("마지막 자르기 취소")
        self.undo_cut_button.setObjectName("GhostButton")
        self.undo_cut_button.clicked.connect(self._undo_cut)
        sel_layout.addWidget(self.clip_label)
        sel_actions = QHBoxLayout()
        sel_actions.setSpacing(6)
        sel_actions.addWidget(self.toggle_exclusion_button, 1)
        sel_actions.addWidget(self.undo_cut_button)
        sel_layout.addLayout(sel_actions)
        clip_layout.addWidget(sel_card)

        removed_card, removed_layout = _card()
        removed_layout.addWidget(_label("글에서 빠질 구간", "CardKicker"))
        self.exclusion_list = QListWidget()
        self.exclusion_list.setAccessibleName("강의 글에서 빠질 구간 목록")
        self.exclusion_list.setMinimumHeight(120)
        self.exclusion_list.itemDoubleClicked.connect(self._seek_to_exclusion)
        self.exclusion_empty = _label(
            "아직 뺄 구간이 없어요. 파형에서 구간을 고르거나 자동으로 찾아보세요.",
            "EmptyState",
            True,
        )
        self.exclusion_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        removed_layout.addWidget(self.exclusion_list, 1)
        removed_layout.addWidget(self.exclusion_empty, 1)
        clip_layout.addWidget(removed_card, 1)

        self.inspector_scroll = QScrollArea()
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inspector_scroll.setWidget(inspector)
        self.inspector_scroll.setFixedWidth(336)

        work_area.addLayout(timeline_column, 1)
        work_area.addWidget(self.inspector_scroll)
        root.addLayout(work_area, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.summary_label = _label("", "MutedLabel")
        cancel_button = QPushButton("나중에 이어하기")
        cancel_button.setObjectName("GhostButton")
        cancel_button.clicked.connect(self.reject)
        self.apply_button = QPushButton("이대로 오디오 정리하기")
        self.apply_button.setObjectName("PrimaryButton")
        self.apply_button.clicked.connect(self._apply)
        bottom.addWidget(self.summary_label)
        bottom.addStretch(1)
        bottom.addWidget(cancel_button)
        bottom.addWidget(self.apply_button)
        root.addLayout(bottom)

        self._refresh_edit_state()
        self._seek_seconds(request.range_start)
        self.player.setSource(QUrl.fromLocalFile(str(request.audio_path)))

    def release_media(self) -> None:
        """Release the media backend's handle to the downloaded source."""
        if self._media_released:
            return
        self._media_released = True
        self._preview_stop_ms = None
        self._media_ready = False
        self.player.stop()
        self.player.setSource(QUrl())
        self.player.setAudioOutput(None)

    def _set_tool(self, tool: str) -> None:
        self.waveform.set_tool(tool)

    def _toggle_auto_detection(self) -> None:
        if self._auto_thread and self._auto_thread.isRunning():
            self._cancel_auto_detection()
            return
        self._auto_previous_state = None
        self._auto_cancel_event = Event()
        thread = QThread(self)
        worker = AutoDetectionWorker(self.request, self._auto_cancel_event)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_auto_progress)
        worker.completed.connect(self._on_auto_detected)
        worker.cancelled.connect(self._on_auto_cancelled)
        worker.failed.connect(self._on_auto_failed)
        worker.completed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_auto_thread_finished)
        self._auto_thread = thread
        self._auto_worker = worker
        self.auto_detect_button.setText("자동 찾기 멈추기")
        self.undo_auto_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.auto_progress.setValue(0)
        self.auto_progress.setVisible(True)
        self.auto_status_label.setText("강의 소리를 살펴볼 준비를 하고 있어요…")
        thread.start()

    def _cancel_auto_detection(self) -> None:
        if self._auto_cancel_event:
            self._auto_cancel_event.set()
        self.auto_detect_button.setEnabled(False)
        self.auto_status_label.setText("자동 찾기를 안전하게 멈추고 있어요…")

    def _on_auto_progress(self, percent: int, detail: str) -> None:
        self.auto_progress.setValue(max(0, min(100, percent)))
        if percent < 25:
            message = "소리를 살펴볼 준비를 하고 있어요…"
        elif percent < 90:
            message = "말소리와 음악·쉬는 시간을 구분하고 있어요…"
        else:
            message = "찾은 구간을 정리하고 있어요…"
        self.auto_status_label.setText(f"{message} {percent}%")

    def _on_auto_detected(self, result: object) -> None:
        if self._close_after_detection:
            return
        candidates = tuple(
            (float(start), float(end)) for start, end in result  # type: ignore[union-attr]
        )
        self._set_auto_controls_idle()
        if not candidates:
            self._auto_previous_state = None
            self.auto_status_label.setText(
                "자동으로 뺄 구간을 찾지 못했어요. 파형을 들으며 직접 잘라주세요."
            )
            return
        # Snapshot now, so manual edits made while detection ran stay undoable.
        self._auto_previous_state = (
            self.cuts,
            self.exclusions,
            self.selected_segment,
            self._auto_candidates,
        )
        cut_points = [*self.cuts]
        for start, end in candidates:
            if start > self.request.range_start + 0.1:
                cut_points.append(start)
            if end < self.request.range_end - 0.1:
                cut_points.append(end)
        self.cuts = tuple(sorted(set(cut_points)))
        self._cut_history = [cut for cut in self._cut_history if cut in self.cuts]
        self.exclusions = normalize_exclusions(
            (*self.exclusions, *candidates),
            self.request.range_start,
            self.request.range_end,
        )
        self._auto_candidates = candidates
        self.selected_segment = None
        self.undo_auto_button.setEnabled(True)
        self._set_checked_tool("select")
        self._refresh_edit_state()
        removed = sum(end - start for start, end in candidates)
        categories = [self._auto_candidate_name(start, end) for start, end in candidates]
        category_summary = " · ".join(
            f"{name} {categories.count(name)}개" for name in dict.fromkeys(categories)
        )
        self.auto_status_label.setText(
            f"{category_summary} · 모두 {format_duration(removed)}입니다. 소리를 들어보고 맞는지 확인해주세요."
        )
        self._seek_seconds(candidates[0][0])
        self._scroll_to_seconds(candidates[0][0])

    def _on_auto_cancelled(self, message: str) -> None:
        self._auto_previous_state = None
        self._set_auto_controls_idle()
        self.auto_status_label.setText(message)

    def _on_auto_failed(self, message: str) -> None:
        self._auto_previous_state = None
        self._set_auto_controls_idle()
        self.auto_status_label.setText(
            "자동 찾기를 마치지 못했어요. 파형을 클릭해 직접 자르는 기능은 계속 사용할 수 있습니다."
        )
        if not self._close_after_detection:
            QMessageBox.warning(self, "자동 찾기를 마치지 못했습니다", message)

    def _set_auto_controls_idle(self) -> None:
        self.auto_detect_button.setText("쉬는 시간 다시 찾기")
        self.auto_detect_button.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.auto_progress.setVisible(False)

    def _undo_auto_detection(self) -> None:
        if self._auto_previous_state is None:
            return
        (
            self.cuts,
            self.exclusions,
            self.selected_segment,
            self._auto_candidates,
        ) = self._auto_previous_state
        self._auto_previous_state = None
        self.undo_auto_button.setEnabled(False)
        self.auto_status_label.setText("자동으로 찾기 전 상태로 되돌렸어요.")
        self._refresh_edit_state()

    def _on_auto_thread_finished(self) -> None:
        thread = self._auto_thread
        self._auto_thread = None
        self._auto_worker = None
        self._auto_cancel_event = None
        if thread:
            thread.deleteLater()
        if self._close_after_detection:
            self._close_after_detection = False
            self.reject()

    def _add_cut(self, seconds: float) -> None:
        seconds = max(self.request.range_start, min(seconds, self.request.range_end))
        if any(abs(seconds - existing) < 0.08 for existing in self.cuts):
            return
        if seconds <= self.request.range_start + 0.1 or seconds >= self.request.range_end - 0.1:
            return
        self.cuts = tuple(sorted((*self.cuts, seconds)))
        self._cut_history.append(seconds)
        self.selected_segment = None
        self._seek_seconds(seconds)
        self._refresh_edit_state()

    def _undo_cut(self) -> None:
        while self._cut_history and self._cut_history[-1] not in self.cuts:
            self._cut_history.pop()
        if not self._cut_history:
            return
        last = self._cut_history.pop()
        self.cuts = tuple(cut for cut in self.cuts if cut != last)
        self.selected_segment = None
        self._refresh_edit_state()

    def _select_segment(self, index: int) -> None:
        segments = cuts_to_segments(
            self.cuts, self.request.range_start, self.request.range_end
        )
        if index >= len(segments):
            return
        self.selected_segment = index
        start, end = segments[index]
        excluded = self._is_excluded(start, end)
        state = "글에서 빠짐" if excluded else "글에 포함"
        self.clip_label.setText(
            f"구간 {index + 1} · {_precise_time(start)} ~ {_precise_time(end)} · {format_duration(end - start)} · {state}"
        )
        self.toggle_exclusion_button.setText(
            "선택 구간 다시 넣기" if excluded else "선택 구간 빼기"
        )
        self.toggle_exclusion_button.setEnabled(True)
        self.play_clip_button.setEnabled(self._media_ready)
        self._refresh_edit_state()

    def _is_excluded(self, start: float, end: float) -> bool:
        midpoint = (start + end) / 2
        return any(ex_start <= midpoint <= ex_end for ex_start, ex_end in self.exclusions)

    def _toggle_selected_exclusion(self) -> None:
        segments = cuts_to_segments(
            self.cuts, self.request.range_start, self.request.range_end
        )
        if self.selected_segment is None or self.selected_segment >= len(segments):
            return
        start, end = segments[self.selected_segment]
        if self._is_excluded(start, end):
            remaining: list[tuple[float, float]] = []
            for ex_start, ex_end in self.exclusions:
                if ex_end <= start or ex_start >= end:
                    remaining.append((ex_start, ex_end))
                    continue
                if ex_start < start:
                    remaining.append((ex_start, start))
                if ex_end > end:
                    remaining.append((end, ex_end))
            self.exclusions = tuple(remaining)
        else:
            self.exclusions = normalize_exclusions(
                (*self.exclusions, (start, end)),
                self.request.range_start,
                self.request.range_end,
            )
        self._select_segment(self.selected_segment)
        self._refresh_exclusion_list()

    def _refresh_edit_state(self) -> None:
        if self.selected_segment is None:
            self._clear_selection_ui()
        self.waveform.set_edit_state(
            self.cuts, self.exclusions, self.selected_segment
        )
        self._refresh_exclusion_list()

    def _clear_selection_ui(self) -> None:
        self.clip_label.setText(self.NO_SELECTION_TEXT)
        self.toggle_exclusion_button.setText("선택 구간 빼기")
        self.toggle_exclusion_button.setEnabled(False)
        self.play_clip_button.setEnabled(False)

    def _refresh_exclusion_list(self) -> None:
        self.exclusion_list.clear()
        for index, (start, end) in enumerate(self.exclusions, start=1):
            automatic_name = next(
                (
                    self._auto_candidate_name(auto_start, auto_end)
                    for auto_start, auto_end in self._auto_candidates
                    if abs(start - auto_start) < 0.25 and abs(end - auto_end) < 0.25
                ),
                None,
            )
            label = automatic_name or f"글에서 뺄 구간 {index}"
            item = QListWidgetItem(
                f"{label}\n{_precise_time(start)}  –  {_precise_time(end)}  ·  {format_duration(end - start)}"
            )
            item.setSizeHint(QSize(0, 52))
            item.setData(Qt.ItemDataRole.UserRole, (start, end))
            self.exclusion_list.addItem(item)
        has_items = bool(self.exclusions)
        self.exclusion_list.setVisible(has_items)
        self.exclusion_empty.setVisible(not has_items)
        removed = sum(end - start for start, end in self.exclusions)
        self.summary_label.setText(
            f"뺄 구간 {len(self.exclusions)}개 · 합쳐서 {format_duration(removed)}"
            if has_items
            else "뺄 구간 없음"
        )

    def _auto_candidate_name(self, start: float, end: float) -> str:
        editable_duration = self.request.range_end - self.request.range_start
        edge_window = min(600.0, max(120.0, editable_duration * 0.08))
        if start <= self.request.range_start + edge_window:
            return "수업 시작 전 후보"
        if end >= self.request.range_end - edge_window:
            return "수업 종료 후 후보"
        return "쉬는 시간 후보"

    def _seek_to_exclusion(self, item: QListWidgetItem) -> None:
        start, _end = item.data(Qt.ItemDataRole.UserRole)
        self._seek_seconds(float(start))

    def _seek_from_input(self) -> None:
        if not self.time_input.hasAcceptableInput():
            QMessageBox.warning(
                self, "시간 형식을 확인해주세요", "HH:MM:SS.mmm 형식으로 입력해주세요."
            )
            self.time_input.setFocus()
            return
        try:
            seconds = parse_timecode(self.time_input.text())
        except ValueError as exc:
            QMessageBox.warning(self, "시간 형식을 확인해주세요", str(exc))
            return
        if seconds is None or not (
            self.request.range_start <= seconds <= self.request.range_end
        ):
            QMessageBox.warning(
                self,
                "시간 범위를 확인해주세요",
                f"{_precise_time(self.request.range_start)}부터 {_precise_time(self.request.range_end)} 사이를 입력해주세요.",
            )
            return
        self._seek_seconds(seconds)
        self._scroll_to_seconds(seconds)

    def _set_zoom(self, value: int) -> None:
        current = self.waveform.playhead
        self.zoom_label.setText(f"{value}×")
        self.waveform.set_zoom_width(1000 * value)
        self._scroll_to_seconds(current)

    def _zoom_from_wheel(self, steps: int, viewport_x: float) -> None:
        old_value = self.zoom_slider.value()
        new_value = max(
            self.zoom_slider.minimum(),
            min(self.zoom_slider.maximum(), old_value + steps),
        )
        if new_value == old_value:
            return

        bar = self.waveform_scroll.horizontalScrollBar()
        old_width = max(1, self.waveform.width())
        anchor_seconds = (
            (bar.value() + viewport_x) / old_width * self.request.duration
        )
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(new_value)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"{new_value}×")
        self.waveform.set_zoom_width(1000 * new_value)
        new_x = anchor_seconds / self.request.duration * self.waveform.width()
        bar.setValue(round(new_x - viewport_x))

    def _seek_seconds(self, seconds: float) -> None:
        seconds = max(self.request.range_start, min(seconds, self.request.range_end))
        self._pending_seek_seconds = seconds
        self._preview_stop_ms = None
        if self._media_ready:
            self.player.setPosition(round(seconds * 1000))
        self.waveform.set_playhead(seconds)
        if not self.time_input.hasFocus():
            self.time_input.setText(_precise_time(seconds))
        self.position_label.setText(
            f"{_precise_time(seconds)} / {format_duration(self.request.duration)}"
        )

    def _scroll_to_seconds(self, seconds: float) -> None:
        x = round(seconds / self.request.duration * self.waveform.width())
        bar = self.waveform_scroll.horizontalScrollBar()
        bar.setValue(max(0, x - self.waveform_scroll.viewport().width() // 2))

    def _toggle_playback(self) -> None:
        if not self._media_ready:
            return
        self._preview_stop_ms = None
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.player.position() >= round(self.request.range_end * 1000) - 50:
                self.player.setPosition(round(self.request.range_start * 1000))
            self.player.play()

    def _play_selected_clip(self) -> None:
        segments = cuts_to_segments(
            self.cuts, self.request.range_start, self.request.range_end
        )
        if (
            self.selected_segment is None
            or self.selected_segment >= len(segments)
            or not self._media_ready
        ):
            return
        start, end = segments[self.selected_segment]
        self.player.setPosition(round(start * 1000))
        self._preview_stop_ms = round(end * 1000)
        self.player.play()

    def _stop(self) -> None:
        self._preview_stop_ms = None
        self.player.stop()
        self._seek_seconds(self.request.range_start)

    def _on_position_changed(self, position_ms: int) -> None:
        stop_ms = self._preview_stop_ms
        range_end_ms = round(self.request.range_end * 1000)
        if stop_ms is None or stop_ms > range_end_ms:
            stop_ms = range_end_ms
        if position_ms >= stop_ms and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self._preview_stop_ms = None
            position_ms = stop_ms
        seconds = position_ms / 1000
        self.waveform.set_playhead(seconds)
        if not self.time_input.hasFocus():
            self.time_input.setText(_precise_time(seconds))
        self.position_label.setText(
            f"{_precise_time(seconds)} / {format_duration(self.request.duration)}"
        )

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.play_button.setText(
            "일시정지"
            if state == QMediaPlayer.PlaybackState.PlayingState
            else "재생"
        )

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status in {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        }:
            self._media_ready = True
            self.player.setPosition(round(self._pending_seek_seconds * 1000))
            self.play_button.setEnabled(True)
            self.play_clip_button.setEnabled(self.selected_segment is not None)
            self.audio_status_label.setText("미리듣기 준비됨")
        elif status in {
            QMediaPlayer.MediaStatus.LoadingMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
        }:
            self.audio_status_label.setText("오디오 준비 중")

    def _on_player_error(self, error, message: str) -> None:
        if error != QMediaPlayer.Error.NoError:
            self.audio_status_label.setText("미리듣기 오류")
            QMessageBox.warning(
                self,
                "오디오를 재생하지 못했습니다",
                f"시스템 재생기가 이 파일을 열지 못했습니다. 편집은 계속할 수 있습니다.\n{message}",
            )

    def _apply(self) -> None:
        try:
            build_included_ranges(
                self.request.range_start, self.request.range_end, self.exclusions
            )
        except ValueError as exc:
            QMessageBox.warning(self, "편집 결과를 확인해주세요", str(exc))
            return
        self.release_media()
        self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_C:
            self._set_checked_tool("razor")
            event.accept()
            return
        if event.key() == Qt.Key.Key_V:
            self._set_checked_tool("select")
            event.accept()
            return
        if event.key() == Qt.Key.Key_P:
            self._set_checked_tool("playhead")
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_checked_tool(self, tool: str) -> None:
        labels = {"select": "구간 선택", "razor": "자르기", "playhead": "위치 이동"}
        for button in self.tool_group.buttons():
            if button.text().startswith(labels[tool]):
                button.setChecked(True)
                self._set_tool(tool)
                return

    def _confirm_discard(self) -> bool:
        if self._skip_discard_confirm or not (self.cuts or self.exclusions):
            return True
        answer = QMessageBox.question(
            self,
            "편집을 버릴까요?",
            "지금까지 자르거나 뺀 구간이 모두 사라져요. 그래도 닫을까요?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Discard:
            self._skip_discard_confirm = True
            return True
        return False

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._auto_thread and self._auto_thread.isRunning():
            self._close_after_detection = True
            self._cancel_auto_detection()
            event.ignore()
            return
        if not self._confirm_discard():
            event.ignore()
            return
        self.release_media()
        super().closeEvent(event)

    def reject(self) -> None:
        if self._auto_thread and self._auto_thread.isRunning():
            self._close_after_detection = True
            self._cancel_auto_detection()
            return
        if not self._confirm_discard():
            return
        self.release_media()
        super().reject()
