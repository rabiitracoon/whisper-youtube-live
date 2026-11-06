from __future__ import annotations

import os
import platform
from pathlib import Path
from string import Template

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


# Shared design tokens. Custom-painted widgets and inline styles must read from
# here instead of hard-coding hex values. Calm light workspace (Linear/Granola
# inspired), a single indigo accent, hairline borders, and a dark media editor.
COLORS: dict[str, str] = {
    # light workspace
    "bg": "#F7F7F8",
    "surface": "#FFFFFF",
    "surface_muted": "#F2F3F5",
    "sidebar": "#F0F1F3",
    "sidebar_active": "#FFFFFF",
    "border": "#E4E5E9",
    "border_strong": "#D3D5DB",
    "text": "#16171A",
    "text_secondary": "#5B5F68",
    "text_tertiary": "#8B909A",
    "accent": "#4F46E5",
    "accent_hover": "#4338CA",
    "accent_pressed": "#3730A3",
    "accent_on": "#FFFFFF",
    "accent_subtle": "#EEF0FF",
    "accent_border": "#C7CBFA",
    "success": "#15803D",
    "success_subtle": "#E8F7EE",
    "warning": "#B45309",
    "warning_subtle": "#FFF5E0",
    "danger": "#DC2626",
    "danger_subtle": "#FDECEC",
    "focus": "#818CF8",
    "sidebar_hover": "#E6E7EB",
    "success_border": "#BFE5CB",
    "warning_border": "#F2D9A6",
    "danger_border": "#F3C1C1",
    # dark media editor / overlay
    "editor_bg": "#0F1012",
    "editor_surface": "#17181B",
    "editor_elevated": "#1F2125",
    "editor_border": "#2A2C31",
    "editor_text": "#ECEDEF",
    "editor_text_secondary": "#A0A4AD",
    "editor_text_tertiary": "#6E727B",
    "editor_accent": "#7C83FF",
    "wave_kept": "#7C83FF",
    "wave_removed": "#F59E0B",
    "wave_cut": "#F472B6",
    "wave_playhead": "#2DD4BF",
    "wave_selection": "#7C83FF",
    "overlay_bg": "rgba(15, 16, 18, 220)",
}

FONT_STACK = '"Pretendard", "Pretendard Variable", "Apple SD Gothic Neo", "Malgun Gothic", "Segoe UI", sans-serif'
MONO_STACK = '"Cascadia Mono", "Consolas", "Menlo", monospace'


ASSET_DIR = Path(__file__).resolve().parent / "assets"


def configure_fonts(app: QApplication) -> None:
    """Set the base application font: 14px, Pretendard if installed, else a native Korean face.

    The size lives here (not in the stylesheet's generic QWidget rule) so that
    widgets calling setFont() keep their own size.
    """
    system = platform.system()
    if system == "Windows":
        # Register the Korean system faces explicitly so they also resolve on
        # platforms without system font enumeration (e.g. offscreen smoke runs).
        fonts_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("malgun.ttf", "malgunbd.ttf", "segoeui.ttf"):
            if (fonts_dir / name).exists():
                QFontDatabase.addApplicationFont(str(fonts_dir / name))
    families = set(QFontDatabase.families())
    if "Pretendard" in families:
        family = "Pretendard"
    elif system == "Darwin":
        family = "Apple SD Gothic Neo"
    elif system == "Windows":
        family = "Malgun Gothic"
    else:
        family = "Noto Sans CJK KR" if "Noto Sans CJK KR" in families else app.font().family()
    font = QFont(family)
    font.setPixelSize(14)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)


_STYLE_TEMPLATE = Template(
    r"""
/* ---- base ---------------------------------------------------------- */
QMainWindow, QWidget#AppCanvas { background: $bg; }
QWidget { color: $text; }
QLabel { background: transparent; }
QMessageBox, QInputDialog { background: $surface; }
QToolTip {
    color: $text; background: $surface; border: 1px solid $border_strong;
    padding: 6px 8px; font-family: $font; font-size: 12px;
}
QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {
    background: transparent; border: 0;
}

/* ---- sidebar ------------------------------------------------------- */
QFrame#Sidebar { background: $sidebar; border: 0; border-right: 1px solid $border; }
QFrame#BrandDot { background: $accent; border: 0; border-radius: 5px; }
QLabel#SidebarBrand { color: $text; font-size: 15px; font-weight: 700; }
QPushButton#NavButton {
    color: $text_secondary; background: transparent;
    border: 1px solid transparent; border-radius: 8px;
    min-height: 36px; padding: 0 12px; text-align: left; font-weight: 500;
}
QPushButton#NavButton:hover { color: $text; background: $sidebar_hover; }
QPushButton#NavButton:checked {
    color: $text; background: $sidebar_active; border-color: $border; font-weight: 600;
}
QPushButton#NavButton:focus { border-color: $accent; }
QLabel#SidebarMeta { color: $text_secondary; font-size: 12px; font-weight: 500; }

/* ---- surfaces and text -------------------------------------------- */
QFrame#Card { background: $surface; border: 1px solid $border; border-radius: 12px; }
QFrame#Card[tone="muted"] { background: $surface_muted; }
QLabel#PageTitle { color: $text; font-size: 24px; font-weight: 700; }
QLabel#PageSubtitle { color: $text_secondary; font-size: 14px; }
QLabel#SectionTitle { color: $text; font-size: 16px; font-weight: 600; }
QLabel#FieldLabel { color: $text_secondary; font-size: 13px; font-weight: 500; }
QLabel#HelperText { color: $text_tertiary; font-size: 12px; }
QLabel#BodyText { color: $text_secondary; font-size: 13px; }
QLabel#InstructionText { color: $text; font-size: 13px; font-weight: 500; }
QLabel#ResultTitle { color: $text; font-size: 16px; font-weight: 600; }
QLabel#TimeLabel { font-family: $mono; font-weight: 600; }

/* ---- status pills -------------------------------------------------- */
QLabel#StatusPill {
    color: $text_secondary; background: $surface; border: 1px solid $border;
    border-radius: 11px; padding: 2px 10px; font-size: 12px; font-weight: 600;
}
QLabel#StatusPill[tone="success"] { color: $success; background: $success_subtle; border-color: $success_border; }
QLabel#StatusPill[tone="warning"] { color: $warning; background: $warning_subtle; border-color: $warning_border; }
QLabel#StatusPill[tone="danger"] { color: $danger; background: $danger_subtle; border-color: $danger_border; }
QLabel#StatusPill[tone="accent"] { color: $accent; background: $accent_subtle; border-color: $accent_border; }

/* ---- buttons ------------------------------------------------------- */
QPushButton {
    min-height: 36px; padding: 0 14px; border-radius: 8px;
    color: $text; background: $surface; border: 1px solid $border_strong;
    font-weight: 500;
}
QPushButton:hover { background: $surface_muted; }
QPushButton:pressed { background: $border; }
QPushButton:focus { border-color: $accent; }
QPushButton:disabled { color: $text_tertiary; background: $surface_muted; border-color: $border; }
QPushButton#SecondaryButton { color: $text; background: $surface; border: 1px solid $border_strong; }
QPushButton#SecondaryButton:hover { background: $surface_muted; }
QPushButton#PrimaryButton {
    color: $accent_on; background: $accent; border: 1px solid $accent; font-weight: 600;
}
QPushButton#PrimaryButton:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton#PrimaryButton:pressed { background: $accent_pressed; border-color: $accent_pressed; }
QPushButton#PrimaryButton:focus { border-color: $accent_pressed; }
QPushButton#PrimaryButton:disabled {
    color: $accent_on; background: $accent_border; border-color: $accent_border;
}
QPushButton#GhostButton {
    color: $text_secondary; background: transparent; border: 1px solid transparent;
}
QPushButton#GhostButton:hover { color: $text; background: $surface_muted; }
QPushButton#GhostButton:pressed { background: $border; }
QPushButton#GhostButton:focus { border-color: $accent; }
QPushButton#GhostButton:disabled { color: $text_tertiary; background: transparent; }
QPushButton#GhostButton[disclosure="true"] { text-align: left; padding: 0 8px; }
QPushButton#DangerButton {
    color: $danger; background: transparent; border: 1px solid transparent;
}
QPushButton#DangerButton:hover { background: $danger_subtle; }
QPushButton#DangerButton:pressed { background: $danger_border; }
QPushButton#DangerButton:focus { border-color: $danger; }
QPushButton#DangerButton:disabled { color: $text_tertiary; background: transparent; }
QPushButton[compact="true"] { min-height: 0; padding: 0 6px; }

/* ---- inputs -------------------------------------------------------- */
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
    color: $text; background: $surface;
    border: 1px solid $border_strong; border-radius: 8px;
    selection-background-color: $accent; selection-color: $accent_on;
}
QLineEdit, QComboBox, QSpinBox { min-height: 36px; padding: 0 12px; }
QPlainTextEdit { padding: 8px 12px; }
QLineEdit#SourceInput { min-height: 44px; font-size: 15px; }
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QPlainTextEdit:hover { border-color: $text_tertiary; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QListWidget:focus {
    border-color: $accent;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QPlainTextEdit:disabled {
    color: $text_tertiary; background: $surface_muted; border-color: $border;
}
QPlainTextEdit#LogView, QPlainTextEdit#ToolLog {
    font-family: $mono; font-size: 12px; background: $surface_muted; border-color: $border;
}
QComboBox { padding-right: 32px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 32px; border: 0; }
QComboBox::down-arrow { image: url("$asset/chevron-down.svg"); width: 12px; height: 12px; }
QComboBox QAbstractItemView {
    color: $text; background: $surface; border: 1px solid $border_strong;
    selection-color: $text; selection-background-color: $accent_subtle;
    outline: 0; padding: 4px;
}
QSpinBox { padding-right: 24px; }
QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border; width: 20px; border: 0; background: transparent;
}
QSpinBox::up-button { subcontrol-position: top right; margin-top: 2px; }
QSpinBox::down-button { subcontrol-position: bottom right; margin-bottom: 2px; }
QSpinBox::up-arrow { image: url("$asset/spin-up.svg"); width: 10px; height: 10px; }
QSpinBox::down-arrow { image: url("$asset/spin-down.svg"); width: 10px; height: 10px; }
QListWidget::item { min-height: 32px; padding: 4px 8px; border-radius: 6px; }
QListWidget::item:selected { color: $text; background: $accent_subtle; }

/* ---- checkbox / progress / slider --------------------------------- */
QCheckBox { spacing: 8px; color: $text; min-height: 28px; background: transparent; }
QCheckBox:disabled { color: $text_tertiary; }
QCheckBox::indicator { width: 18px; height: 18px; }
QCheckBox::indicator:unchecked { image: url("$asset/check_unchecked.svg"); }
QCheckBox::indicator:checked { image: url("$asset/check_checked.svg"); }
QProgressBar {
    min-height: 6px; max-height: 6px; border: 0; border-radius: 3px;
    background: $border; color: transparent;
}
QProgressBar::chunk { background: $accent; border-radius: 3px; }
QSlider::groove:horizontal { height: 4px; background: $border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px; margin: -6px 0; background: $surface;
    border: 1px solid $border_strong; border-radius: 8px;
}

/* ---- stepper ------------------------------------------------------- */
QLabel#StepDot {
    min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px;
    border-radius: 12px; font-size: 12px; font-weight: 600;
    color: $text_tertiary; background: $surface; border: 1px solid $border_strong;
}
QLabel#StepDot[state="active"] { color: $accent_on; background: $accent; border-color: $accent; }
QLabel#StepDot[state="done"] { color: $accent_on; background: $accent; border-color: $accent; }
QLabel#StepDot[state="error"] { color: $accent_on; background: $danger; border-color: $danger; }
QLabel#StepLabel { color: $text_tertiary; font-size: 13px; font-weight: 500; }
QLabel#StepLabel[state="active"] { color: $text; font-weight: 600; }
QLabel#StepLabel[state="done"] { color: $text_secondary; }
QLabel#StepLabel[state="error"] { color: $danger; font-weight: 600; }
QFrame#StepLine { background: $border_strong; border: 0; min-height: 2px; max-height: 2px; }
QFrame#StepLine[done="true"] { background: $accent; }

/* ---- scrollbars ---------------------------------------------------- */
QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: $border_strong; border-radius: 4px; min-height: 32px; min-width: 32px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: $text_tertiary; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""
)


def build_app_style() -> str:
    return _STYLE_TEMPLATE.substitute(
        COLORS,
        font=FONT_STACK.replace('"', "'"),
        mono=MONO_STACK.replace('"', "'"),
        asset=ASSET_DIR.as_posix(),
    )


APP_STYLE = build_app_style()
