from __future__ import annotations

from string import Template

from .theme import COLORS, FONT_STACK, MONO_STACK

# Dark stylesheet for the audio editor dialog. Every rule is scoped under
# QDialog#EditorDialog so it fully overrides the light app-wide stylesheet.
_TEMPLATE = Template(
    """
QDialog#EditorDialog {
    background: $bg;
    color: $text;
    font-family: $font;
}
QDialog#EditorDialog QWidget { font-family: $font; }
QDialog#EditorDialog QLabel { color: $text; background: transparent; font-size: 13px; }
QDialog#EditorDialog QLabel#EditorKicker {
    color: $accent; font-size: 11px; font-weight: 700; letter-spacing: 1px;
}
QDialog#EditorDialog QLabel#EditorTitle { color: $text; font-size: 22px; font-weight: 700; }
QDialog#EditorDialog QLabel#HelperText { color: $text2; font-size: 13px; }
QDialog#EditorDialog QLabel#CardTitle { color: $text; font-size: 14px; font-weight: 700; }
QDialog#EditorDialog QLabel#CardKicker {
    color: $text3; font-size: 11px; font-weight: 700; letter-spacing: 1px;
}
QDialog#EditorDialog QLabel#TimeLabel {
    color: $text; font-family: $mono; font-size: 14px; font-weight: 600;
}
QDialog#EditorDialog QLabel#MutedLabel { color: $text3; font-size: 12px; }
QDialog#EditorDialog QLabel#EmptyState { color: $text3; font-size: 12px; padding: 14px 8px; }

QDialog#EditorDialog QFrame#EditorPanel,
QDialog#EditorDialog QFrame#Toolbar,
QDialog#EditorDialog QFrame#EditorCard {
    background: $surface; border: 1px solid $border; border-radius: 12px;
}
QDialog#EditorDialog QFrame#EditorCard[tone="raised"] {
    background: $elevated; border-radius: 8px;
}
QDialog#EditorDialog QFrame#Toolbar { border-radius: 8px; }
QDialog#EditorDialog QFrame#Legend { background: transparent; border: 0; }

QDialog#EditorDialog QPushButton {
    background: $elevated; color: $text; border: 1px solid $border;
    border-radius: 8px; padding: 6px 14px; min-height: 22px; font-size: 13px;
}
QDialog#EditorDialog QPushButton:hover { background: $border; }
QDialog#EditorDialog QPushButton:pressed { background: $surface; }
QDialog#EditorDialog QPushButton:focus { border-color: $accent; }
QDialog#EditorDialog QPushButton:disabled {
    color: $text3; background: $surface; border-color: $border;
}
QDialog#EditorDialog QPushButton#PrimaryButton,
QDialog#EditorDialog QPushButton#AccentButton {
    background: $accent; color: $bg; border: 1px solid $accent; font-weight: 700;
}
QDialog#EditorDialog QPushButton#PrimaryButton:hover,
QDialog#EditorDialog QPushButton#AccentButton:hover { background: $text; border-color: $text; }
QDialog#EditorDialog QPushButton#PrimaryButton:disabled,
QDialog#EditorDialog QPushButton#AccentButton:disabled {
    background: $elevated; color: $text3; border-color: $border;
}
QDialog#EditorDialog QPushButton#GhostButton {
    background: transparent; border: 1px solid transparent; color: $text2;
}
QDialog#EditorDialog QPushButton#GhostButton:hover { background: $elevated; color: $text; }
QDialog#EditorDialog QPushButton[editorTool="true"] {
    background: transparent; color: $text2; border: 1px solid transparent;
    border-radius: 6px; padding: 5px 12px;
}
QDialog#EditorDialog QPushButton[editorTool="true"]:hover { background: $elevated; color: $text; }
QDialog#EditorDialog QPushButton[editorTool="true"]:checked {
    background: $elevated; color: $text; border: 1px solid $accent; font-weight: 700;
}
QDialog#EditorDialog QPushButton[compact="true"] {
    min-height: 0; padding: 2px 8px; border-radius: 6px;
}

QDialog#EditorDialog QLineEdit {
    background: $bg; color: $text; border: 1px solid $border; border-radius: 6px;
    padding: 5px 9px; min-height: 22px; font-family: $mono; font-size: 13px;
    selection-background-color: $accent; selection-color: $bg;
}
QDialog#EditorDialog QLineEdit:focus { border-color: $accent; }

QDialog#EditorDialog QSlider::groove:horizontal {
    height: 4px; background: $border; border-radius: 2px;
}
QDialog#EditorDialog QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QDialog#EditorDialog QSlider::handle:horizontal {
    background: $text; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}
QDialog#EditorDialog QSlider::handle:horizontal:hover { background: $accent; }

QDialog#EditorDialog QProgressBar {
    background: $border; border: 0; border-radius: 3px; max-height: 6px; text-align: center;
    color: transparent;
}
QDialog#EditorDialog QProgressBar::chunk { background: $accent; border-radius: 3px; }

QDialog#EditorDialog QListWidget {
    background: $bg; color: $text; border: 1px solid $border; border-radius: 8px;
    padding: 4px; outline: 0; font-size: 13px;
}
QDialog#EditorDialog QListWidget::item { padding: 6px 8px; border-radius: 6px; }
QDialog#EditorDialog QListWidget::item:hover { background: $elevated; }
QDialog#EditorDialog QListWidget::item:selected { background: $elevated; color: $text; }

QDialog#EditorDialog QScrollArea { background: transparent; border: 0; }
QDialog#EditorDialog QScrollArea > QWidget > QWidget { background: transparent; }
QDialog#EditorDialog QScrollArea#TimelineScroll {
    background: $bg; border: 1px solid $border; border-radius: 8px;
}
QDialog#EditorDialog QScrollBar:horizontal {
    background: transparent; height: 12px; margin: 0;
}
QDialog#EditorDialog QScrollBar:vertical {
    background: transparent; width: 12px; margin: 0;
}
QDialog#EditorDialog QScrollBar::handle:horizontal {
    background: $border; border-radius: 4px; min-width: 32px; margin: 2px;
}
QDialog#EditorDialog QScrollBar::handle:vertical {
    background: $border; border-radius: 4px; min-height: 32px; margin: 2px;
}
QDialog#EditorDialog QScrollBar::handle:horizontal:hover,
QDialog#EditorDialog QScrollBar::handle:vertical:hover { background: $text3; }
QDialog#EditorDialog QScrollBar::add-line, QDialog#EditorDialog QScrollBar::sub-line {
    width: 0; height: 0; background: none; border: 0;
}
QDialog#EditorDialog QScrollBar::add-page, QDialog#EditorDialog QScrollBar::sub-page {
    background: transparent;
}

QDialog#EditorDialog QMessageBox { background: $surface; }
QDialog#EditorDialog QToolTip {
    background: $elevated; color: $text; border: 1px solid $border; padding: 4px 8px;
}
"""
)

EDITOR_STYLE: str = _TEMPLATE.substitute(
    bg=COLORS["editor_bg"],
    surface=COLORS["editor_surface"],
    elevated=COLORS["editor_elevated"],
    border=COLORS["editor_border"],
    text=COLORS["editor_text"],
    text2=COLORS["editor_text_secondary"],
    text3=COLORS["editor_text_tertiary"],
    accent=COLORS["editor_accent"],
    font=FONT_STACK,
    mono=MONO_STACK,
)
