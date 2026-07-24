from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


def configure_fonts(app: QApplication) -> None:
    """Use a crisp native Windows face with complete Korean coverage."""
    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    for candidate in [
        windows_root / "Fonts" / "malgun.ttf",
        windows_root / "Fonts" / "segoeui.ttf",
    ]:
        if not candidate.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(candidate))
        if font_id < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            font = QFont(families[0], 10)
            font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
            app.setFont(font)
            return


# "Studio" system: a dark application chrome, light bento workspace, and a
# dedicated dark media editor. The contrast between these surfaces communicates
# context rather than decorating the old form layout.
APP_STYLE = r"""
QWidget {
    color: #1A1D2B;
    background: #F4F5F8;
    font-family: "Malgun Gothic", "Segoe UI Variable", "Segoe UI", sans-serif;
    font-size: 14px;
}
QLabel { background: transparent; }
QMainWindow, QWidget#AppCanvas { background: #151722; }
QFrame#Workspace, QStackedWidget#PageStack {
    background: #F5F6F9;
    border: 0;
}

/* Sidebar application chrome */
QFrame#Sidebar {
    background: #151722;
    border: 0;
    border-right: 1px solid #292C38;
}
QFrame#SidebarBrandMark {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #8A78FF, stop:1 #5D4DD2);
    border: 1px solid #9486FA;
    border-radius: 12px;
}
QLabel#SidebarBrandLetter { color: #FFFFFF; font-size: 18px; font-weight: 750; }
QLabel#SidebarBrand { color: #F8F8FC; font-size: 15px; font-weight: 700; }
QLabel#SidebarEyebrow {
    color: #7F8493; font-size: 9px; font-weight: 700; letter-spacing: 1.3px;
}
QLabel#NavSection {
    color: #777C8B; font-size: 10px; font-weight: 700;
    letter-spacing: 1px; padding: 0 9px 5px 9px;
}
QPushButton#NavButton {
    color: #AAAFBD;
    background: transparent;
    border: 0;
    border-radius: 11px;
    min-height: 46px;
    padding: 0 13px;
    text-align: left;
    font-weight: 600;
}
QPushButton#NavButton:hover { color: #F3F4F8; background: #20232E; }
QPushButton#NavButton:checked {
    color: #FFFFFF;
    background: #2A2C3C;
    border: 1px solid #393C4F;
}
QPushButton#NavButton:focus { border: 2px solid #8B7DF2; }
QFrame#SidebarStatusCard {
    background: #1C1F2A;
    border: 1px solid #2B2F3C;
    border-radius: 14px;
}
QLabel#SidebarCardTitle { color: #E6E8EF; font-size: 12px; font-weight: 700; }
QFrame#SidebarStatusCard QLabel#StatusGood,
QFrame#SidebarStatusCard QLabel#StatusWarn,
QFrame#SidebarStatusCard QLabel#StatusNeutral {
    background: transparent;
    border: 0;
    border-radius: 0;
    padding: 1px 0;
    font-size: 11px;
    font-weight: 600;
}
QFrame#SidebarStatusCard QLabel#StatusGood { color: #7EDAA4; }
QFrame#SidebarStatusCard QLabel#StatusWarn { color: #F0C879; }
QFrame#SidebarStatusCard QLabel#StatusNeutral { color: #A5AAB8; }
QLabel#SidebarFooter { color: #666B79; font-size: 10px; padding-top: 4px; }

/* Workspace header */
QFrame#WorkspaceHeader { background: transparent; border: 0; }
QLabel#PageTitle { color: #151827; font-size: 23px; font-weight: 750; }
QLabel#PageSubtitle { color: #73798A; font-size: 12px; }
QLabel#PrivacyChip {
    color: #4D5061;
    background: #ECEEF3;
    border: 1px solid #DEE1E8;
    border-radius: 11px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 600;
}

/* Project hero and workflow */
QFrame#ProjectHero {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1C1E2A, stop:0.58 #25243B, stop:1 #352D68);
    border: 1px solid #3C3A57;
    border-radius: 20px;
}
QLabel#HeroKicker {
    color: #A99EFF; font-size: 10px; font-weight: 750; letter-spacing: 1.2px;
}
QLabel#ProjectHeroTitle { color: #FFFFFF; font-size: 23px; font-weight: 750; }
QLabel#ProjectHeroSubtitle { color: #BBBECB; font-size: 12px; }
QFrame#HeroBadge {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
}
QLabel#HeroBadgeNumber { color: #FFFFFF; font-size: 22px; font-weight: 750; }
QLabel#HeroBadgeLabel { color: #BFC1CC; font-size: 10px; }
QFrame[journeyStep="true"] {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 11px;
}
QLabel#JourneyNumber {
    color: #FFFFFF;
    background: #7162E4;
    border-radius: 9px;
    min-width: 18px; max-width: 18px;
    min-height: 18px; max-height: 18px;
    font-size: 10px; font-weight: 750;
}
QLabel#JourneyText { color: #E1E2E9; font-size: 11px; font-weight: 600; }

/* Bento surfaces */
QScrollArea, QScrollArea > QWidget > QWidget { background: #F5F6F9; border: 0; }
QFrame#Card {
    background: #FFFFFF;
    border: 1px solid #E0E2E9;
    border-radius: 18px;
}
QFrame#Card[cardRole="source"] { border: 1px solid #D9D8EA; }
QFrame#Card[cardRole="settings"] { background: #F0F0F7; border-color: #E0E0EC; }
QFrame#Card[cardRole="progress"] { border-color: #CBC7EE; }
QFrame#Card[cardRole="connection"] { border-color: #D8D5EF; }
QFrame#Card[cardRole="update"] { background: #F0F0F8; }
QFrame#ActionDock {
    background: #181A25;
    border: 1px solid #2E3140;
    border-radius: 16px;
}
QLabel#DockTitle { color: #FFFFFF; font-size: 14px; font-weight: 700; }
QLabel#DockHelper { color: #959AA8; font-size: 11px; }
QFrame#StagePanel {
    background: rgba(255,255,255,0.68);
    border: 1px solid #D9DBE4;
    border-radius: 12px;
}
QLabel#SectionTitle { color: #1A1D2B; font-size: 17px; font-weight: 700; }
QLabel#FieldLabel { color: #3A3E4E; font-size: 12px; font-weight: 650; }
QLabel#HelperText { color: #747A8A; font-size: 11px; }
QLabel#ResultTitle { color: #171A28; font-size: 16px; font-weight: 700; }

/* Note profile */
QFrame#NoteProfile {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ECEAFF, stop:1 #F5F1FF);
    border: 1px solid #DCD7FA;
    border-radius: 18px;
}
QLabel#ProfileKicker { color: #6859D3; font-size: 10px; font-weight: 750; }
QLabel#ProfileTitle { color: #242037; font-size: 16px; font-weight: 700; }
QLabel#ProfileText { color: #6F6980; font-size: 11px; }

/* Status chips */
QLabel#StatusGood {
    color: #17653A; background: #EAF7EF; border: 1px solid #C9E7D4;
    border-radius: 9px; padding: 5px 9px; font-weight: 650;
}
QLabel#StatusWarn {
    color: #845117; background: #FFF5E5; border: 1px solid #EED6A9;
    border-radius: 9px; padding: 5px 9px; font-weight: 650;
}
QLabel#StatusNeutral {
    color: #53596A; background: #EEF0F4; border: 1px solid #DDE0E7;
    border-radius: 9px; padding: 5px 9px; font-weight: 650;
}

/* Form controls */
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
    color: #1C1F2C;
    background: #FBFCFE;
    border: 1px solid #CDD2DC;
    border-radius: 10px;
    padding: 9px 11px;
    selection-background-color: #6B5CDE;
    selection-color: #FFFFFF;
}
QLineEdit, QComboBox, QSpinBox { min-height: 24px; }
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QPlainTextEdit:hover, QListWidget:hover {
    border-color: #A9AFBC;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QListWidget:focus {
    border: 2px solid #7567E4;
    padding: 8px 10px;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QPlainTextEdit:disabled {
    color: #989EAA; background: #ECEEF2; border-color: #DEE1E7;
}
QComboBox::drop-down { width: 28px; border: 0; }
QComboBox QAbstractItemView {
    color: #1C1F2C; background: #FFFFFF; border: 1px solid #D5D8E0;
    selection-color: #28234E; selection-background-color: #EAE8FF; padding: 5px;
}
QPlainTextEdit#PromptEditor {
    background: #F9F9FC;
    border-color: #DADCE5;
    padding: 15px;
}

QPushButton {
    min-height: 40px;
    color: #303443;
    background: #FFFFFF;
    border: 1px solid #CACFD9;
    border-radius: 10px;
    padding: 0 15px;
    font-weight: 650;
}
QPushButton:hover { color: #242735; background: #F4F5F8; border-color: #A7ADBA; }
QPushButton:pressed { background: #E8EAF0; }
QPushButton:focus { border: 2px solid #7567E4; }
QPushButton:disabled { color: #A2A7B2; background: #EDEEF2; border-color: #DFE1E7; }
QPushButton#PrimaryButton {
    min-height: 46px;
    color: #FFFFFF;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7A69ED, stop:1 #5F52D4);
    border: 1px solid #6959DB;
    border-radius: 11px;
    padding: 0 22px;
    font-weight: 700;
}
QPushButton#PrimaryButton:hover { background: #6758DD; border-color: #594BC9; }
QPushButton#PrimaryButton:pressed { background: #5649C8; }
QPushButton#AccentButton { color: #5A4BC6; background: #EBE8FF; border-color: #D4CEF8; }
QPushButton#AccentButton:hover { background: #DED9FF; border-color: #C3BCF2; }
QPushButton#SecondaryButton { color: #5B4DC9; border-color: #C9C5E7; }
QPushButton#DangerButton { color: #B42318; background: #FFFFFF; border-color: #EBC0BC; }
QPushButton#DangerButton:hover { background: #FFF3F2; border-color: #E3A29C; }
QPushButton#DangerButton:disabled { color: #9FA4AE; background: #EDEEF2; border-color: #DFE1E7; }

QProgressBar {
    min-height: 9px; max-height: 9px; border: 0; border-radius: 4px;
    background: #E2E4EA; color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #8271F0, stop:1 #5D51D3);
    border-radius: 4px;
}
QCheckBox { spacing: 9px; color: #3E4352; min-height: 24px; background: transparent; }
QCheckBox::indicator { width: 18px; height: 18px; }
QCheckBox::indicator:unchecked { image: url(lecture_scribe/ui/assets/check_unchecked.svg); }
QCheckBox::indicator:checked { image: url(lecture_scribe/ui/assets/check_checked.svg); }
QLabel#TimeLabel {
    color: #2B2F3D; font-family: "Cascadia Mono", "Consolas", monospace;
    font-weight: 650; min-width: 180px;
}
QSlider::groove:horizontal { height: 6px; background: #D9DCE4; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #7062E1; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 18px; margin: -7px 0; background: #FFFFFF;
    border: 1px solid #858C9B; border-radius: 9px;
}
QListWidget::item { min-height: 34px; padding: 5px 9px; border-radius: 7px; }
QListWidget::item:selected { color: #2D275E; background: #E9E6FF; }

/* Dark media-editing workspace */
QDialog#EditorDialog, QDialog#EditorDialog QWidget { background: #0D0F15; color: #D9DCE6; }
QDialog#EditorDialog QLabel { background: transparent; }
QDialog#EditorDialog QFrame#EditorHeader {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #171923, stop:1 #26213F);
    border: 1px solid #303345;
    border-radius: 17px;
}
QDialog#EditorDialog QLabel#Eyebrow { color: #9F93FF; font-size: 10px; font-weight: 750; }
QDialog#EditorDialog QLabel#EditorTitle { color: #FFFFFF; font-size: 24px; font-weight: 750; }
QDialog#EditorDialog QLabel#HelperText { color: #8E94A3; font-size: 11px; }
QDialog#EditorDialog QFrame#Toolbar {
    background: #151821; border: 1px solid #2A2E3C; border-radius: 12px;
}
QDialog#EditorDialog QFrame#EditorPanel {
    background: #13161E; border: 1px solid #292D3A; border-radius: 14px;
}
QDialog#EditorDialog QFrame#EditorPanel QLabel { color: #D5D8E2; }
QDialog#EditorDialog QLabel#InspectorKicker {
    color: #9184F4; font-size: 9px; font-weight: 750; letter-spacing: 1px;
}
QDialog#EditorDialog QLabel#InspectorTitle { color: #FFFFFF; font-size: 16px; font-weight: 700; }
QDialog#EditorDialog QLabel#InspectorSection { color: #AEB3C1; font-size: 11px; font-weight: 700; }
QDialog#EditorDialog QPushButton {
    color: #DADDE6; background: #1A1D27; border-color: #333746;
}
QDialog#EditorDialog QPushButton:hover { color: #FFFFFF; background: #242835; border-color: #4A5062; }
QDialog#EditorDialog QPushButton:disabled { color: #5E6370; background: #161820; border-color: #272A34; }
QDialog#EditorDialog QPushButton[editorTool="true"] {
    min-height: 38px; background: transparent; border-color: transparent; padding: 0 13px;
}
QDialog#EditorDialog QPushButton[editorTool="true"]:checked {
    color: #FFFFFF; background: #7161E5; border-color: #8173E9;
}
QDialog#EditorDialog QPushButton#PrimaryButton {
    color: #FFFFFF; background: #7161E5; border-color: #8173E9;
}
QDialog#EditorDialog QPushButton#PrimaryButton:hover { background: #806FF0; }
QDialog#EditorDialog QPushButton#AccentButton {
    color: #C9C2FF; background: #28243E; border-color: #474064;
}
QDialog#EditorDialog QPushButton#SecondaryButton { color: #B9B0FF; background: #171922; }
QDialog#EditorDialog QLineEdit, QDialog#EditorDialog QComboBox,
QDialog#EditorDialog QSpinBox, QDialog#EditorDialog QPlainTextEdit,
QDialog#EditorDialog QListWidget {
    color: #E2E4EC; background: #10131A; border-color: #343847;
    selection-background-color: #7161E5;
}
QDialog#EditorDialog QLineEdit:focus, QDialog#EditorDialog QPlainTextEdit:focus,
QDialog#EditorDialog QListWidget:focus { border: 2px solid #8273EF; }
QDialog#EditorDialog QLabel#TimeLabel { color: #DDE0E8; }
QDialog#EditorDialog QProgressBar { background: #292C37; }
QDialog#EditorDialog QSlider::groove:horizontal { background: #343846; }
QDialog#EditorDialog QSlider::sub-page:horizontal { background: #8273EF; }
QDialog#EditorDialog QScrollArea#TimelineScroll,
QDialog#EditorDialog QScrollArea#TimelineScroll > QWidget > QWidget { background: #10121A; }
QDialog#EditorDialog QListWidget::item:selected { color: #FFFFFF; background: #302A55; }

QScrollBar:vertical { background: transparent; width: 11px; margin: 2px; }
QScrollBar:horizontal { background: #252832; height: 12px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #8F95A3; border-radius: 5px; min-height: 34px; min-width: 34px;
}
QDialog#EditorDialog QScrollBar::handle:vertical,
QDialog#EditorDialog QScrollBar::handle:horizontal { background: #555B6A; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QToolTip { color: #FFFFFF; background: #20232D; border: 1px solid #383C49; padding: 7px; }
"""
