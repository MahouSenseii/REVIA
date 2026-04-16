"""Compose Qt stylesheets from ThemeDefinition tokens."""
from __future__ import annotations


class StyleComposer:
    def compose(self, theme) -> str:
        t = theme.tokens()
        return f"""
QWidget {{
    background-color: {t["PrimaryBackground"]};
    color: {t["PrimaryText"]};
    selection-background-color: {t["AccentActive"]};
    selection-color: {t["PrimaryText"]};
}}

QWidget#centerPanel, QScrollArea, QTabWidget::pane {{
    background-color: {t["SecondaryBackground"]};
    border: 1px solid {t["Border"]};
}}

QFrame#sidebar, QFrame#topBar, QFrame#statusPanel,
QFrame#chatPanel, QFrame#inferencePanel {{
    background-color: {t["Surface"]};
    border: 1px solid {t["Border"]};
}}

QGroupBox, QTextEdit, QPlainTextEdit, QListWidget, QTableWidget {{
    background-color: {t["Surface"]};
    border: 1px solid {t["Border"]};
    border-radius: 6px;
}}

QGroupBox::title {{
    color: {t["SecondaryText"]};
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}

QLabel {{
    color: {t["PrimaryText"]};
    background: transparent;
}}

QLabel#tabHeader, QLabel#panelHeader, QLabel#sidebarName {{
    color: {t["Accent"]};
}}

QLabel#sidebarRole, QLabel#sidebarSection, QLabel#metricLabel {{
    color: {t["SecondaryText"]};
}}

QLabel#metricValue {{
    color: {t["PrimaryText"]};
}}

QLabel[statusRole="success"] {{
    color: {t["Success"]};
}}

QLabel[statusRole="warning"] {{
    color: {t["Warning"]};
}}

QLabel[statusRole="error"] {{
    color: {t["Error"]};
}}

QLabel[statusRole="info"] {{
    color: {t["Info"]};
}}

QLabel[statusRole="accent"] {{
    color: {t["Accent"]};
}}

QLabel[statusRole="muted"] {{
    color: {t["SecondaryText"]};
}}

QLabel#statusDot {{
    font-size: 10px;
}}

QLabel#pill, QLabel#titlePill, QLabel#healthPill,
QLabel#healthPillOnline {{
    background-color: {t["SurfaceAlt"]};
    border: 1px solid {t["Border"]};
    border-radius: 5px;
    padding: 4px 8px;
}}

QLabel#healthPillOnline {{
    color: {t["Success"]};
    border-color: {t["Success"]};
}}

QLabel#chatActivity {{
    border-top: 1px solid {t["Border"]};
    padding: 6px 10px;
}}

QLabel#chatActivity[activityState="active"] {{
    color: {t["Warning"]};
    background-color: {t["SurfaceAlt"]};
    border-top-color: {t["Warning"]};
}}

QLabel#chatActivity[activityState="idle"] {{
    color: {t["SecondaryText"]};
    background-color: {t["Surface"]};
}}

QFrame#chatInputRow {{
    background-color: {t["Surface"]};
    border-top: 1px solid {t["Border"]};
}}

QFrame#webcamFrame {{
    background-color: {t["SurfaceAlt"]};
    border: 1px solid {t["Border"]};
    border-radius: 6px;
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {t["SurfaceAlt"]};
    color: {t["PrimaryText"]};
    border: 1px solid {t["Border"]};
    border-radius: 4px;
    padding: 5px 7px;
}}

QPushButton {{
    background-color: {t["ButtonSecondary"]};
    color: {t["PrimaryText"]};
    border: 1px solid {t["Border"]};
    border-radius: 5px;
    padding: 6px 10px;
}}

QPushButton:hover {{
    border-color: {t["AccentHover"]};
}}

QPushButton:pressed {{
    background-color: {t["AccentActive"]};
}}

QPushButton[primary="true"] {{
    background-color: {t["ButtonPrimary"]};
    border-color: {t["Accent"]};
}}

QPushButton#primaryBtn, QPushButton#sendBtn {{
    background-color: {t["ButtonPrimary"]};
    border-color: {t["Accent"]};
}}

QPushButton#micBtn:checked {{
    background-color: {t["Error"]};
    border-color: {t["Error"]};
}}

QPushButton#visionBtn:checked {{
    background-color: {t["Info"]};
    border-color: {t["Info"]};
}}

QPushButton:disabled, QLineEdit:disabled, QComboBox:disabled {{
    color: {t["Disabled"]};
    border-color: {t["Border"]};
}}

QTabBar::tab {{
    background-color: {t["Surface"]};
    color: {t["SecondaryText"]};
    border: 1px solid {t["Border"]};
    padding: 7px 10px;
}}

QTabBar::tab:selected {{
    background-color: {t["SurfaceAlt"]};
    color: {t["PrimaryText"]};
    border-bottom-color: {t["Accent"]};
}}

QScrollBar:vertical, QScrollBar:horizontal {{
    background: {t["SecondaryBackground"]};
    border: none;
}}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {t["Disabled"]};
    border-radius: 4px;
}}

QSplitter::handle {{
    background-color: {t["Border"]};
}}

QProgressBar {{
    background-color: {t["SurfaceAlt"]};
    border: 1px solid {t["Border"]};
    border-radius: 4px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {t["Accent"]};
}}

QFrame#themeSwatch {{
    border: 1px solid {t["Border"]};
    border-radius: 4px;
    min-width: 28px;
    min-height: 22px;
}}
"""
