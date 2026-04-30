from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog,
    QCheckBox,
    QComboBox,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.backend_sync_client import BackendSyncClient
from app.theme_manager import THEME_TOKENS, ThemeDefinition
from app.ui_status import apply_status_style, clear_status_role
from gui.widgets.settings_card import SettingsCard


class ThemeTab(QScrollArea):
    """Theme editor owned by the controller UI and synced to Core."""

    def __init__(self, event_bus, client, theme_mgr, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.theme_mgr = theme_mgr
        self._sync_client = BackendSyncClient(event_bus)
        self._loading = False
        self._source_theme_id = ""
        self._token_fields: dict[str, QLineEdit] = {}
        self._swatches: dict[str, QFrame] = {}

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Theme")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        selector_card = SettingsCard(
            "Theme Selection",
            subtitle="Choose & apply themes",
            icon="T",
        )
        selector_layout = QHBoxLayout()
        selector_layout.setContentsMargins(10, 10, 10, 10)
        selector_layout.setSpacing(8)

        self.theme_selector = QComboBox()
        self.theme_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.theme_selector.currentIndexChanged.connect(self._on_theme_selected)
        selector_layout.addWidget(self.theme_selector, 1)

        self.use_btn = QPushButton("Use")
        self.use_btn.setObjectName("primaryBtn")
        self.use_btn.clicked.connect(self._use_selected_theme)
        selector_layout.addWidget(self.use_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("secondaryBtn")
        self.reset_btn.clicked.connect(self._reset_theme)
        selector_layout.addWidget(self.reset_btn)

        selector_card.add_layout(selector_layout)
        layout.addWidget(selector_card)

        editor_card = SettingsCard(
            "Theme Tokens",
            subtitle="Customize colors & style",
            icon="#",
        )
        editor_layout = QVBoxLayout()

        identity_form = QFormLayout()
        identity_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.theme_id = QLineEdit()
        self.theme_id.textChanged.connect(self._on_draft_changed)
        identity_form.addRow("ID:", self.theme_id)
        self.display_name = QLineEdit()
        self.display_name.textChanged.connect(self._on_draft_changed)
        identity_form.addRow("Name:", self.display_name)
        editor_layout.addLayout(identity_form)

        token_form = QFormLayout()
        token_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for token in THEME_TOKENS:
            row = QHBoxLayout()
            field = QLineEdit()
            field.setMinimumWidth(0)
            field.setPlaceholderText("#RRGGBB")
            field.textChanged.connect(self._on_draft_changed)
            row.addWidget(field, 1)

            swatch = QFrame()
            swatch.setObjectName("themeSwatch")
            swatch.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            row.addWidget(swatch)

            pick_btn = QPushButton("Pick")
            pick_btn.setObjectName("secondaryBtn")
            pick_btn.clicked.connect(lambda _checked=False, key=token: self._pick_color(key))
            row.addWidget(pick_btn)

            token_form.addRow(f"{token}:", row)
            self._token_fields[token] = field
            self._swatches[token] = swatch
        editor_layout.addLayout(token_form)

        option_row = QHBoxLayout()
        self.live_preview = QCheckBox("Live preview")
        self.live_preview.setChecked(True)
        self.live_preview.toggled.connect(self._on_live_preview_toggled)
        option_row.addWidget(self.live_preview)
        option_row.addStretch()

        self.save_btn = QPushButton("Save Custom")
        self.save_btn.setObjectName("primaryBtn")
        self.save_btn.clicked.connect(self._save_custom_theme)
        option_row.addWidget(self.save_btn)

        self.apply_custom_btn = QPushButton("Apply Custom")
        self.apply_custom_btn.setObjectName("primaryBtn")
        self.apply_custom_btn.clicked.connect(self._apply_custom_theme)
        option_row.addWidget(self.apply_custom_btn)

        self.delete_btn = QPushButton("Delete Custom")
        self.delete_btn.setObjectName("secondaryBtn")
        self.delete_btn.clicked.connect(self._delete_custom_theme)
        option_row.addWidget(self.delete_btn)
        editor_layout.addLayout(option_row)

        self.validation_label = QLabel("")
        self.validation_label.setObjectName("metricLabel")
        self.validation_label.setWordWrap(True)
        editor_layout.addWidget(self.validation_label)

        editor_card.add_layout(editor_layout)
        layout.addWidget(editor_card)
        layout.addStretch()
        self.setWidget(container)

        if hasattr(self.event_bus, "ui_theme_changed"):
            self.event_bus.ui_theme_changed.connect(self._on_external_theme_changed)
        self._populate_selector()

    def _populate_selector(self, selected_theme_id: str | None = None) -> None:
        selected_theme_id = selected_theme_id or self.theme_mgr.current_theme
        self._loading = True
        self.theme_selector.clear()
        for theme in self.theme_mgr.available_themes():
            suffix = " (built-in)" if theme.IsBuiltIn else " (custom)"
            self.theme_selector.addItem(f"{theme.DisplayName}{suffix}", theme.ThemeId)
        index = self.theme_selector.findData(selected_theme_id)
        self.theme_selector.setCurrentIndex(max(index, 0))
        self._loading = False
        self._load_selected_theme()

    def _on_theme_selected(self) -> None:
        if not self._loading:
            self._load_selected_theme()

    def _load_selected_theme(self) -> None:
        theme_id = self.theme_selector.currentData()
        if not theme_id:
            return
        theme = self.theme_mgr.get_theme(theme_id)
        self._source_theme_id = theme.ThemeId
        self._loading = True
        if theme.IsBuiltIn:
            self.theme_id.setText(f"{theme.ThemeId}_custom")
            self.display_name.setText(f"{theme.DisplayName} Custom")
        else:
            self.theme_id.setText(theme.ThemeId)
            self.display_name.setText(theme.DisplayName)
        for token, value in theme.tokens().items():
            self._token_fields[token].setText(value)
            self._set_swatch(token, value)
        self.delete_btn.setEnabled(not theme.IsBuiltIn)
        self._loading = False
        self._validate_draft()

    def _use_selected_theme(self) -> None:
        theme_id = str(self.theme_selector.currentData() or "")
        if not theme_id:
            return
        applied = self.theme_mgr.apply_theme(theme_id)
        self._publish_theme_change("apply", applied)
        self._emit_theme_changed(applied)
        self._set_validation("Theme applied.", "success")

    def _reset_theme(self) -> None:
        applied = self.theme_mgr.reset_to_default()
        self._populate_selector(applied)
        self._publish_theme_change("reset", applied)
        self._emit_theme_changed(applied)
        self._set_validation("Theme reset.", "success")

    def _save_custom_theme(self) -> None:
        try:
            draft = self._draft_theme()
            self.theme_mgr.save_custom_theme(draft)
        except ValueError as exc:
            self._set_validation(str(exc), "error")
            return
        self._populate_selector(draft.ThemeId)
        self._publish_theme_change("save", draft.ThemeId)
        self._set_validation("Custom theme saved.", "success")

    def _apply_custom_theme(self) -> None:
        try:
            draft = self._draft_theme()
            self.theme_mgr.save_custom_theme(draft)
            applied = self.theme_mgr.apply_theme(draft.ThemeId)
        except ValueError as exc:
            self._set_validation(str(exc), "error")
            return
        self._populate_selector(applied)
        self._publish_theme_change("apply_custom", applied)
        self._emit_theme_changed(applied)
        self._set_validation("Custom theme applied.", "success")

    def _delete_custom_theme(self) -> None:
        theme_id = str(self.theme_selector.currentData() or "")
        if not theme_id:
            return
        try:
            self.theme_mgr.delete_custom_theme(theme_id)
        except ValueError as exc:
            self._set_validation(str(exc), "error")
            return
        self._populate_selector(self.theme_mgr.current_theme)
        self._publish_theme_change("delete", theme_id)
        self._emit_theme_changed(self.theme_mgr.current_theme)
        self._set_validation("Custom theme deleted.", "success")

    def _pick_color(self, token: str) -> None:
        current = self._token_fields[token].text().strip()
        color = QColorDialog.getColor(QColor(current), self, token)
        if color.isValid():
            self._token_fields[token].setText(color.name())

    def _on_draft_changed(self) -> None:
        if self._loading:
            return
        for token, field in self._token_fields.items():
            self._set_swatch(token, field.text())
        validation = self._validate_draft()
        if validation["valid"] and self.live_preview.isChecked():
            try:
                self.theme_mgr.apply_preview(self._draft_theme())
            except ValueError:
                pass

    def _on_live_preview_toggled(self, enabled: bool) -> None:
        if enabled:
            self._on_draft_changed()
        else:
            self.theme_mgr.apply_theme(self.theme_mgr.current_theme)

    def _on_external_theme_changed(self, theme_id: str) -> None:
        if not theme_id:
            return
        index = self.theme_selector.findData(theme_id)
        if index < 0:
            self._populate_selector(theme_id)
            return
        if index != self.theme_selector.currentIndex():
            self._loading = True
            self.theme_selector.setCurrentIndex(index)
            self._loading = False
            self._load_selected_theme()

    def _validate_draft(self) -> dict:
        try:
            draft = self._draft_theme()
            validation = self.theme_mgr.validate_theme(draft)
        except ValueError as exc:
            validation = {"valid": False, "errors": [str(exc)], "warnings": []}

        if validation["errors"]:
            self._set_validation("; ".join(validation["errors"]), "error")
        elif validation["warnings"]:
            self._set_validation("; ".join(validation["warnings"]), "warning")
        else:
            self._set_validation("Theme draft valid.", "success")
        return validation

    def _draft_theme(self) -> ThemeDefinition:
        theme_id = self._normalize_theme_id(self.theme_id.text())
        display_name = self.display_name.text().strip() or theme_id
        raw = {
            "ThemeId": theme_id,
            "DisplayName": display_name,
            "IsBuiltIn": False,
        }
        for token, field in self._token_fields.items():
            raw[token] = field.text().strip()
        return ThemeDefinition.from_dict(raw)

    def _set_validation(self, text: str, role: str) -> None:
        self.validation_label.setText(text)
        if role:
            apply_status_style(self.validation_label, role=role)
        else:
            clear_status_role(self.validation_label)

    def _set_swatch(self, token: str, color: str) -> None:
        clean_color = color.strip()
        if QColor(clean_color).isValid():
            self._swatches[token].setStyleSheet(f"background-color: {clean_color};")
        else:
            self._swatches[token].setStyleSheet("")

    def _publish_theme_change(self, action: str, theme_id: str) -> None:
        self._sync_client.publish_config_change(
            {
                "scope": "ui.theme",
                "action": action,
                "theme_id": theme_id,
            }
        )
        if hasattr(self.event_bus, "log_entry"):
            self.event_bus.log_entry.emit(
                f"[CoreSync] UI theme {action}: {theme_id}"
            )

    def _emit_theme_changed(self, theme_id: str) -> None:
        if hasattr(self.event_bus, "ui_theme_changed"):
            self.event_bus.ui_theme_changed.emit(theme_id)

    @staticmethod
    def _normalize_theme_id(theme_id: str) -> str:
        return str(theme_id or "").strip().lower().replace(" ", "_")
