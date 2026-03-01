import json as json_mod
from pathlib import Path

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QPushButton, QFileDialog, QInputDialog, QMessageBox,
)
from PySide6.QtGui import QFont

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"


class ProfileTab(QScrollArea):
    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.setObjectName("profileTab")
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Profile Settings")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        # Character Identity
        identity_group = QGroupBox("Character Identity")
        identity_group.setObjectName("settingsGroup")
        ig = QFormLayout(identity_group)

        self.char_name = QLineEdit("Revia")
        ig.addRow("Character Name:", self.char_name)

        self.persona = QTextEdit()
        self.persona.setMaximumHeight(80)
        self.persona.setPlaceholderText("Describe the character's persona...")
        ig.addRow("Persona:", self.persona)

        self.traits = QLineEdit()
        self.traits.setPlaceholderText("e.g. friendly, curious, witty")
        ig.addRow("Traits:", self.traits)

        layout.addWidget(identity_group)

        # Voice
        voice_group = QGroupBox("Voice")
        voice_group.setObjectName("settingsGroup")
        vg = QFormLayout(voice_group)

        voice_row = QHBoxLayout()
        self.voice_path = QLineEdit()
        self.voice_path.setPlaceholderText("Path to voice model...")
        voice_browse = QPushButton("Browse")
        voice_browse.setObjectName("browseBtn")
        voice_browse.clicked.connect(self._browse_voice)
        voice_row.addWidget(self.voice_path)
        voice_row.addWidget(voice_browse)
        vg.addRow("Voice Path:", voice_row)

        self.voice_tone = QComboBox()
        self.voice_tone.addItems(
            ["Warm", "Neutral", "Energetic", "Calm", "Serious"]
        )
        vg.addRow("Voice Tone:", self.voice_tone)

        self.language = QComboBox()
        self.language.addItems(
            ["English", "Japanese", "Spanish", "French", "German", "Chinese"]
        )
        vg.addRow("Language:", self.language)

        layout.addWidget(voice_group)

        # Behavior
        behavior_group = QGroupBox("Behavior")
        behavior_group.setObjectName("settingsGroup")
        bg = QFormLayout(behavior_group)

        self.response_style = QComboBox()
        self.response_style.addItems(
            ["Conversational", "Concise", "Detailed", "Creative", "Technical"]
        )
        bg.addRow("Response Style:", self.response_style)

        self.verbosity = QComboBox()
        self.verbosity.addItems(["Minimal", "Normal", "Verbose"])
        self.verbosity.setCurrentIndex(1)
        bg.addRow("Verbosity:", self.verbosity)

        self.fallback_msg = QLineEdit(
            "Uh... something's wrong. Someone tell my operator he messed up."
        )
        bg.addRow("Fallback Msg:", self.fallback_msg)

        self.greeting = QLineEdit(
            "Hello! I'm Revia, your neural assistant."
        )
        bg.addRow("Greeting:", self.greeting)

        layout.addWidget(behavior_group)

        # Character Prompt
        prompt_group = QGroupBox("Character Prompt")
        prompt_group.setObjectName("settingsGroup")
        pg = QVBoxLayout(prompt_group)

        self.char_prompt = QTextEdit()
        self.char_prompt.setMinimumHeight(100)
        self.char_prompt.setPlaceholderText(
            "Enter the character system prompt..."
        )
        pg.addWidget(self.char_prompt)

        layout.addWidget(prompt_group)

        # Save row
        save_row = QHBoxLayout()
        self.profile_name = QLineEdit("default")
        self.profile_name.setPlaceholderText("Profile name...")
        save_row.addWidget(self.profile_name, stretch=1)

        save_btn = QPushButton("Save Profile")
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._save)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)

        # Load row
        load_row = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(160)
        self._refresh_profile_list()
        load_row.addWidget(self.profile_combo, stretch=1)

        load_btn = QPushButton("Load Profile")
        load_btn.setObjectName("secondaryBtn")
        load_btn.clicked.connect(self._load)
        load_row.addWidget(load_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("secondaryBtn")
        refresh_btn.clicked.connect(self._refresh_profile_list)
        load_row.addWidget(refresh_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("secondaryBtn")
        delete_btn.clicked.connect(self._delete_profile)
        load_row.addWidget(delete_btn)
        layout.addLayout(load_row)

        # Export / Import row
        ei_row = QHBoxLayout()
        export_btn = QPushButton("Export JSON")
        export_btn.setObjectName("secondaryBtn")
        export_btn.clicked.connect(self._export)
        ei_row.addWidget(export_btn)

        import_btn = QPushButton("Import JSON")
        import_btn.setObjectName("secondaryBtn")
        import_btn.clicked.connect(self._import)
        ei_row.addWidget(import_btn)
        layout.addLayout(ei_row)

        # Status
        self.profile_status = QLabel("")
        self.profile_status.setFont(QFont("Consolas", 8))
        self.profile_status.setObjectName("metricLabel")
        self.profile_status.setWordWrap(True)
        layout.addWidget(self.profile_status)

        layout.addStretch()
        self.setWidget(container)

    # --- Helpers ---

    def _ensure_profiles_dir(self):
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    def _refresh_profile_list(self):
        self._ensure_profiles_dir()
        self.profile_combo.clear()
        for f in sorted(PROFILES_DIR.glob("*.json")):
            self.profile_combo.addItem(f.stem)

    def _collect(self):
        return {
            "character_name": self.char_name.text(),
            "persona": self.persona.toPlainText(),
            "traits": self.traits.text(),
            "voice_path": self.voice_path.text(),
            "voice_tone": self.voice_tone.currentText(),
            "language": self.language.currentText(),
            "response_style": self.response_style.currentText(),
            "verbosity": self.verbosity.currentText(),
            "fallback_msg": self.fallback_msg.text(),
            "greeting": self.greeting.text(),
            "character_prompt": self.char_prompt.toPlainText(),
        }

    def _apply(self, data):
        self.char_name.setText(data.get("character_name", ""))
        self.persona.setPlainText(data.get("persona", ""))
        self.traits.setText(data.get("traits", ""))
        self.voice_path.setText(data.get("voice_path", ""))

        idx = self.voice_tone.findText(data.get("voice_tone", ""))
        if idx >= 0:
            self.voice_tone.setCurrentIndex(idx)
        idx = self.language.findText(data.get("language", ""))
        if idx >= 0:
            self.language.setCurrentIndex(idx)
        idx = self.response_style.findText(data.get("response_style", ""))
        if idx >= 0:
            self.response_style.setCurrentIndex(idx)
        idx = self.verbosity.findText(data.get("verbosity", ""))
        if idx >= 0:
            self.verbosity.setCurrentIndex(idx)

        self.fallback_msg.setText(data.get("fallback_msg", ""))
        self.greeting.setText(data.get("greeting", ""))
        self.char_prompt.setPlainText(data.get("character_prompt", ""))

    # --- Actions ---

    def _save(self):
        self._ensure_profiles_dir()
        name = self.profile_name.text().strip()
        if not name:
            self.profile_status.setText("Enter a profile name first.")
            self.profile_status.setStyleSheet("color: #cc3040;")
            return

        path = PROFILES_DIR / f"{name}.json"
        data = self._collect()
        with open(path, "w", encoding="utf-8") as f:
            json_mod.dump(data, f, indent=2, ensure_ascii=False)

        self.client.save_profile(data)
        self._refresh_profile_list()
        self.profile_status.setText(f"Saved: {path.name}")
        self.profile_status.setStyleSheet("color: #00aa40;")

    def _load(self):
        name = self.profile_combo.currentText()
        if not name:
            self.profile_status.setText("No profile selected.")
            self.profile_status.setStyleSheet("color: #cc3040;")
            return

        path = PROFILES_DIR / f"{name}.json"
        if not path.exists():
            self.profile_status.setText(f"File not found: {path.name}")
            self.profile_status.setStyleSheet("color: #cc3040;")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json_mod.load(f)

        self._apply(data)
        self.profile_name.setText(name)
        self.client.save_profile(data)
        self.profile_status.setText(f"Loaded: {name}")
        self.profile_status.setStyleSheet("color: #00aa40;")

    def _delete_profile(self):
        name = self.profile_combo.currentText()
        if not name:
            return
        path = PROFILES_DIR / f"{name}.json"
        if path.exists():
            reply = QMessageBox.question(
                self, "Delete Profile",
                f"Delete profile '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                path.unlink()
                self._refresh_profile_list()
                self.profile_status.setText(f"Deleted: {name}")
                self.profile_status.setStyleSheet("color: #ccaa00;")

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Profile", "profile.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json_mod.dump(self._collect(), f, indent=2, ensure_ascii=False)
            self.profile_status.setText(f"Exported to {Path(path).name}")
            self.profile_status.setStyleSheet("color: #00aa40;")

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Profile", "", "JSON (*.json);;All Files (*)"
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json_mod.load(f)
                self._apply(data)
                self.profile_status.setText(f"Imported: {Path(path).name}")
                self.profile_status.setStyleSheet("color: #00aa40;")
            except Exception as e:
                self.profile_status.setText(f"Import error: {e}")
                self.profile_status.setStyleSheet("color: #cc3040;")

    def _browse_voice(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Model", "", "All Files (*)"
        )
        if path:
            self.voice_path.setText(path)
