import copy
import json as json_mod
from pathlib import Path

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QFrame,
    QPushButton, QFileDialog, QInputDialog, QMessageBox, QSizePolicy,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import QTimer, Qt

from app.ui_status import apply_status_style
from gui.widgets.settings_card import SettingsCard

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"
PERSONA_PRESETS = [
    ("Custom", "custom"),
    ("Default Revia", "default"),
    ("Diana-Inspired", "diana_inspired"),
    ("Casual", "casual"),
    ("Serious", "serious"),
    ("Empathetic", "empathetic"),
]

VOICE_TONES = [
    "Warm", "Neutral", "Energetic", "Calm", "Serious",
    "Bright-Android", "Soft", "Confident", "Mysterious",
    "Playful", "Stoic", "Whisper", "Authoritative",
    "Cheerful", "Sultry", "Sweet", "Cold", "Sharp",
    "Curious", "Protective", "Dreamy", "Sarcastic",
    "Melancholic", "Excited",
]

RESPONSE_STYLES = [
    "Conversational", "Concise", "Detailed", "Creative", "Technical",
    "Field-Partner", "Storyteller", "Mentor", "Analytical", "Direct",
    "Empathetic", "Playful", "Formal", "Poetic", "Witty",
    "Encouraging", "Socratic", "Blunt",
]

LANGUAGES = [
    "English", "Japanese", "Spanish", "French", "German",
    "Chinese", "Italian", "Portuguese", "Korean", "Russian",
    "Dutch", "Polish", "Arabic", "Hindi",
]

VERBOSITY_LEVELS = ["Minimal", "Concise", "Normal", "Verbose", "Expansive"]


class ProfileTab(QScrollArea):
    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self._loaded_profile_data = {}
        self.setObjectName("profileTab")
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel("Profile Settings")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        hint = QLabel(
            "Configure who Revia is, how she sounds, and how she behaves. "
            "Changes apply once you save the profile."
        )
        hint.setObjectName("cardSubText")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # ----- Character Identity Card -----
        identity_card = SettingsCard(
            "Character Identity",
            subtitle="Name, persona, traits",
            icon="*",
        )
        ig = QFormLayout()
        ig.setLabelAlignment(Qt.AlignRight)
        ig.setSpacing(8)

        self.char_name = QLineEdit("Revia")
        self.char_name.setToolTip("The character's display and self-reference name.")
        ig.addRow("Character Name:", self.char_name)

        self.persona = QTextEdit()
        self.persona.setMaximumHeight(80)
        self.persona.setPlaceholderText(
            "One-paragraph summary of who this character is..."
        )
        self.persona.setToolTip("Short description shown to the model as the persona summary.")
        ig.addRow("Persona Summary:", self.persona)

        self.traits = QLineEdit()
        self.traits.setPlaceholderText("e.g. friendly, curious, witty")
        self.traits.setToolTip("Comma-separated personality traits.")
        ig.addRow("Traits:", self.traits)

        self.persona_preset = QComboBox()
        for label, value in PERSONA_PRESETS:
            self.persona_preset.addItem(label, value)
        self.persona_preset.setToolTip("Preset persona templates. 'Custom' keeps your edits.")
        ig.addRow("Persona Preset:", self.persona_preset)

        identity_card.add_layout(ig)
        layout.addWidget(identity_card)

        # ----- Voice Card -----
        voice_card = SettingsCard(
            "Voice",
            subtitle="Sound, tone, language",
            icon="~",
        )
        vg = QFormLayout()
        vg.setLabelAlignment(Qt.AlignRight)
        vg.setSpacing(8)

        voice_row = QHBoxLayout()
        self.voice_path = QLineEdit()
        self.voice_path.setPlaceholderText("Path to reference voice WAV...")
        self.voice_path.setToolTip("Reference audio used by the TTS clone backend.")
        voice_browse = QPushButton("Browse")
        voice_browse.setObjectName("browseBtn")
        voice_browse.clicked.connect(self._browse_voice)
        voice_row.addWidget(self.voice_path)
        voice_row.addWidget(voice_browse)
        vg.addRow("Voice Path:", voice_row)

        self.voice_tone = QComboBox()
        self.voice_tone.setEditable(True)
        self.voice_tone.addItems(VOICE_TONES)
        self.voice_tone.setToolTip(
            "Vocal coloring. You can also type a custom tone."
        )
        vg.addRow("Voice Tone:", self.voice_tone)

        self.language = QComboBox()
        self.language.addItems(LANGUAGES)
        self.language.setToolTip("Primary spoken language.")
        vg.addRow("Language:", self.language)

        voice_card.add_layout(vg)
        layout.addWidget(voice_card)

        # ----- Behavior Card -----
        behavior_card = SettingsCard(
            "Behavior",
            subtitle="Style, verbosity, fallback",
            icon=">",
        )
        bg = QFormLayout()
        bg.setLabelAlignment(Qt.AlignRight)
        bg.setSpacing(8)

        self.response_style = QComboBox()
        self.response_style.setEditable(True)
        self.response_style.addItems(RESPONSE_STYLES)
        self.response_style.setToolTip(
            "How Revia structures her replies. Custom values are accepted."
        )
        bg.addRow("Response Style:", self.response_style)

        self.verbosity = QComboBox()
        self.verbosity.addItems(VERBOSITY_LEVELS)
        self.verbosity.setCurrentIndex(2)
        self.verbosity.setToolTip("How long replies should be on average.")
        bg.addRow("Verbosity:", self.verbosity)

        self.fallback_msg = QLineEdit(
            "Uh... something's wrong. Someone tell my operator he messed up."
        )
        self.fallback_msg.setToolTip(
            "Used when the model fails or times out. Stay in character."
        )
        bg.addRow("Fallback Msg:", self.fallback_msg)

        # Greeting field removed: Revia now greets dynamically each session.
        greet_hint = QLabel(
            "Greetings rotate automatically. Add optional greeting flavors below "
            "(one per line) to influence the pool, or leave blank for the default mix."
        )
        greet_hint.setObjectName("cardSubText")
        greet_hint.setWordWrap(True)
        bg.addRow("Greeting Pool:", greet_hint)

        self.greeting_variants = QTextEdit()
        self.greeting_variants.setMaximumHeight(80)
        self.greeting_variants.setPlaceholderText(
            "I'm here.\nBack online.\nOkay, let's go.\n..."
        )
        self.greeting_variants.setToolTip(
            "Optional list of greeting flavors. One per line. The system picks one "
            "at random each time Revia greets you."
        )
        bg.addRow("Variants:", self.greeting_variants)

        behavior_card.add_layout(bg)
        layout.addWidget(behavior_card)

        # ----- Character Prompt Card -----
        prompt_card = SettingsCard(
            "Character Prompt",
            subtitle="Identity instruction sent to the model",
            icon="#",
        )
        self.char_prompt = QTextEdit()
        self.char_prompt.setMinimumHeight(120)
        self.char_prompt.setPlaceholderText(
            "Enter the character system prompt..."
        )
        self.char_prompt.setToolTip(
            "Primary system identity sent to the LLM. Long-form is OK; the model "
            "treats this as the spine of the character."
        )
        prompt_card.add_widget(self.char_prompt)
        layout.addWidget(prompt_card)

        # ----- Persona Modules Card -----
        modules_card = SettingsCard(
            "Persona Modules",
            subtitle="Voice guide & collaboration guide",
            icon="^",
        )
        pm = QFormLayout()
        pm.setLabelAlignment(Qt.AlignRight)
        pm.setSpacing(8)

        self.persona_style_prompt = QTextEdit()
        self.persona_style_prompt.setMaximumHeight(80)
        self.persona_style_prompt.setPlaceholderText(
            "How this persona should sound in replies..."
        )
        pm.addRow("Voice Guide:", self.persona_style_prompt)

        self.persona_collab_prompt = QTextEdit()
        self.persona_collab_prompt.setMaximumHeight(80)
        self.persona_collab_prompt.setPlaceholderText(
            "How this persona should collaborate with the user..."
        )
        pm.addRow("Collab Guide:", self.persona_collab_prompt)
        modules_card.add_layout(pm)
        layout.addWidget(modules_card)

        # ----- Profile Management Card -----
        manage_card = SettingsCard(
            "Profile Management",
            subtitle="Save, load, import, export",
            icon="@",
        )

        # Save row
        save_row = QHBoxLayout()
        self.profile_name = QLineEdit("default")
        self.profile_name.setPlaceholderText("Profile name...")
        self.profile_name.setToolTip("The filename (without .json) this profile saves to.")
        save_row.addWidget(self.profile_name, stretch=1)

        save_btn = QPushButton("Save Profile")
        save_btn.setObjectName("primaryBtn")
        save_btn.setToolTip("Save current settings to a profile JSON.")
        save_btn.clicked.connect(self._save)
        save_row.addWidget(save_btn)
        manage_card.add_layout(save_row)

        # Load row
        load_row = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(160)
        self._refresh_profile_list()
        load_row.addWidget(self.profile_combo, stretch=1)

        load_btn = QPushButton("Load")
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
        manage_card.add_layout(load_row)

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

        reset_btn = QPushButton("Reset Form")
        reset_btn.setObjectName("secondaryBtn")
        reset_btn.setToolTip("Reset the form fields to last loaded values.")
        reset_btn.clicked.connect(self._reset_form)
        ei_row.addWidget(reset_btn)

        manage_card.add_layout(ei_row)

        # Status
        self.profile_status = QLabel("")
        self.profile_status.setFont(QFont("Consolas", 8))
        self.profile_status.setObjectName("metricLabel")
        self.profile_status.setWordWrap(True)
        manage_card.add_widget(self.profile_status)

        layout.addWidget(manage_card)

        layout.addStretch()
        self.setWidget(container)

        # Auto-fill profile filename when character name changes
        self.char_name.editingFinished.connect(self._sync_profile_name_default)

        self.event_bus.connection_changed.connect(self._on_core_connection)
        if getattr(self.client, "connected", False):
            QTimer.singleShot(0, lambda: self._on_core_connection(True))

    # --- Helpers ---

    def _ensure_profiles_dir(self):
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    def _refresh_profile_list(self):
        self._ensure_profiles_dir()
        self.profile_combo.clear()
        for f in sorted(PROFILES_DIR.glob("*.json")):
            self.profile_combo.addItem(f.stem)

    def _selected_persona_preset(self):
        value = self.persona_preset.currentData()
        return str(value or "custom")

    def _set_persona_preset(self, preset_name):
        preset = str(preset_name or "custom").strip().lower()
        for idx in range(self.persona_preset.count()):
            if self.persona_preset.itemData(idx) == preset:
                self.persona_preset.setCurrentIndex(idx)
                return
        self.persona_preset.setCurrentIndex(0)

    def _sync_profile_name_default(self):
        current = self.profile_name.text().strip()
        if current and current != "default":
            return
        name = self.char_name.text().strip().lower().replace(" ", "_")
        if name:
            self.profile_name.setText(name)

    def _reset_form(self):
        self._apply(self._loaded_profile_data or {})
        self.profile_status.setText("Form reset to last loaded values.")
        apply_status_style(self.profile_status, "color: #ccaa00;")

    @staticmethod
    def _split_csv(value):
        return [item.strip() for item in str(value or "").split(",") if item.strip()]

    @staticmethod
    def _split_lines(value):
        return [
            line.strip()
            for line in str(value or "").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _module_text(persona_def, module_name):
        if not isinstance(persona_def, dict):
            return ""
        direct = str(persona_def.get(f"{module_name}_prompt", "") or "").strip()
        if direct:
            return direct
        for module in persona_def.get("modules", []) or []:
            if not isinstance(module, dict):
                continue
            name = str(module.get("name", "") or "").strip().lower()
            if name == module_name:
                return str(module.get("content", "") or "").strip()
        return ""

    def _collect(self):
        data = copy.deepcopy(self._loaded_profile_data or {})
        persona_def = copy.deepcopy(data.get("persona_definition", {}) or {})

        greeting_variants = self._split_lines(
            self.greeting_variants.toPlainText()
        )

        persona_def.update({
            "name": self.char_name.text().strip() or "Revia",
            "preset": self._selected_persona_preset(),
            "summary": self.persona.toPlainText().strip(),
            "identity_prompt": self.char_prompt.toPlainText().strip(),
            "style_prompt": self.persona_style_prompt.toPlainText().strip(),
            "collaboration_prompt": self.persona_collab_prompt.toPlainText().strip(),
            "traits": self._split_csv(self.traits.text()),
            "interaction_style": {
                "response_style": self.response_style.currentText().strip(),
                "verbosity": self.verbosity.currentText(),
                "greeting_variants": greeting_variants,
            },
        })

        modules = []
        preserved_modules = []
        for module in persona_def.get("modules", []) or []:
            if not isinstance(module, dict):
                continue
            name = str(module.get("name", "") or "").strip().lower()
            if name in {"identity", "style", "collaboration"}:
                continue
            preserved_modules.append(copy.deepcopy(module))
        if persona_def.get("identity_prompt"):
            modules.append({
                "name": "identity",
                "content": persona_def["identity_prompt"],
            })
        if persona_def.get("style_prompt"):
            modules.append({
                "name": "style",
                "content": persona_def["style_prompt"],
            })
        if persona_def.get("collaboration_prompt"):
            modules.append({
                "name": "collaboration",
                "content": persona_def["collaboration_prompt"],
            })
        persona_def["modules"] = modules + preserved_modules

        data.update({
            "character_name": self.char_name.text(),
            "persona": self.persona.toPlainText(),
            "traits": self.traits.text(),
            "voice_path": self.voice_path.text(),
            "voice_tone": self.voice_tone.currentText().strip(),
            "language": self.language.currentText(),
            "response_style": self.response_style.currentText().strip(),
            "verbosity": self.verbosity.currentText(),
            "fallback_msg": self.fallback_msg.text(),
            "greeting_variants": greeting_variants,
            "character_prompt": self.char_prompt.toPlainText(),
            "persona_preset": self._selected_persona_preset(),
            "persona_definition": persona_def,
        })
        # Keep legacy single-greeting field empty so the runtime randomizer
        # picks from variants/defaults rather than echoing a fixed line.
        data["greeting"] = ""
        return data

    def _apply(self, data):
        self._loaded_profile_data = copy.deepcopy(data or {})
        persona_def = data.get("persona_definition", {}) or {}
        traits_text = data.get("traits", "")
        if isinstance(traits_text, list):
            traits_text = ", ".join(str(item) for item in traits_text if str(item).strip())
        if not str(traits_text or "").strip():
            traits_text = ", ".join(persona_def.get("traits", []) or [])

        self.char_name.setText(
            str(data.get("character_name") or persona_def.get("name") or "")
        )
        self.persona.setPlainText(
            str(data.get("persona") or persona_def.get("summary") or "")
        )
        self.traits.setText(str(traits_text or ""))
        self.voice_path.setText(str(data.get("voice_path", "") or ""))
        self._set_persona_preset(
            data.get("persona_preset") or persona_def.get("preset") or "custom"
        )

        tone = str(data.get("voice_tone", "") or "")
        if tone:
            idx = self.voice_tone.findText(tone)
            if idx >= 0:
                self.voice_tone.setCurrentIndex(idx)
            else:
                self.voice_tone.setEditText(tone)
        idx = self.language.findText(data.get("language", ""))
        if idx >= 0:
            self.language.setCurrentIndex(idx)

        rs = str(
            data.get("response_style")
            or ((persona_def.get("interaction_style") or {}).get("response_style"))
            or ""
        )
        if rs:
            idx = self.response_style.findText(rs)
            if idx >= 0:
                self.response_style.setCurrentIndex(idx)
            else:
                self.response_style.setEditText(rs)
        idx = self.verbosity.findText(
            str(
                data.get("verbosity")
                or ((persona_def.get("interaction_style") or {}).get("verbosity"))
                or ""
            )
        )
        if idx >= 0:
            self.verbosity.setCurrentIndex(idx)

        self.fallback_msg.setText(str(data.get("fallback_msg", "") or ""))

        # Greeting variants (new). Falls back to the legacy single greeting if
        # an older profile is loaded, so users do not lose their text.
        variants = data.get("greeting_variants")
        if not variants:
            variants = (persona_def.get("interaction_style") or {}).get(
                "greeting_variants"
            )
        if not variants:
            legacy = (
                data.get("greeting")
                or (persona_def.get("interaction_style") or {}).get("greeting")
                or ""
            )
            variants = [legacy] if legacy else []
        if isinstance(variants, str):
            variants = [variants]
        self.greeting_variants.setPlainText(
            "\n".join(str(v).strip() for v in variants if str(v).strip())
        )

        self.char_prompt.setPlainText(
            str(data.get("character_prompt") or persona_def.get("identity_prompt") or "")
        )
        self.persona_style_prompt.setPlainText(
            self._module_text(persona_def, "style")
        )
        self.persona_collab_prompt.setPlainText(
            self._module_text(persona_def, "collaboration")
        )

    # --- Actions ---

    def _save(self):
        self._ensure_profiles_dir()
        name = self.profile_name.text().strip()
        if not name:
            self.profile_status.setText("Enter a profile name first.")
            apply_status_style(self.profile_status, "color: #cc3040;")
            return

        path = PROFILES_DIR / f"{name}.json"
        data = self._collect()
        with open(path, "w", encoding="utf-8") as f:
            json_mod.dump(data, f, indent=2, ensure_ascii=False)

        self.client.save_profile(data)
        self._refresh_profile_list()
        self.profile_status.setText(f"Saved: {path.name}")
        apply_status_style(self.profile_status, "color: #00aa40;")

    def _load(self):
        name = self.profile_combo.currentText()
        if not name:
            self.profile_status.setText("No profile selected.")
            apply_status_style(self.profile_status, "color: #cc3040;")
            return

        path = PROFILES_DIR / f"{name}.json"
        if not path.exists():
            self.profile_status.setText(f"File not found: {path.name}")
            apply_status_style(self.profile_status, "color: #cc3040;")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json_mod.load(f)

        self._apply(data)
        self.profile_name.setText(name)
        self.client.save_profile(data)
        self.profile_status.setText(f"Loaded: {name}")
        apply_status_style(self.profile_status, "color: #00aa40;")

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
                apply_status_style(self.profile_status, "color: #ccaa00;")

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Profile", "profile.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json_mod.dump(self._collect(), f, indent=2, ensure_ascii=False)
            self.profile_status.setText(f"Exported to {Path(path).name}")
            apply_status_style(self.profile_status, "color: #00aa40;")

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
                apply_status_style(self.profile_status, "color: #00aa40;")
            except Exception as e:
                self.profile_status.setText(f"Import error: {e}")
                apply_status_style(self.profile_status, "color: #cc3040;")

    def _browse_voice(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Model", "", "All Files (*)"
        )
        if path:
            self.voice_path.setText(path)

    def _on_core_connection(self, connected):
        if not connected:
            return
        self._load_from_core()

    def _load_from_core(self):
        self.client.get_async(
            "/api/profile",
            timeout=2,
            default={},
            on_success=self._apply_profile_from_core,
        )

    def _apply_profile_from_core(self, data):
        if not data:
            return
        self._apply(data)
        name = data.get("character_name", "default").strip() or "default"
        self.profile_name.setText(name)
        self.profile_status.setText("Loaded profile from core.")
        apply_status_style(self.profile_status, "color: #00aa40;")
