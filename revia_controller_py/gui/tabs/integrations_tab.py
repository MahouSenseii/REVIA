"""
REVIA Integrations Tab
Configure and control Discord & Twitch bot integrations.
"""
import logging
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QCheckBox, QPushButton,
    QSpinBox, QTextEdit,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from gui.widgets.settings_card import SettingsCard

logger = logging.getLogger(__name__)


class IntegrationsTab(QScrollArea):
    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.setWidgetResizable(True)
        self._status_refresh_inflight = False

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        # ---------- Header ----------
        header = QLabel("Platform Integrations")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        sub = QLabel(
            "Connect REVIA to Discord and Twitch so it can read and reply in chat."
        )
        sub.setWordWrap(True)
        sub.setObjectName("subLabel")
        layout.addWidget(sub)

        # ---------- Discord ----------
        discord_card = SettingsCard(
            "Discord Bot",
            subtitle="Chat integration",
            icon="D",
        )
        dg = QVBoxLayout()

        self.discord_enabled = QCheckBox("Enable Discord integration")
        dg.addWidget(self.discord_enabled)

        d_form = QFormLayout()
        d_form.setLabelAlignment(Qt.AlignRight)

        self.discord_token = QLineEdit()
        self.discord_token.setPlaceholderText("Bot token from Discord Developer Portal")
        self.discord_token.setEchoMode(QLineEdit.Password)
        d_form.addRow("Bot Token:", self.discord_token)
        self.discord_token_status = QLabel("Token: not configured")
        self.discord_token_status.setObjectName("metricLabel")
        d_form.addRow("", self.discord_token_status)

        self.discord_channels = QLineEdit()
        self.discord_channels.setPlaceholderText("Channel IDs, comma-separated (leave blank for all)")
        d_form.addRow("Channel IDs:", self.discord_channels)

        self.discord_guilds = QLineEdit()
        self.discord_guilds.setPlaceholderText("Server IDs, comma-separated (leave blank for all)")
        d_form.addRow("Server IDs:", self.discord_guilds)

        self.discord_prefix = QLineEdit("!")
        self.discord_prefix.setMaximumWidth(60)
        d_form.addRow("Command Prefix:", self.discord_prefix)

        self.discord_mention_only = QCheckBox("Respond only when @mentioned")
        d_form.addRow("", self.discord_mention_only)

        dg.addLayout(d_form)

        d_btns = QHBoxLayout()
        self.discord_start_btn = QPushButton("Start Bot")
        self.discord_start_btn.setObjectName("primaryBtn")
        self.discord_start_btn.clicked.connect(self._discord_start)
        self.discord_stop_btn = QPushButton("Stop Bot")
        self.discord_stop_btn.clicked.connect(self._discord_stop)
        self.discord_status_lbl = QLabel("Status: stopped")
        self.discord_status_lbl.setObjectName("statusLabel")
        d_btns.addWidget(self.discord_start_btn)
        d_btns.addWidget(self.discord_stop_btn)
        d_btns.addWidget(self.discord_status_lbl, stretch=1)
        dg.addLayout(d_btns)

        self.discord_msgs_lbl = QLabel("Messages processed: 0")
        dg.addWidget(self.discord_msgs_lbl)

        discord_card.add_layout(dg)
        layout.addWidget(discord_card)

        # ---------- Twitch ----------
        twitch_card = SettingsCard(
            "Twitch Chat Bot",
            subtitle="Live stream integration",
            icon="T",
        )
        tg = QVBoxLayout()

        self.twitch_enabled = QCheckBox("Enable Twitch integration")
        tg.addWidget(self.twitch_enabled)

        t_form = QFormLayout()
        t_form.setLabelAlignment(Qt.AlignRight)

        self.twitch_token = QLineEdit()
        self.twitch_token.setPlaceholderText("oauth:xxxxxxxxxxxxxxxxxxxxxxxx")
        self.twitch_token.setEchoMode(QLineEdit.Password)
        t_form.addRow("OAuth Token:", self.twitch_token)
        self.twitch_token_status = QLabel("Token: not configured")
        self.twitch_token_status.setObjectName("metricLabel")
        t_form.addRow("", self.twitch_token_status)

        self.twitch_channels = QLineEdit()
        self.twitch_channels.setPlaceholderText("channel1, channel2 (lowercase names)")
        t_form.addRow("Channels:", self.twitch_channels)

        self.twitch_prefix = QLineEdit("!")
        self.twitch_prefix.setMaximumWidth(60)
        t_form.addRow("Command Prefix:", self.twitch_prefix)

        self.twitch_command = QLineEdit("revia")
        self.twitch_command.setMaximumWidth(120)
        t_form.addRow("Command Name:", self.twitch_command)

        self.twitch_respond_all = QCheckBox("Respond to every chat message (not just commands)")
        t_form.addRow("", self.twitch_respond_all)

        self.twitch_max_len = QSpinBox()
        self.twitch_max_len.setRange(50, 500)
        self.twitch_max_len.setValue(450)
        t_form.addRow("Max Response Length:", self.twitch_max_len)

        tg.addLayout(t_form)

        t_btns = QHBoxLayout()
        self.twitch_start_btn = QPushButton("Start Bot")
        self.twitch_start_btn.setObjectName("primaryBtn")
        self.twitch_start_btn.clicked.connect(self._twitch_start)
        self.twitch_stop_btn = QPushButton("Stop Bot")
        self.twitch_stop_btn.clicked.connect(self._twitch_stop)
        self.twitch_status_lbl = QLabel("Status: stopped")
        self.twitch_status_lbl.setObjectName("statusLabel")
        t_btns.addWidget(self.twitch_start_btn)
        t_btns.addWidget(self.twitch_stop_btn)
        t_btns.addWidget(self.twitch_status_lbl, stretch=1)
        tg.addLayout(t_btns)

        self.twitch_msgs_lbl = QLabel("Messages processed: 0")
        tg.addWidget(self.twitch_msgs_lbl)

        twitch_card.add_layout(tg)
        layout.addWidget(twitch_card)

        # ---------- Save button ----------
        save_btn = QPushButton("Save Configuration")
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._save_config)
        layout.addWidget(save_btn)

        layout.addStretch()
        self.setWidget(container)

        # Poll status every 5 s
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._refresh_status)
        self._poll_timer.start(5000)

        self._load_config()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_comma_list(text: str) -> list:
        return [item.strip() for item in text.split(",") if item.strip()]

    @staticmethod
    def _format_secret_status(secret_name: str, data: dict) -> str:
        if not isinstance(data, dict):
            return f"{secret_name}: not configured"
        if data.get(f"{secret_name}_set"):
            source = data.get(f"{secret_name}_source", "saved")
            preview = data.get(f"{secret_name}_preview", "")
            detail = f" ({preview})" if preview else ""
            return f"{secret_name}: configured via {source}{detail}"
        return f"{secret_name}: not configured"

    def _build_discord_config(self) -> dict:
        return {
            "enabled": self.discord_enabled.isChecked(),
            "bot_token": self.discord_token.text().strip(),
            "channel_ids": self._parse_comma_list(self.discord_channels.text()),
            "guild_ids": self._parse_comma_list(self.discord_guilds.text()),
            "prefix": self.discord_prefix.text().strip() or "!",
            "mention_only": self.discord_mention_only.isChecked(),
        }

    def _build_twitch_config(self) -> dict:
        return {
            "enabled": self.twitch_enabled.isChecked(),
            "oauth_token": self.twitch_token.text().strip(),
            "channels": self._parse_comma_list(self.twitch_channels.text()),
            "prefix": self.twitch_prefix.text().strip() or "!",
            "command": self.twitch_command.text().strip() or "revia",
            "respond_to_all": self.twitch_respond_all.isChecked(),
            "max_response_len": self.twitch_max_len.value(),
        }

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _save_config(self):
        self._save_config_async()

    def _save_config_async(self, on_complete=None):
        payload = {
            "discord": self._build_discord_config(),
            "twitch": self._build_twitch_config(),
        }
        self.client.post_async(
            "/api/integrations/config",
            json=payload,
            timeout=5,
            default={},
            on_success=lambda _data: (
                self._load_config(),
                on_complete() if callable(on_complete) else None,
            ),
            on_error=lambda error, _detail=None: logger.error(
                "Error saving integration config: %s", error
            ),
        )

    def _load_config(self):
        self.client.get_async(
            "/api/integrations/config",
            timeout=3,
            default={},
            on_success=self._apply_config,
            on_error=lambda error, _detail=None: logger.warning(
                "Error loading integration config: %s", error
            ),
        )

    def _apply_config(self, cfg):
        if not cfg:
            return
        d = cfg.get("discord", {})
        t = cfg.get("twitch", {})

        self.discord_enabled.setChecked(d.get("enabled", False))
        self.discord_token.clear()
        self.discord_token.setPlaceholderText(
            "Bot token from Discord Developer Portal"
        )
        self.discord_token_status.setText(
            self._format_secret_status("bot_token", d)
        )
        self.discord_channels.setText(
            ", ".join(str(c) for c in d.get("channel_ids", []))
        )
        self.discord_guilds.setText(
            ", ".join(str(g) for g in d.get("guild_ids", []))
        )
        self.discord_prefix.setText(d.get("prefix", "!"))
        self.discord_mention_only.setChecked(d.get("mention_only", False))

        self.twitch_enabled.setChecked(t.get("enabled", False))
        self.twitch_token.clear()
        self.twitch_token.setPlaceholderText("oauth:xxxxxxxxxxxxxxxxxxxxxxxx")
        self.twitch_token_status.setText(
            self._format_secret_status("oauth_token", t)
        )
        self.twitch_channels.setText(", ".join(t.get("channels", [])))
        self.twitch_prefix.setText(t.get("prefix", "!"))
        self.twitch_command.setText(t.get("command", "revia"))
        self.twitch_respond_all.setChecked(t.get("respond_to_all", False))
        self.twitch_max_len.setValue(t.get("max_response_len", 450))

    def _discord_start(self):
        self._save_config_async(
            on_complete=lambda: self.client.post_async(
                "/api/integrations/discord/start",
                default={},
                on_success=lambda _data: self._refresh_status(),
                on_error=lambda error, _detail=None: logger.error(
                    "Error starting Discord bot: %s", error
                ),
            )
        )

    def _discord_stop(self):
        self.client.post_async(
            "/api/integrations/discord/stop",
            default={},
            on_success=lambda _data: self._refresh_status(),
            on_error=lambda error, _detail=None: logger.error(
                "Error stopping Discord bot: %s", error
            ),
        )

    def _twitch_start(self):
        self._save_config_async(
            on_complete=lambda: self.client.post_async(
                "/api/integrations/twitch/start",
                default={},
                on_success=lambda _data: self._refresh_status(),
                on_error=lambda error, _detail=None: logger.error(
                    "Error starting Twitch bot: %s", error
                ),
            )
        )

    def _twitch_stop(self):
        self.client.post_async(
            "/api/integrations/twitch/stop",
            default={},
            on_success=lambda _data: self._refresh_status(),
            on_error=lambda error, _detail=None: logger.error(
                "Error stopping Twitch bot: %s", error
            ),
        )

    def _refresh_status(self):
        if self._status_refresh_inflight:
            return
        self._status_refresh_inflight = True
        self.client.get_async(
            "/api/integrations/status",
            timeout=3,
            default={},
            on_success=self._apply_status,
            on_error=self._on_refresh_status_error,
        )

    def _on_refresh_status_error(self, error, _detail=None):
        self._status_refresh_inflight = False
        logger.warning("Error refreshing integration status: %s", error)

    def _apply_status(self, status):
        self._status_refresh_inflight = False
        if not status:
            return
        d = status.get("discord", {})
        t = status.get("twitch", {})

        self.discord_status_lbl.setText(f"Status: {d.get('status', 'stopped')}")
        self.discord_msgs_lbl.setText(
            f"Messages processed: {d.get('messages_processed', 0)}"
        )
        if d.get("last_error"):
            self.discord_status_lbl.setText(
                    f"Status: error — {d['last_error'][:60]}"
            )

        self.twitch_status_lbl.setText(f"Status: {t.get('status', 'stopped')}")
        self.twitch_msgs_lbl.setText(
            f"Messages processed: {t.get('messages_processed', 0)}"
        )
        if t.get("last_error"):
            self.twitch_status_lbl.setText(
                    f"Status: error — {t['last_error'][:60]}"
            )
