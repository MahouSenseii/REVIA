"""
REVIA Integration Manager
Manages Discord and Twitch bot lifecycle and configuration.
"""
import json
import logging
from pathlib import Path

from .discord_bot import REVIADiscordBot
from .twitch_bot import REVIATwitchBot

logger = logging.getLogger("revia.integrations")

_CONFIG_PATH = Path(__file__).parent.parent.parent / "integrations_config.json"

_DEFAULT_CONFIG: dict = {
    "discord": {
        "enabled": False,
        "bot_token": "",
        "channel_ids": [],
        "guild_ids": [],
        "prefix": "!",
        "mention_only": False,
        # [min_ms, max_ms] random delay before replying — makes responses feel
        # less instant and more human-like.  Set to [0, 0] to disable.
        "typing_delay_ms": [600, 1800],
    },
    "twitch": {
        "enabled": False,
        "oauth_token": "",
        "channels": [],
        "prefix": "!",
        "command": "revia",
        "respond_to_all": False,
        "max_response_len": 450,
        # Seconds a user must wait between responses (prevents spam)
        "user_cooldown_s": 8,
        # Seconds identical questions stay cached (0 = disabled)
        "cache_ttl_s": 300,
    },
}


class IntegrationManager:
    """Top-level manager for all platform integrations."""

    def __init__(self, pipeline_fn):
        """
        Args:
            pipeline_fn: Callable(text: str) -> str
                A function that processes text through the REVIA pipeline
                and returns the AI's response.
        """
        self.pipeline_fn = pipeline_fn
        self._config = _load_config()
        self.discord_bot: REVIADiscordBot | None = None
        self.twitch_bot: REVIATwitchBot | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_enabled(self):
        """Start any integrations that are marked enabled in config."""
        if self._config["discord"].get("enabled"):
            self.start_discord()
        if self._config["twitch"].get("enabled"):
            self.start_twitch()

    def start_discord(self):
        if self.discord_bot and self.discord_bot.running:
            logger.info("[Integrations] Discord bot already running")
            return
        self.discord_bot = REVIADiscordBot(self.pipeline_fn, self._config["discord"])
        self.discord_bot.start()
        logger.info("[Integrations] Discord bot starting…")

    def stop_discord(self):
        if self.discord_bot:
            self.discord_bot.stop()
            logger.info("[Integrations] Discord bot stopped")

    def start_twitch(self):
        if self.twitch_bot and self.twitch_bot.running:
            logger.info("[Integrations] Twitch bot already running")
            return
        self.twitch_bot = REVIATwitchBot(self.pipeline_fn, self._config["twitch"])
        self.twitch_bot.start()
        logger.info("[Integrations] Twitch bot starting…")

    def stop_twitch(self):
        if self.twitch_bot:
            self.twitch_bot.stop()
            logger.info("[Integrations] Twitch bot stopped")

    def stop_all(self):
        self.stop_discord()
        self.stop_twitch()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return {k: dict(v) for k, v in self._config.items()}

    def update_config(self, new_cfg: dict):
        """Merge partial config update and persist to disk."""
        for platform in ("discord", "twitch"):
            if platform in new_cfg and isinstance(new_cfg[platform], dict):
                self._config[platform].update(new_cfg[platform])
        _save_config(self._config)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "discord": (
                self.discord_bot.get_status()
                if self.discord_bot
                else {"running": False, "status": "stopped", "messages_processed": 0,
                      "last_error": None, "available": True}
            ),
            "twitch": (
                self.twitch_bot.get_status()
                if self.twitch_bot
                else {"running": False, "status": "stopped", "messages_processed": 0,
                      "last_error": None, "available": True}
            ),
        }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                data = json.load(f)
            cfg: dict = {}
            for k in ("discord", "twitch"):
                cfg[k] = {**_DEFAULT_CONFIG[k], **data.get(k, {})}
            return cfg
        except Exception as exc:
            logger.warning(f"[Integrations] Could not load config: {exc}; using defaults")
    return {k: dict(v) for k, v in _DEFAULT_CONFIG.items()}


def _save_config(cfg: dict):
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as exc:
        logger.error(f"[Integrations] Failed to save config: {exc}")
