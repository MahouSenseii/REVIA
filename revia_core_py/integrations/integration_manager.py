"""
REVIA Integration Manager
Manages Discord and Twitch bot lifecycle and configuration.
"""
import json
import logging
import os
from pathlib import Path

from .discord_bot import REVIADiscordBot
from .twitch_bot import REVIATwitchBot

logger = logging.getLogger("revia.integrations")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "data" / "integrations_config.json"
_LEGACY_CONFIG_PATH = Path(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            os.pardir,
            "integrations_config.json",
        )
    )
)
_SECRET_ENV_KEYS = {
    "discord": {"bot_token": "REVIA_DISCORD_BOT_TOKEN"},
    "twitch": {"oauth_token": "REVIA_TWITCH_OAUTH_TOKEN"},
}

_DEFAULT_BOT_STATUS: dict = {
    "running": False, "status": "stopped",
    "messages_processed": 0, "last_error": None, "available": True,
}

_DEFAULT_CONFIG: dict = {
    "discord": {
        "enabled": False,
        "bot_token": "",
        "channel_ids": [],
        "guild_ids": [],
        "prefix": "!",
        "mention_only": False,
        # [min_ms, max_ms] random delay before replying - makes responses feel
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
        self._sing_command_handler = None

    def _effective_config(self) -> dict:
        return _apply_env_secrets(self._config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_enabled(self):
        """Start any integrations that are marked enabled in config."""
        effective = self._effective_config()
        if effective["discord"].get("enabled"):
            self.start_discord()
        if effective["twitch"].get("enabled"):
            self.start_twitch()

    def set_sing_command_handler(self, handler):
        """Wire the sing command handler so !sing works in chat."""
        self._sing_command_handler = handler
        if self.discord_bot:
            self.discord_bot.sing_command_handler = handler
        if self.twitch_bot:
            self.twitch_bot.sing_command_handler = handler

    def start_discord(self):
        if self.discord_bot and self.discord_bot.running:
            logger.info("[Integrations] Discord bot already running")
            return
        effective = self._effective_config()
        discord_cfg = effective["discord"]

        # Validate required config before instantiating the bot so errors
        # surface immediately rather than inside the daemon thread.
        errors = _validate_discord_config(discord_cfg)
        if errors:
            for err in errors:
                logger.error("[Integrations] Discord config error: %s", err)
            return

        self.discord_bot = REVIADiscordBot(self.pipeline_fn, discord_cfg)
        if self._sing_command_handler:
            self.discord_bot.sing_command_handler = self._sing_command_handler
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
        effective = self._effective_config()
        twitch_cfg = effective["twitch"]

        # Validate required config before instantiating the bot.
        errors = _validate_twitch_config(twitch_cfg)
        if errors:
            for err in errors:
                logger.error("[Integrations] Twitch config error: %s", err)
            return

        self.twitch_bot = REVIATwitchBot(self.pipeline_fn, twitch_cfg)
        if self._sing_command_handler:
            self.twitch_bot.sing_command_handler = self._sing_command_handler
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
        return _public_config(self._effective_config())

    def update_config(self, new_cfg: dict):
        """Merge partial config update and persist to disk."""
        for platform in ("discord", "twitch"):
            if platform in new_cfg and isinstance(new_cfg[platform], dict):
                for key, value in new_cfg[platform].items():
                    if key in _SECRET_ENV_KEYS.get(platform, {}) and not str(value).strip():
                        continue
                    self._config[platform][key] = value
        _save_config(self._config)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "discord": self.discord_bot.get_status() if self.discord_bot else dict(_DEFAULT_BOT_STATUS),
            "twitch": self.twitch_bot.get_status() if self.twitch_bot else dict(_DEFAULT_BOT_STATUS),
        }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    for path in (_CONFIG_PATH, _LEGACY_CONFIG_PATH):
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            cfg: dict = {}
            for k in ("discord", "twitch"):
                cfg[k] = {**_DEFAULT_CONFIG[k], **data.get(k, {})}
            if path == _LEGACY_CONFIG_PATH and not _CONFIG_PATH.exists():
                _save_config(cfg)
            return cfg
        except json.JSONDecodeError as exc:
            logger.warning(f"[Integrations] Invalid JSON in config: {exc}; using defaults")
        except IOError as exc:
            logger.warning(f"[Integrations] Failed to read config file: {exc}; using defaults")
    return {k: dict(v) for k, v in _DEFAULT_CONFIG.items()}


def _save_config(cfg: dict):
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except PermissionError as exc:
        logger.error(f"[Integrations] Permission denied saving config: {exc}")
    except OSError as exc:
        logger.error(f"[Integrations] OS error saving config: {exc}")


def _mask_secret(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= 8:
        return raw[:2] + "..." if len(raw) > 2 else "***"
    return raw[:4] + "..." + raw[-4:]


def _apply_env_secrets(cfg: dict) -> dict:
    effective = {k: dict(v) for k, v in cfg.items()}
    for platform, mapping in _SECRET_ENV_KEYS.items():
        target = effective.setdefault(platform, {})
        for field, env_name in mapping.items():
            env_value = os.environ.get(env_name, "").strip()
            if env_value:
                target[field] = env_value
    return effective


def _validate_discord_config(cfg: dict) -> list[str]:
    """Return a list of human-readable error strings, or [] if config is valid."""
    errors: list[str] = []
    token = str(cfg.get("bot_token", "") or "").strip()
    if not token:
        errors.append(
            "bot_token is missing — set it in integrations_config.json "
            "or via the REVIA_DISCORD_BOT_TOKEN environment variable"
        )
    # Channel IDs and guild IDs are optional — bots can run without them
    # (they'll respond in all guilds/channels they have access to).
    return errors


def _validate_twitch_config(cfg: dict) -> list[str]:
    """Return a list of human-readable error strings, or [] if config is valid."""
    errors: list[str] = []
    token = str(cfg.get("oauth_token", "") or "").strip()
    if not token:
        errors.append(
            "oauth_token is missing — set it in integrations_config.json "
            "or via the REVIA_TWITCH_OAUTH_TOKEN environment variable"
        )
    channels = cfg.get("channels", [])
    if not channels:
        errors.append(
            "channels list is empty — add at least one Twitch channel name to join"
        )
    return errors


def _public_config(cfg: dict) -> dict:
    public_cfg = {k: dict(v) for k, v in cfg.items()}
    for platform, mapping in _SECRET_ENV_KEYS.items():
        target = public_cfg.setdefault(platform, {})
        for field, env_name in mapping.items():
            raw = str(target.get(field, "")).strip()
            target[field] = ""
            target[f"{field}_set"] = bool(raw)
            target[f"{field}_source"] = (
                "environment"
                if os.environ.get(env_name, "").strip()
                else ("file" if raw else "")
            )
            target[f"{field}_preview"] = _mask_secret(raw)
    return public_cfg
