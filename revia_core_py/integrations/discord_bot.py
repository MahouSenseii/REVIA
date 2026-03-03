"""
REVIA Discord Integration
Routes Discord messages through the REVIA AI pipeline and replies in-channel.
"""
import asyncio
import threading
import logging
from typing import Callable

logger = logging.getLogger("revia.discord")

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("[Discord] discord.py not installed — run: pip install discord.py")


class REVIADiscordBot:
    """Discord bot that routes messages through REVIA's AI pipeline."""

    def __init__(self, pipeline_fn: Callable, config: dict):
        self.pipeline_fn = pipeline_fn
        self.config = config
        self._client = None
        self._thread = None
        self._loop = None
        self.running = False
        self.status = "stopped"
        self.last_error = None
        self.messages_processed = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self):
        if not DISCORD_AVAILABLE:
            raise RuntimeError("discord.py is not installed")

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        bot_ref = self

        @client.event
        async def on_ready():
            bot_ref.status = "connected"
            logger.info(f"[Discord] Logged in as {client.user} (ID: {client.user.id})")

        @client.event
        async def on_message(message):
            # Never reply to ourselves
            if message.author == client.user:
                return

            cfg = bot_ref.config

            # Guild filter
            allowed_guilds = cfg.get("guild_ids", [])
            if allowed_guilds and str(message.guild.id) not in [str(g) for g in allowed_guilds]:
                return

            # Channel filter
            allowed_channels = cfg.get("channel_ids", [])
            if allowed_channels and str(message.channel.id) not in [str(c) for c in allowed_channels]:
                return

            content = message.content.strip()
            mention_only = cfg.get("mention_only", False)
            prefix = cfg.get("prefix", "!")

            # Determine whether to respond
            if mention_only:
                if not client.user.mentioned_in(message):
                    return
                # Strip the mention so the AI doesn't see raw IDs
                content = (
                    content
                    .replace(f"<@{client.user.id}>", "")
                    .replace(f"<@!{client.user.id}>", "")
                    .strip()
                )
            elif prefix and content.startswith(prefix):
                content = content[len(prefix):].strip()
            else:
                return  # Not triggered

            if not content:
                return

            username = message.author.display_name
            platform_context = f"[Discord user {username}]: {content}"

            async with message.channel.typing():
                loop = asyncio.get_event_loop()
                try:
                    response = await loop.run_in_executor(
                        None, bot_ref.pipeline_fn, platform_context
                    )
                    bot_ref.messages_processed += 1
                    if response:
                        for chunk in _split_message(response, 1900):
                            await message.channel.send(chunk)
                except Exception as exc:
                    logger.error(f"[Discord] Pipeline error: {exc}")
                    bot_ref.last_error = str(exc)

        return client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        if self.running:
            return

        token = self.config.get("bot_token", "").strip()
        if not token:
            self.status = "error"
            self.last_error = "No bot token configured"
            logger.error("[Discord] Cannot start — no bot token provided")
            return

        self.running = True
        self.status = "connecting"

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._client = self._build_client()
                self._loop.run_until_complete(self._client.start(token))
            except Exception as exc:
                self.status = "error"
                self.last_error = str(exc)
                logger.error(f"[Discord] Fatal error: {exc}")
            finally:
                self.running = False
                if self.status != "error":
                    self.status = "stopped"

        self._thread = threading.Thread(target=_run, daemon=True, name="revia-discord")
        self._thread.start()

    def stop(self):
        if not self.running or not self._client:
            return

        async def _shutdown():
            await self._client.close()

        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)

        self.running = False
        self.status = "stopped"

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "status": self.status,
            "last_error": self.last_error,
            "messages_processed": self.messages_processed,
            "available": DISCORD_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _split_message(text: str, max_len: int) -> list:
    """Split a long string into chunks at word boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind(" ", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks
