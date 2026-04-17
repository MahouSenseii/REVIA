"""
REVIA Discord Integration
Routes Discord messages through the REVIA AI pipeline and replies in-channel.
"""
import asyncio
import threading
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger("revia.discord")

try:
    from ipc import get_event_publisher
except Exception:
    try:
        from revia_core_py.ipc import get_event_publisher
    except Exception:
        get_event_publisher = None

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
        # Dedicated thread pool - keeps Discord I/O and LLM work off the default
        # executor so other asyncio tasks are never starved.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="revia-discord-llm")
        self.running = False
        self.status = "stopped"
        self.last_error = None
        self.messages_processed = 0
        self._counter_lock = threading.Lock()
        self._core_event_publisher = get_event_publisher() if get_event_publisher else None
        # Sing command handler (set externally by IntegrationManager)
        self.sing_command_handler = None
        # Pre-convert ID lists to sets of strings for O(1) per-message lookup
        self._allowed_guilds: set[str] = {str(g) for g in config.get("guild_ids", [])}
        self._allowed_channels: set[str] = {str(c) for c in config.get("channel_ids", [])}

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

            # Ignore DMs - guild is None for direct messages
            if message.guild is None:
                return

            # Guild filter (sets pre-built at init for O(1) lookup)
            if bot_ref._allowed_guilds and str(message.guild.id) not in bot_ref._allowed_guilds:
                return

            # Channel filter
            if bot_ref._allowed_channels and str(message.channel.id) not in bot_ref._allowed_channels:
                return

            content = message.content.strip()
            mention_only = cfg.get("mention_only", False)
            prefix = cfg.get("prefix", "!")

            # Handle !sing command before general routing
            sing_prefix = f"{prefix}sing"
            if content.lower().startswith(sing_prefix):
                handler = bot_ref.sing_command_handler
                if handler:
                    args = content[len(sing_prefix):].strip()
                    username = message.author.display_name
                    async with message.channel.typing():
                        loop = asyncio.get_running_loop()
                        try:
                            reply = await loop.run_in_executor(
                                bot_ref._executor, handler.handle, args, username
                            )
                            if reply:
                                chunks = _split_message(reply, 1900)
                                for chunk in chunks:
                                    await message.channel.send(chunk)
                        except Exception as exc:
                            logger.error("[Discord] !sing error: %s", exc)
                            await message.channel.send("Something went wrong with !sing.")
                else:
                    await message.channel.send("Sing mode is not enabled yet!")
                return

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
            bot_ref._publish_user_text_event(message, content, username)

            # Natural typing delay - makes responses feel less instant/robotic (configurable, reduced default)
            delay_range = cfg.get("typing_delay_ms", [100, 400])  # Reduced from [600, 1800]
            if isinstance(delay_range, list) and len(delay_range) == 2:
                delay_s = random.uniform(delay_range[0], delay_range[1]) / 1000
            else:
                delay_s = 0.0

            async with message.channel.typing():
                if delay_s > 0:
                    await asyncio.sleep(delay_s)

                loop = asyncio.get_running_loop()
                try:
                    response = await loop.run_in_executor(
                        bot_ref._executor, bot_ref.pipeline_fn, platform_context
                    )
                    with bot_ref._counter_lock:
                        bot_ref.messages_processed += 1
                    if response:
                        try:
                            chunks = _split_message(response, 1900)
                            if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
                                raise ValueError("Invalid response format from _split_message")
                            for chunk in chunks:
                                await message.channel.send(chunk)
                        except Exception as exc:
                            logger.error(f"[Discord] Message splitting error: {exc}")
                            bot_ref.last_error = str(exc)
                except Exception as exc:
                    logger.error(f"[Discord] Pipeline error: {exc}")
                    bot_ref.last_error = str(exc)

        return client

    def _publish_user_text_event(self, message, content: str, username: str) -> None:
        """Fire-and-forget Phase 1 bridge into the cpp EventBus.

        The bridge is non-authoritative: Discord still calls the legacy
        pipeline for the actual response until Core owns action selection.
        """
        publisher = self._core_event_publisher
        if publisher is None:
            return

        author = getattr(message, "author", None)
        channel = getattr(message, "channel", None)
        guild = getattr(message, "guild", None)

        user_id = str(getattr(author, "id", "")) if author else None
        channel_id = str(getattr(channel, "id", "")) if channel else None
        guild_id = str(getattr(guild, "id", "")) if guild else None

        def _send() -> None:
            publisher.publish_user_text(
                content,
                source="Discord",
                user_id=user_id,
                username=username,
                channel_id=channel_id,
                guild_id=guild_id,
                metadata={"bridge": "discord_bot.phase1"},
            )

        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(self._executor, _send)
        except RuntimeError:
            _send()

    # ------------------------------------------------------------------
    # Voice channel support (placeholder)
    # ------------------------------------------------------------------

    async def join_voice_channel(self, channel_id: int):
        """Join a voice channel for audio interaction.

        TODO: Implement full voice support with:
        - discord.VoiceClient connection
        - Audio sink for STT (pipe to continuous_audio)
        - Audio source for TTS (pipe from tts_backend)
        - Speaker identification for multi-user conversation
        """
        logger.info("Voice channel support not yet implemented (channel: %d)", channel_id)
        return None

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
                if self._loop and not self._loop.is_closed():
                    self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True, name="revia-discord")
        self._thread.start()

    def stop(self):
        if not self.running or not self._client:
            return

        async def _shutdown():
            await self._client.close()

        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)

        self._executor.shutdown(wait=False)
        self.running = False
        self.status = "stopped"

    def get_status(self) -> dict:
        with self._counter_lock:
            count = self.messages_processed
        return {
            "running": self.running,
            "status": self.status,
            "last_error": self.last_error,
            "messages_processed": count,
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
