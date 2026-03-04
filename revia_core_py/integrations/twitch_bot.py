"""
REVIA Twitch Integration
Reads Twitch chat and responds via the REVIA AI pipeline.
"""
import asyncio
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger("revia.twitch")

try:
    from twitchio.ext import commands as twitchio_commands
    import twitchio  # noqa: F401
    TWITCHIO_AVAILABLE = True
except ImportError:
    TWITCHIO_AVAILABLE = False
    logger.warning("[Twitch] twitchio not installed — run: pip install twitchio")


class REVIATwitchBot:
    """Twitch IRC bot that routes messages through REVIA's AI pipeline."""

    def __init__(self, pipeline_fn: Callable, config: dict):
        self.pipeline_fn = pipeline_fn
        self.config = config
        self._bot = None
        self._thread = None
        self._loop = None
        # Dedicated thread pool — keeps Twitch I/O off the default asyncio executor
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="revia-twitch-llm")
        self.running = False
        self.status = "stopped"
        self.last_error = None
        self.messages_processed = 0

        # Per-user cooldown: {username: last_response_timestamp}
        self._cooldowns: dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        # Simple response cache: {message_text: (response, timestamp)}
        # Avoids re-calling the LLM for identical questions (common in Twitch chat)
        self._cache: dict[str, tuple[str, float]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Cooldown helpers
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, username: str) -> bool:
        cooldown_s = self.config.get("user_cooldown_s", 8)
        if cooldown_s <= 0:
            return False
        with self._cooldown_lock:
            last = self._cooldowns.get(username, 0.0)
            return (time.monotonic() - last) < cooldown_s

    def _record_response(self, username: str):
        with self._cooldown_lock:
            self._cooldowns[username] = time.monotonic()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: str):
        ttl = self.config.get("cache_ttl_s", 300)
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry:
                response, ts = entry
                if time.monotonic() - ts < ttl:
                    return response
                del self._cache[key]
        return None

    def _cache_set(self, key: str, value: str):
        with self._cache_lock:
            # Evict oldest entry if cache exceeds 100 items
            if len(self._cache) >= 100:
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (value, time.monotonic())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_bot(self):
        if not TWITCHIO_AVAILABLE:
            raise RuntimeError("twitchio is not installed")

        cfg = self.config
        token = cfg.get("oauth_token", "")
        channels = cfg.get("channels", [])
        prefix = cfg.get("prefix", "!")
        command_name = cfg.get("command", "revia")
        respond_to_all = cfg.get("respond_to_all", False)
        max_len = cfg.get("max_response_len", 450)
        parent = self

        class _Bot(twitchio_commands.Bot):
            def __init__(self):
                super().__init__(
                    token=token,
                    prefix=prefix,
                    initial_channels=channels,
                )

            async def event_ready(self):
                parent.status = "connected"
                logger.info(f"[Twitch] Connected as {self.nick}")

            async def event_message(self, message):
                if message.echo:
                    return

                content = message.content.strip()
                author = message.author.name if message.author else "chatter"

                # Run registered commands first
                await self.handle_commands(message)

                # Optionally respond to ALL chat messages
                if respond_to_all and not content.startswith(prefix):
                    if parent._is_on_cooldown(author):
                        return

                    platform_context = f"[Twitch chatter {author}]: {content}"

                    # Check cache for identical messages (common repeat questions)
                    cached = parent._cache_get(content.lower())
                    if cached:
                        parent.messages_processed += 1
                        await message.channel.send(
                            f"@{author} {_truncate(cached, max_len)}"
                        )
                        parent._record_response(author)
                        return

                    loop = asyncio.get_event_loop()
                    try:
                        response = await loop.run_in_executor(
                            parent._executor, parent.pipeline_fn, platform_context
                        )
                        parent.messages_processed += 1
                        parent._record_response(author)
                        if response:
                            parent._cache_set(content.lower(), response)
                            await message.channel.send(
                                f"@{author} {_truncate(response, max_len)}"
                            )
                    except Exception as exc:
                        logger.error(f"[Twitch] Pipeline error: {exc}")
                        parent.last_error = str(exc)

            # Dynamic command registered below

        # Attach the AI command dynamically to avoid closure issues
        @_Bot.command(name=command_name)
        async def _cmd_revia(ctx: twitchio_commands.Context):
            raw = ctx.message.content
            parts = raw.split(maxsplit=1)
            text = parts[1].strip() if len(parts) > 1 else ""
            author = ctx.author.name

            if not text:
                await ctx.send(f"@{author} Ask me something after the command!")
                return

            if parent._is_on_cooldown(author):
                remaining = parent.config.get("user_cooldown_s", 8)
                await ctx.send(f"@{author} Hold on, I'm still thinking! Try again in a moment.")
                return

            platform_context = f"[Twitch chatter {author}]: {text}"

            # Check cache
            cached = parent._cache_get(text.lower())
            if cached:
                parent.messages_processed += 1
                parent._record_response(author)
                await ctx.send(f"@{author} {_truncate(cached, max_len)}")
                return

            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(
                    parent._executor, parent.pipeline_fn, platform_context
                )
                parent.messages_processed += 1
                parent._record_response(author)
                if response:
                    parent._cache_set(text.lower(), response)
                    await ctx.send(f"@{author} {_truncate(response, max_len)}")
            except Exception as exc:
                logger.error(f"[Twitch] Command pipeline error: {exc}")
                parent.last_error = str(exc)
                await ctx.send(f"@{author} Sorry, something went wrong!")

        return _Bot()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        if self.running:
            return

        token = self.config.get("oauth_token", "").strip()
        channels = self.config.get("channels", [])

        if not token:
            self.status = "error"
            self.last_error = "No OAuth token configured"
            logger.error("[Twitch] Cannot start — no OAuth token provided")
            return

        if not channels:
            self.status = "error"
            self.last_error = "No channels configured"
            logger.error("[Twitch] Cannot start — no channels configured")
            return

        self.running = True
        self.status = "connecting"

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._bot = self._build_bot()
                self._loop.run_until_complete(self._bot.start())
            except Exception as exc:
                self.status = "error"
                self.last_error = str(exc)
                logger.error(f"[Twitch] Fatal error: {exc}")
            finally:
                self.running = False
                if self.status != "error":
                    self.status = "stopped"

        self._thread = threading.Thread(target=_run, daemon=True, name="revia-twitch")
        self._thread.start()

    def stop(self):
        if not self.running or not self._bot:
            return

        async def _shutdown():
            await self._bot.close()

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
            "available": TWITCHIO_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    """Trim response to Twitch's character limit."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3].rstrip() + "..."
