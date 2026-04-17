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
        # Dedicated thread pool - keeps Twitch I/O off the default asyncio executor
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="revia-twitch-llm")
        self.running = False
        self.status = "stopped"
        self.last_error = None
        self.messages_processed = 0
        self._counter_lock = threading.Lock()

        # Sing command handler (set externally by IntegrationManager)
        self.sing_command_handler = None

        # Cache frequently-read config values at init time
        self._cooldown_s: float = config.get("user_cooldown_s", 3)  # Reduced from 8 to 3 seconds
        self._cache_ttl_s: float = config.get("cache_ttl_s", 300)

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

    def _is_on_cooldown(self, username: str, is_privileged: bool = False) -> bool:
        if self._cooldown_s <= 0:
            return False
        # Mods/subs get fast-track (1 second cooldown)
        effective_cooldown = 1 if is_privileged else self._cooldown_s
        with self._cooldown_lock:
            last = self._cooldowns.get(username, 0.0)
            return (time.monotonic() - last) < effective_cooldown

    def _record_response(self, username: str):
        """Record response time. Must be called under _cooldown_lock."""
        now = time.monotonic()
        self._cooldowns[username] = now
        # Evict expired cooldown entries to prevent unbounded growth.
        # If no entries are expired (all users were very recently active), fall back
        # to removing the oldest entry so the dict never exceeds the cap.
        if len(self._cooldowns) > 500:
            cutoff = now - max(self._cooldown_s * 10, 300)
            expired = [u for u, ts in self._cooldowns.items() if ts < cutoff]
            if expired:
                for u in expired:
                    del self._cooldowns[u]
            else:
                # All entries are fresh - remove the single oldest to stay under cap
                oldest = min(self._cooldowns, key=self._cooldowns.__getitem__)
                del self._cooldowns[oldest]

    def _record_response_locked(self, username: str):
        """Record response time with lock acquisition."""
        with self._cooldown_lock:
            self._record_response(username)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: str):
        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry:
                response, ts = entry
                if now - ts < self._cache_ttl_s:
                    return response
                del self._cache[key]
            # Periodic eviction of expired entries
            if len(self._cache) > 50:
                expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self._cache_ttl_s]
                for k in expired:
                    del self._cache[k]
        return None

    def _cache_set(self, key: str, value: str):
        now = time.monotonic()
        with self._cache_lock:
            # Evict oldest entry if cache exceeds 100 items
            if len(self._cache) >= 100:
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (value, now)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_response(self, prompt: str, source: str = "twitch_event") -> str:
        """Generate a response from the AI pipeline for events."""
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self.pipeline_fn, prompt),
                timeout=30.0,
            )
            return response or ""
        except asyncio.TimeoutError:
            logger.error("[Twitch] Event response timed out after 30s (source=%s)", source)
            return ""
        except Exception as exc:
            logger.error(f"[Twitch] Event response generation failed: {exc}")
            return ""

    async def _run_pipeline(self, loop, author: str, text: str, send_fn) -> None:
        """Shared pipeline execution for both command and respond_to_all modes."""
        max_len = self.config.get("max_response_len", 450)
        cached = self._cache_get(text.lower())
        if cached:
            with self._counter_lock:
                self.messages_processed += 1
            with self._cooldown_lock:
                self._record_response(author)
            await send_fn(_truncate(cached, max_len))
            return
        try:
            platform_context = f"[Twitch chatter {author}]: {text}"
            response = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self.pipeline_fn, platform_context),
                timeout=30.0,
            )
            with self._counter_lock:
                self.messages_processed += 1
            self._record_response_locked(author)
            if response:
                self._cache_set(text.lower(), response)
                await send_fn(_truncate(response, max_len))
        except asyncio.TimeoutError:
            logger.error("[Twitch] Pipeline timed out after 30s (user=%s)", author)
            self.last_error = "pipeline_timeout"
            await send_fn(f"@{author} I took too long to think — please try again.")
        except Exception as exc:
            logger.error(f"[Twitch] Pipeline error: {exc}")
            self.last_error = str(exc)

    def _build_bot(self):
        if not TWITCHIO_AVAILABLE:
            raise RuntimeError("twitchio is not installed")

        cfg = self.config
        token = cfg.get("oauth_token", "")
        channels = cfg.get("channels", [])
        prefix = cfg.get("prefix", "!")
        command_name = cfg.get("command", "revia")
        respond_to_all = cfg.get("respond_to_all", False)
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
                    is_privileged = getattr(message.author, 'is_mod', False) or getattr(message.author, 'is_subscriber', False)
                    if parent._is_on_cooldown(author, is_privileged):
                        return
                    loop = asyncio.get_running_loop()
                    await parent._run_pipeline(
                        loop, author, content,
                        lambda r: message.channel.send(f"@{author} {r}"),
                    )

            # Dynamic command registered below

            async def event_subscription(self, event):
                """Handle new subscriber events."""
                if not hasattr(event, 'channel') or event.channel is None:
                    logger.warning("Subscription event received with no channel, skipping")
                    return
                try:
                    user = event.user.name if hasattr(event, 'user') else 'Someone'
                    tier_names = {'1000': 'Tier 1', '2000': 'Tier 2', '3000': 'Tier 3'}
                    tier = tier_names.get(str(getattr(event, 'tier', '1000')), 'Tier 1')
                    prompt = f"[STREAM EVENT] {user} just subscribed with a {tier} sub! React with excitement and thank them personally."
                    response = await parent._generate_response(prompt, source="twitch_event")
                    if response and hasattr(event, 'channel'):
                        await event.channel.send(response)
                except Exception as e:
                    logger.error("Subscription event handler failed: %s", e)

            async def event_raid(self, event):
                """Handle raid events."""
                if not hasattr(event, 'channel') or event.channel is None:
                    logger.warning("Raid event received with no channel, skipping")
                    return
                try:
                    raider = getattr(event, 'raider', None)
                    raider_name = raider.name if raider else 'Someone'
                    viewers = getattr(event, 'viewers', 0)
                    prompt = f"[STREAM EVENT] {raider_name} is raiding with {viewers} viewers! Give them an energetic, welcoming reaction."
                    response = await parent._generate_response(prompt, source="twitch_event")
                    if response:
                        for channel in self.connected_channels:
                            await channel.send(response)
                except Exception as e:
                    logger.error("Raid event handler failed: %s", e)

            async def event_cheer(self, event):
                """Handle bits/cheer events."""
                if not hasattr(event, 'channel') or event.channel is None:
                    logger.warning("Cheer event received with no channel, skipping")
                    return
                try:
                    user = event.user.name if hasattr(event, 'user') else 'Someone'
                    bits = getattr(event, 'bits', 0)
                    message = getattr(event, 'message', '')
                    prompt = f"[STREAM EVENT] {user} cheered {bits} bits"
                    if message:
                        prompt += f" with message: '{message}'"
                    prompt += "! React with gratitude and energy proportional to the amount."
                    response = await parent._generate_response(prompt, source="twitch_event")
                    if response and hasattr(event, 'channel'):
                        await event.channel.send(response)
                except Exception as e:
                    logger.error("Cheer event handler failed: %s", e)

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

            is_privileged = getattr(ctx.author, 'is_mod', False) or getattr(ctx.author, 'is_subscriber', False)
            if parent._is_on_cooldown(author, is_privileged):
                await ctx.send(f"@{author} Hold on, I'm still thinking! Try again in a moment.")
                return

            loop = asyncio.get_running_loop()
            await parent._run_pipeline(
                loop, author, text,
                lambda r: ctx.send(f"@{author} {r}"),
            )

        # !sing command - routes to SingCommandHandler
        @_Bot.command(name="sing")
        async def _cmd_sing(ctx: twitchio_commands.Context):
            raw = ctx.message.content
            parts = raw.split(maxsplit=1)
            args = parts[1].strip() if len(parts) > 1 else ""
            author = ctx.author.name

            handler = parent.sing_command_handler
            if not handler:
                await ctx.send(f"@{author} Sing mode is not enabled yet!")
                return

            # Run command handler in executor to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                reply = await asyncio.wait_for(
                    loop.run_in_executor(parent._executor, handler.handle, args, author),
                    timeout=30.0,
                )
                if reply:
                    max_len = parent.config.get("max_response_len", 450)
                    await ctx.send(f"@{author} {_truncate(reply, max_len)}")
            except asyncio.TimeoutError:
                logger.error("[Twitch] !sing timed out after 30s")
                await ctx.send(f"@{author} That took too long — please try again.")
            except Exception as exc:
                logger.error("[Twitch] !sing command error: %s", exc)
                await ctx.send(f"@{author} Something went wrong with !sing.")

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
