"""
REVIA !sing Command Handler — bridges chat commands to the sing pipeline.

This module is platform-agnostic: Twitch, Discord, and the GUI all route
through ``SingCommandHandler`` which coordinates the song library, queue,
SingMode pipeline, and TTS backend.

Chat commands:
    !sing                → Revia picks a song (mood or random)
    !sing <song name>    → Request a specific song from the library
    !sing skip           → Skip the current song
    !sing queue          → Show the current queue
    !sing list           → List available songs
    !sing stop           → Stop singing entirely
    !sing np             → Show what's currently playing ("now playing")
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

_log = logging.getLogger(__name__)


class SingCommandHandler:
    """Processes ``!sing`` commands and drives the karaoke pipeline.

    This is the central orchestrator that connects:
      - ``SongLibrary``  (what songs are available)
      - ``SingQueue``    (what's requested / next up)
      - ``SingMode``     (the actual karaoke processing + playback)

    Args:
        library: ``SongLibrary`` instance.
        queue: ``SingQueue`` instance.
        sing_mode_factory: A callable that returns a fresh ``SingMode``
            instance (typically ``tts_backend.get_sing_mode``).
        voice_profile_fn: Callable that returns the current voice profile.
        on_chat_reply: Callback ``(text: str) -> None`` used to send
            responses back to the platform (Twitch/Discord/GUI).
        on_state_change: Callback when the sing pipeline state changes.
    """

    def __init__(
        self,
        library,
        queue,
        sing_mode_factory: Callable,
        voice_profile_fn: Callable | None = None,
        on_chat_reply: Callable[[str], None] | None = None,
        on_state_change: Callable[[str], None] | None = None,
    ):
        self._library = library
        self._queue = queue
        self._sing_factory = sing_mode_factory
        self._voice_profile_fn = voice_profile_fn
        self._on_reply = on_chat_reply
        self._on_state = on_state_change

        self._sing: object | None = None       # active SingMode instance
        self._busy = False
        self._lock = threading.Lock()

        # Wire queue callbacks
        self._queue.on_now_playing = self._on_now_playing_changed

    def _on_now_playing_changed(self, item):
        """Called by the queue when now-playing changes."""
        if item and self._on_reply:
            self._reply(f"♪ Now playing: {item.title}")

    # ------------------------------------------------------------------
    # Main command dispatcher
    # ------------------------------------------------------------------

    def handle(self, raw_args: str, author: str = "") -> str:
        """Parse and dispatch a ``!sing`` command.

        Args:
            raw_args: Everything after ``!sing `` (may be empty).
            author: The username of whoever issued the command.

        Returns:
            A reply string to send back to chat.
        """
        args = raw_args.strip()
        cmd = args.split()[0].lower() if args else ""

        if cmd == "skip":
            return self._cmd_skip()
        elif cmd == "stop":
            return self._cmd_stop()
        elif cmd == "queue":
            return self._cmd_queue()
        elif cmd == "list":
            return self._cmd_list()
        elif cmd in ("np", "nowplaying", "now"):
            return self._cmd_now_playing()
        elif args:
            # Treat as song request
            return self._cmd_request(args, author)
        else:
            # No args -> auto-pick
            return self._cmd_auto_pick(author)

    # ------------------------------------------------------------------
    # Sub-commands
    # ------------------------------------------------------------------

    def _cmd_request(self, query: str, author: str) -> str:
        """Request a specific song by name."""
        item = self._queue.request_song(query, requested_by=author)
        if not item:
            # Try fuzzy search to suggest alternatives
            available = self._library.list_songs(prepared_only=True)
            if not available:
                return "No songs in the library yet! Add some WAVs first."
            titles = [s.title for s in available[:5]]
            return (f"Couldn't find \"{query}\". "
                    f"Try one of: {', '.join(titles)}")

        pos = self._queue.size
        if pos == 1 and not self._busy:
            # Queue was empty, start playing immediately
            self._process_next()
            return f"♪ Now singing: {item.title}!"
        return f"♪ Added to queue: {item.title} (position #{pos})"

    def _cmd_auto_pick(self, author: str) -> str:
        """Auto-pick a song (mood or random) and start."""
        if self._busy:
            # Already singing - queue an auto-pick
            item = self._queue.preview_auto_pick()
            if item:
                self._queue.add_by_id(item.song_id, requested_by=author,
                                       pick_mode=item.pick_mode)
                return f"♪ Queued: {item.title} (Revia's choice!)"
            return "No songs available to pick from!"

        item = self._queue.next()
        if not item:
            return "No songs in the library! Add some WAVs first."

        self._start_playing(item)
        mode_text = "feeling like" if item.pick_mode.value == "mood" else "randomly picked"
        return f"♪ Revia {mode_text} singing: {item.title}!"

    def _cmd_skip(self) -> str:
        """Skip the current song."""
        if not self._busy:
            return "Nothing is playing right now."
        self._stop_current()
        self._process_next()
        np = self._queue.now_playing
        if np:
            return f"Skipped! Now playing: {np.title}"
        return "Skipped! Queue is empty."

    def _cmd_stop(self) -> str:
        """Stop singing and clear queue."""
        self._stop_current()
        self._queue.clear()
        self._queue.clear_now_playing()
        return "Stopped singing and cleared the queue."

    def _cmd_queue(self) -> str:
        """Show the current queue."""
        items = self._queue.peek()
        np = self._queue.now_playing
        parts = []
        if np:
            parts.append(f"♪ Now: {np.title}")
        if not items:
            if not np:
                return "Queue is empty. Use !sing to start!"
            parts.append("Queue: empty")
        else:
            for i, item in enumerate(items[:10], 1):
                by = f" ({item.requested_by})" if item.requested_by else ""
                parts.append(f"  {i}. {item.title}{by}")
            if len(items) > 10:
                parts.append(f"  ...and {len(items) - 10} more")
        return " | ".join(parts) if len(parts) <= 3 else "\n".join(parts)

    def _cmd_list(self) -> str:
        """List available songs."""
        songs = self._library.list_songs(prepared_only=True)
        if not songs:
            all_songs = self._library.list_songs()
            if not all_songs:
                return "No songs in the library yet!"
            return (f"{len(all_songs)} songs in library, but none are "
                    f"prepared for singing yet. Processing needed first.")

        titles = [f"• {s.title}" + (f" - {s.artist}" if s.artist else "")
                  for s in songs[:15]]
        header = f"♪ Song Library ({len(songs)} songs):"
        return header + "\n" + "\n".join(titles)

    def _cmd_now_playing(self) -> str:
        """Show what's currently playing."""
        np = self._queue.now_playing
        if not np:
            return "Nothing is playing right now."
        by = f" (requested by {np.requested_by})" if np.requested_by else ""
        return f"♪ Now playing: {np.title}{by}"

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def _process_next(self):
        """Pull the next song from the queue and start processing."""
        item = self._queue.next()
        if item:
            self._start_playing(item)

    def _start_playing(self, item):
        """Start the karaoke pipeline for a queue item."""
        entry = self._library.get_song(item.song_id)
        if not entry:
            self._reply(f"Song '{item.title}' not found in library!")
            self._process_next()
            return

        self._busy = True
        if self._on_state:
            self._on_state("preparing")

        def _run():
            try:
                sing = self._sing_factory()
                self._sing = sing

                voice = self._voice_profile_fn() if self._voice_profile_fn else None

                if entry.prepared and entry.karaoke_output_path:
                    # Instant playback from pre-processed cache
                    _log.info("[SingCmd] Playing cached: %s", entry.title)
                    if self._on_state:
                        self._on_state("playing")
                    sing._tts.play_wav(entry.karaoke_output_path)
                    self._library.record_play(item.song_id)
                else:
                    # On-the-fly processing
                    _log.info("[SingCmd] Processing on-the-fly: %s", entry.title)

                    def _on_ready(analysis):
                        # Cache for future instant playback
                        self._library.mark_prepared(
                            item.song_id, analysis.to_dict(),
                            analysis.karaoke_output_path,
                        )
                        self._library.record_play(item.song_id)
                        if self._on_state:
                            self._on_state("playing")

                    def _on_progress(stage, current, total):
                        if self._on_state:
                            self._on_state(f"{stage} ({current}/{total})")

                    analysis = sing.prepare(entry.original_path, voice)
                    if analysis and analysis.karaoke_output_path:
                        _on_ready(analysis)
                        sing.play(analysis)

                # Song finished - advance queue
                self._busy = False
                self._queue.clear_now_playing()
                if self._on_state:
                    self._on_state("idle")
                self._process_next()

            except Exception as exc:
                _log.error("[SingCmd] Playback error: %s", exc, exc_info=True)
                self._busy = False
                self._queue.clear_now_playing()
                if self._on_state:
                    self._on_state("error")
                self._reply(f"Oops, something went wrong playing {item.title}!")
                self._process_next()

        threading.Thread(target=_run, daemon=True, name="revia-sing-play").start()

    def _stop_current(self):
        """Stop the currently playing song."""
        if self._sing:
            try:
                self._sing.stop()
            except Exception:
                pass
            try:
                self._sing.cleanup()
            except Exception:
                pass
            self._sing = None
        self._busy = False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _reply(self, text: str):
        """Send a reply through the platform callback."""
        if self._on_reply:
            try:
                self._on_reply(text)
            except Exception:
                pass

    @property
    def is_singing(self) -> bool:
        return self._busy

    def set_reply_callback(self, fn: Callable[[str], None]):
        """Update the chat reply callback (e.g. when switching platforms)."""
        self._on_reply = fn
