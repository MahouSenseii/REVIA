"""
REVIA Sing Queue — manages song requests and auto-picking for !sing.

Three modes of song selection:
  1. **Request** — ``!sing <song name>`` adds a specific song to the queue.
  2. **Mood / Feelings** — Revia picks a song that matches her current
     emotional state (e.g. happy → upbeat, sad → slow ballad).
  3. **Random** — Pick any prepared song from the library at random.

The queue is processed FIFO.  When the queue is empty and auto-pick is
enabled, Revia will choose a song based on mood or at random.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PickMode(str, Enum):
    REQUEST = "request"       # viewer/user requested a specific song
    MOOD = "mood"             # Revia picks based on current emotion
    RANDOM = "random"         # random pick from library


@dataclass
class QueueItem:
    """A single entry in the sing queue."""
    song_id: str
    title: str
    requested_by: str = ""    # username of requester (empty = auto-pick)
    pick_mode: PickMode = PickMode.REQUEST
    queued_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "song_id": self.song_id,
            "title": self.title,
            "requested_by": self.requested_by,
            "pick_mode": self.pick_mode.value,
            "queued_at": self.queued_at,
        }


# ---------------------------------------------------------------------------
# Mood → tag mapping
# ---------------------------------------------------------------------------

# Maps Revia's emotion states to song tags that match the vibe.
# Tags are matched against SongEntry.tags in the library.
_MOOD_TAG_MAP: dict[str, list[str]] = {
    "happy":      ["upbeat", "happy", "energetic", "pop", "dance"],
    "excited":    ["energetic", "hype", "fast", "dance", "edm"],
    "sad":        ["slow", "sad", "ballad", "melancholy", "acoustic"],
    "angry":      ["intense", "rock", "metal", "aggressive", "loud"],
    "calm":       ["chill", "lofi", "ambient", "relaxing", "soft"],
    "playful":    ["fun", "quirky", "cute", "pop", "playful"],
    "tired":      ["slow", "soft", "ambient", "lofi", "sleepy"],
    "neutral":    ["pop", "chill", "upbeat", "any"],
    "love":       ["romantic", "love", "ballad", "sweet"],
    "scared":     ["eerie", "atmospheric", "slow", "dark"],
}


# ---------------------------------------------------------------------------
# Queue manager
# ---------------------------------------------------------------------------

class SingQueue:
    """Thread-safe song request queue with mood-based auto-picking.

    Args:
        library: A ``SongLibrary`` instance for song lookups.
        get_current_mood: Callable that returns Revia's current emotion
            string (e.g. ``"happy"``, ``"sad"``).  If *None*, mood-based
            picking falls back to random.
        max_queue_size: Maximum number of pending requests.
        auto_pick_enabled: When *True* and the queue drains, Revia
            automatically picks the next song via mood or random.
        auto_pick_random_chance: Probability (0-1) that auto-pick uses
            random mode instead of mood mode.
    """

    def __init__(
        self,
        library,
        get_current_mood: Callable[[], str] | None = None,
        max_queue_size: int = 50,
        auto_pick_enabled: bool = True,
        auto_pick_random_chance: float = 0.3,
    ):
        self._library = library
        self._get_mood = get_current_mood
        self._queue: deque[QueueItem] = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()
        self._max = max_queue_size
        self.auto_pick_enabled = auto_pick_enabled
        self.auto_pick_random_chance = auto_pick_random_chance

        # Callbacks
        self.on_queue_changed: Callable[[], None] | None = None
        self.on_now_playing: Callable[[QueueItem], None] | None = None

        # Currently playing
        self._now_playing: QueueItem | None = None

    # ------------------------------------------------------------------
    # Request handling (!sing <name>)
    # ------------------------------------------------------------------

    def request_song(self, query: str, requested_by: str = "") -> QueueItem | None:
        """Add a song request to the queue by title search.

        Returns the QueueItem if found and added, None otherwise.
        """
        entry = self._library.find_by_title(query)
        if not entry:
            _log.info("[SingQueue] Song not found: '%s'", query)
            return None

        item = QueueItem(
            song_id=entry.song_id,
            title=entry.title,
            requested_by=requested_by,
            pick_mode=PickMode.REQUEST,
            queued_at=time.time(),
        )
        with self._lock:
            if len(self._queue) >= self._max:
                _log.warning("[SingQueue] Queue full, rejecting: %s", entry.title)
                return None
            self._queue.append(item)

        _log.info("[SingQueue] Queued: '%s' (by %s)", entry.title,
                  requested_by or "auto")
        self._notify_changed()
        return item

    def add_by_id(self, song_id: str, requested_by: str = "",
                  pick_mode: PickMode = PickMode.REQUEST) -> QueueItem | None:
        """Add a song by its library ID."""
        entry = self._library.get_song(song_id)
        if not entry:
            return None
        item = QueueItem(
            song_id=song_id,
            title=entry.title,
            requested_by=requested_by,
            pick_mode=pick_mode,
            queued_at=time.time(),
        )
        with self._lock:
            if len(self._queue) >= self._max:
                return None
            self._queue.append(item)
        self._notify_changed()
        return item

    # ------------------------------------------------------------------
    # Next song
    # ------------------------------------------------------------------

    def next(self) -> QueueItem | None:
        """Pop the next song from the queue.

        If the queue is empty and auto-pick is enabled, Revia picks one
        based on mood or at random.
        """
        with self._lock:
            if self._queue:
                item = self._queue.popleft()
                self._now_playing = item
                self._notify_changed()
                return item

        # Queue empty — auto-pick if enabled
        if not self.auto_pick_enabled:
            return None

        item = self._auto_pick()
        if item:
            self._now_playing = item
            self._notify_changed()
        return item

    def _auto_pick(self) -> QueueItem | None:
        """Auto-pick a song: mood-based or random."""
        use_random = random.random() < self.auto_pick_random_chance

        if not use_random and self._get_mood:
            item = self._pick_by_mood()
            if item:
                return item

        return self._pick_random()

    def _pick_by_mood(self) -> QueueItem | None:
        """Pick a prepared song matching Revia's current mood."""
        try:
            mood = self._get_mood() if self._get_mood else "neutral"
        except Exception:
            mood = "neutral"

        mood = (mood or "neutral").lower()
        target_tags = _MOOD_TAG_MAP.get(mood, _MOOD_TAG_MAP["neutral"])

        prepared = self._library.list_songs(prepared_only=True)
        if not prepared:
            return None

        # Score songs by how many mood-tags they match
        scored: list[tuple[float, object]] = []
        for song in prepared:
            song_tags = {t.lower() for t in song.tags}
            match_count = sum(1 for t in target_tags if t in song_tags)
            # Boost songs that haven't been played recently
            recency_penalty = 0.0
            if song.last_played_ts:
                hours_ago = (time.time() - song.last_played_ts) / 3600
                recency_penalty = max(0, 1.0 - hours_ago / 24)  # decays over 24h
            score = match_count - recency_penalty * 0.5
            scored.append((score, song))

        # Sort by score descending, take top 5, pick randomly among them
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:min(5, len(scored))]
        chosen = random.choice(top)[1]

        _log.info("[SingQueue] Mood pick (%s): '%s'", mood, chosen.title)
        return QueueItem(
            song_id=chosen.song_id,
            title=chosen.title,
            requested_by="",
            pick_mode=PickMode.MOOD,
            queued_at=time.time(),
        )

    def _pick_random(self) -> QueueItem | None:
        """Pick a random prepared song."""
        prepared = self._library.list_songs(prepared_only=True)
        if not prepared:
            # Fall back to any song (will need on-the-fly processing)
            all_songs = self._library.list_songs()
            if not all_songs:
                return None
            chosen = random.choice(all_songs)
        else:
            chosen = random.choice(prepared)

        _log.info("[SingQueue] Random pick: '%s'", chosen.title)
        return QueueItem(
            song_id=chosen.song_id,
            title=chosen.title,
            requested_by="",
            pick_mode=PickMode.RANDOM,
            queued_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Queue state
    # ------------------------------------------------------------------

    @property
    def now_playing(self) -> QueueItem | None:
        return self._now_playing

    def clear_now_playing(self):
        self._now_playing = None
        self._notify_changed()

    def peek(self) -> list[QueueItem]:
        """Return a snapshot of the current queue (non-destructive)."""
        with self._lock:
            return list(self._queue)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def clear(self):
        """Clear all pending requests."""
        with self._lock:
            self._queue.clear()
        self._notify_changed()

    def remove(self, song_id: str) -> bool:
        """Remove the first occurrence of a song from the queue."""
        with self._lock:
            for i, item in enumerate(self._queue):
                if item.song_id == song_id:
                    del self._queue[i]
                    self._notify_changed()
                    return True
        return False

    def skip(self) -> QueueItem | None:
        """Skip the currently playing song and move to next.

        Returns the next QueueItem, or None if queue is empty.
        """
        self._now_playing = None
        return self.next()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        with self._lock:
            queue_list = [item.to_dict() for item in self._queue]
        return {
            "queue": queue_list,
            "now_playing": self._now_playing.to_dict() if self._now_playing else None,
            "size": len(queue_list),
            "auto_pick_enabled": self.auto_pick_enabled,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _notify_changed(self):
        if self.on_queue_changed:
            try:
                self.on_queue_changed()
            except Exception:
                pass
