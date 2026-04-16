"""
REVIA Song Library — manages a persistent catalogue of songs for !sing.

Songs can be:
  - **Raw**: A WAV/MP3 uploaded but not yet processed through the karaoke
    pipeline.  Processing happens on-the-fly when requested.
  - **Prepared**: Already run through vocal separation, lyrics extraction,
    pitch analysis, and synthesis.  Instant playback.

The library is stored on disk as a JSON catalogue (``songs.json``) alongside
a ``songs/`` directory containing the original and processed artefacts.

Typical layout::

    <repo>/songs/
        library.json          ← catalogue index
        my-song/
            original.wav
            instrumental.wav
            vocals.wav
            karaoke_output.wav
            analysis.json     ← serialised SongAnalysis
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SongEntry:
    """A single song in the library."""
    song_id: str                       # slug / directory name
    title: str                         # human-readable name
    artist: str = ""                   # optional artist tag
    original_path: str = ""            # path to the uploaded WAV
    prepared: bool = False             # has the karaoke pipeline run?
    analysis_path: str = ""            # path to serialised SongAnalysis JSON
    karaoke_output_path: str = ""      # path to final mixed WAV
    duration_sec: float = 0.0
    bpm: float = 0.0
    key: str = ""
    tags: list[str] = field(default_factory=list)   # mood / genre tags
    added_ts: float = 0.0             # epoch timestamp
    last_played_ts: float = 0.0
    play_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SongEntry:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Library manager
# ---------------------------------------------------------------------------

class SongLibrary:
    """Persistent on-disk song library.

    Args:
        base_dir: Root directory for the library (default ``<repo>/songs``).
    """

    CATALOGUE_FILE = "library.json"

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2] / "songs"
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._catalogue_path = self._base / self.CATALOGUE_FILE
        self._songs: dict[str, SongEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self._catalogue_path.exists():
            try:
                with open(self._catalogue_path) as f:
                    data = json.load(f)
                for entry in data.get("songs", []):
                    se = SongEntry.from_dict(entry)
                    self._songs[se.song_id] = se
                _log.info("[SongLibrary] Loaded %d songs", len(self._songs))
            except Exception as exc:
                _log.warning("[SongLibrary] Failed to load catalogue: %s", exc)

    def _save(self):
        try:
            payload = {"songs": [s.to_dict() for s in self._songs.values()]}
            with open(self._catalogue_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            _log.error("[SongLibrary] Failed to save catalogue: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_songs(self, *, prepared_only: bool = False,
                   tag: str | None = None) -> list[SongEntry]:
        """Return songs, optionally filtered."""
        songs = list(self._songs.values())
        if prepared_only:
            songs = [s for s in songs if s.prepared]
        if tag:
            tag_lower = tag.lower()
            songs = [s for s in songs if tag_lower in [t.lower() for t in s.tags]]
        return sorted(songs, key=lambda s: s.title.lower())

    def get_song(self, song_id: str) -> SongEntry | None:
        return self._songs.get(song_id)

    def find_by_title(self, query: str) -> SongEntry | None:
        """Fuzzy-ish title lookup (case-insensitive substring)."""
        q = query.lower().strip()
        for s in self._songs.values():
            if q in s.title.lower() or q in s.song_id:
                return s
        return None

    def add_song(self, wav_path: str, title: str, artist: str = "",
                 tags: list[str] | None = None) -> SongEntry:
        """Import a WAV into the library (copies the file)."""
        song_id = _slugify(title)
        # Handle duplicate IDs
        base_id = song_id
        counter = 1
        while song_id in self._songs:
            song_id = f"{base_id}-{counter}"
            counter += 1

        song_dir = self._base / song_id
        song_dir.mkdir(parents=True, exist_ok=True)

        # Copy original WAV
        ext = Path(wav_path).suffix or ".wav"
        dest = song_dir / f"original{ext}"
        shutil.copy2(wav_path, dest)

        entry = SongEntry(
            song_id=song_id,
            title=title,
            artist=artist,
            original_path=str(dest),
            tags=tags or [],
            added_ts=time.time(),
        )

        # Try to get duration
        try:
            from .sing_mode import _get_audio_duration
            entry.duration_sec = _get_audio_duration(str(dest))
        except Exception:
            pass

        self._songs[song_id] = entry
        self._save()
        _log.info("[SongLibrary] Added song: %s (%s)", title, song_id)
        return entry

    def mark_prepared(self, song_id: str, analysis_dict: dict,
                      karaoke_wav: str):
        """Mark a song as prepared after running the karaoke pipeline.

        The karaoke output WAV is copied into the library's song directory
        so it persists independently of SingMode's temp dir.

        Args:
            song_id: The library song ID.
            analysis_dict: Serialised SongAnalysis (from ``to_dict()``).
            karaoke_wav: Path to the mixed karaoke WAV file.
        """
        entry = self._songs.get(song_id)
        if not entry:
            _log.warning("[SongLibrary] Cannot mark unknown song: %s", song_id)
            return

        song_dir = self._base / song_id
        song_dir.mkdir(parents=True, exist_ok=True)

        # Save analysis JSON
        analysis_path = song_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis_dict, f, indent=2)

        # Copy karaoke WAV into library
        lib_wav = song_dir / "karaoke_output.wav"
        if karaoke_wav and Path(karaoke_wav).exists():
            shutil.copy2(karaoke_wav, lib_wav)

        entry.prepared = True
        entry.analysis_path = str(analysis_path)
        entry.karaoke_output_path = str(lib_wav) if lib_wav.exists() else ""
        entry.bpm = analysis_dict.get("bpm", 0.0)
        entry.key = analysis_dict.get("key", "")
        entry.duration_sec = analysis_dict.get("duration_sec", entry.duration_sec)

        self._save()
        _log.info("[SongLibrary] Marked prepared: %s", song_id)

    def record_play(self, song_id: str):
        """Increment play count and update last-played time."""
        entry = self._songs.get(song_id)
        if entry:
            entry.play_count += 1
            entry.last_played_ts = time.time()
            self._save()

    def remove_song(self, song_id: str) -> bool:
        """Remove a song and its files from the library."""
        entry = self._songs.pop(song_id, None)
        if not entry:
            return False
        song_dir = self._base / song_id
        if song_dir.exists():
            shutil.rmtree(song_dir, ignore_errors=True)
        self._save()
        _log.info("[SongLibrary] Removed song: %s", song_id)
        return True

    @property
    def song_count(self) -> int:
        return len(self._songs)

    @property
    def prepared_count(self) -> int:
        return sum(1 for s in self._songs.values() if s.prepared)

    @property
    def base_dir(self) -> Path:
        return self._base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert a title into a filesystem-safe slug."""
    import re
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    return slug[:80] or "untitled"
