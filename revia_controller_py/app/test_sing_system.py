"""Tests for the !sing system: song_library, sing_queue, sing_command."""

import os
import tempfile
import unittest
import wave
import struct
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: str, duration_sec: float = 1.0, sr: int = 44100):
    """Create a minimal valid WAV file for testing."""
    n_samples = int(sr * duration_sec)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


# ===========================================================================
# SongLibrary tests
# ===========================================================================

class TestSongLibrary(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="revia_test_lib_")
        from song_library import SongLibrary
        self.lib = SongLibrary(base_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_library(self):
        self.assertEqual(self.lib.song_count, 0)
        self.assertEqual(self.lib.prepared_count, 0)
        self.assertEqual(self.lib.list_songs(), [])

    def test_add_and_list(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        entry = self.lib.add_song(wav, "Test Song", artist="TestArtist", tags=["pop"])
        self.assertEqual(entry.title, "Test Song")
        self.assertEqual(entry.artist, "TestArtist")
        self.assertEqual(entry.tags, ["pop"])
        self.assertFalse(entry.prepared)
        self.assertEqual(self.lib.song_count, 1)

    def test_find_by_title(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        self.lib.add_song(wav, "My Awesome Song")
        found = self.lib.find_by_title("awesome")
        self.assertIsNotNone(found)
        self.assertEqual(found.title, "My Awesome Song")

    def test_find_by_title_not_found(self):
        self.assertIsNone(self.lib.find_by_title("nonexistent"))

    def test_mark_prepared(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        entry = self.lib.add_song(wav, "Prep Test")
        karaoke_wav = os.path.join(self.tmpdir, "karaoke.wav")
        _make_wav(karaoke_wav)
        self.lib.mark_prepared(entry.song_id, {"bpm": 120, "key": "C"}, karaoke_wav)
        updated = self.lib.get_song(entry.song_id)
        self.assertTrue(updated.prepared)
        self.assertEqual(updated.bpm, 120)
        self.assertEqual(self.lib.prepared_count, 1)

    def test_record_play(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        entry = self.lib.add_song(wav, "Play Test")
        self.lib.record_play(entry.song_id)
        updated = self.lib.get_song(entry.song_id)
        self.assertEqual(updated.play_count, 1)
        self.assertGreater(updated.last_played_ts, 0)

    def test_remove_song(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        entry = self.lib.add_song(wav, "Remove Me")
        self.assertTrue(self.lib.remove_song(entry.song_id))
        self.assertEqual(self.lib.song_count, 0)
        self.assertFalse(self.lib.remove_song("nonexistent"))

    def test_duplicate_id_handling(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        e1 = self.lib.add_song(wav, "Same Title")
        e2 = self.lib.add_song(wav, "Same Title")
        self.assertNotEqual(e1.song_id, e2.song_id)
        self.assertEqual(self.lib.song_count, 2)

    def test_persistence(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        self.lib.add_song(wav, "Persistent Song")
        # Reload
        from song_library import SongLibrary
        lib2 = SongLibrary(base_dir=self.tmpdir)
        self.assertEqual(lib2.song_count, 1)
        self.assertEqual(lib2.list_songs()[0].title, "Persistent Song")

    def test_filter_prepared_only(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        self.lib.add_song(wav, "Unprepared")
        e2 = self.lib.add_song(wav, "Prepared")
        karaoke = os.path.join(self.tmpdir, "k.wav")
        _make_wav(karaoke)
        self.lib.mark_prepared(e2.song_id, {}, karaoke)
        prepared = self.lib.list_songs(prepared_only=True)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].title, "Prepared")

    def test_filter_by_tag(self):
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        self.lib.add_song(wav, "Rock Song", tags=["rock", "loud"])
        self.lib.add_song(wav, "Pop Song", tags=["pop", "upbeat"])
        rock = self.lib.list_songs(tag="rock")
        self.assertEqual(len(rock), 1)
        self.assertEqual(rock[0].title, "Rock Song")


# ===========================================================================
# SingQueue tests
# ===========================================================================

class TestSingQueue(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="revia_test_q_")
        from song_library import SongLibrary
        self.lib = SongLibrary(base_dir=self.tmpdir)
        # Add some songs
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        self.s1 = self.lib.add_song(wav, "Song One", tags=["happy", "upbeat"])
        self.s2 = self.lib.add_song(wav, "Song Two", tags=["sad", "ballad"])
        self.s3 = self.lib.add_song(wav, "Song Three", tags=["chill", "lofi"])
        # Mark all prepared
        k = os.path.join(self.tmpdir, "k.wav")
        _make_wav(k)
        for s in [self.s1, self.s2, self.s3]:
            self.lib.mark_prepared(s.song_id, {"bpm": 120}, k)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_queue(self, **kwargs):
        from sing_queue import SingQueue
        return SingQueue(self.lib, **kwargs)

    def test_request_song(self):
        q = self._make_queue(auto_pick_enabled=False)
        item = q.request_song("Song One", requested_by="viewer1")
        self.assertIsNotNone(item)
        self.assertEqual(item.title, "Song One")
        self.assertEqual(q.size, 1)

    def test_request_not_found(self):
        q = self._make_queue(auto_pick_enabled=False)
        item = q.request_song("Nonexistent Song")
        self.assertIsNone(item)

    def test_fifo_order(self):
        q = self._make_queue(auto_pick_enabled=False)
        q.request_song("Song One")
        q.request_song("Song Two")
        item = q.next()
        self.assertEqual(item.title, "Song One")
        item = q.next()
        self.assertEqual(item.title, "Song Two")

    def test_auto_pick_random(self):
        q = self._make_queue(auto_pick_enabled=True, auto_pick_random_chance=1.0)
        item = q.next()
        self.assertIsNotNone(item)
        self.assertIn(item.title, ["Song One", "Song Two", "Song Three"])

    def test_auto_pick_mood(self):
        q = self._make_queue(
            get_current_mood=lambda: "happy",
            auto_pick_enabled=True,
            auto_pick_random_chance=0.0,
        )
        item = q.next()
        self.assertIsNotNone(item)
        # Should prefer Song One (happy, upbeat tags)

    def test_queue_full(self):
        q = self._make_queue(auto_pick_enabled=False, max_queue_size=2)
        q.request_song("Song One")
        q.request_song("Song Two")
        result = q.request_song("Song Three")
        self.assertIsNone(result)  # Queue full

    def test_clear(self):
        q = self._make_queue(auto_pick_enabled=False)
        q.request_song("Song One")
        q.request_song("Song Two")
        q.clear()
        self.assertTrue(q.is_empty)

    def test_remove(self):
        q = self._make_queue(auto_pick_enabled=False)
        q.request_song("Song One")
        q.request_song("Song Two")
        removed = q.remove(self.s1.song_id)
        self.assertTrue(removed)
        self.assertEqual(q.size, 1)

    def test_peek(self):
        q = self._make_queue(auto_pick_enabled=False)
        q.request_song("Song One")
        q.request_song("Song Two")
        peeked = q.peek()
        self.assertEqual(len(peeked), 2)
        self.assertEqual(q.size, 2)  # peek doesn't consume

    def test_to_dict(self):
        q = self._make_queue(auto_pick_enabled=False)
        q.request_song("Song One")
        d = q.to_dict()
        self.assertEqual(d["size"], 1)
        self.assertEqual(d["queue"][0]["title"], "Song One")

    def test_add_by_id(self):
        q = self._make_queue(auto_pick_enabled=False)
        from sing_queue import PickMode
        item = q.add_by_id(self.s2.song_id, pick_mode=PickMode.MOOD)
        self.assertIsNotNone(item)
        self.assertEqual(item.pick_mode, PickMode.MOOD)


# ===========================================================================
# SingCommandHandler tests
# ===========================================================================

class TestSingCommandHandler(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="revia_test_cmd_")
        from song_library import SongLibrary
        from sing_queue import SingQueue
        from sing_command import SingCommandHandler

        self.lib = SongLibrary(base_dir=self.tmpdir)
        wav = os.path.join(self.tmpdir, "test.wav")
        _make_wav(wav)
        self.s1 = self.lib.add_song(wav, "Alpha Song", tags=["pop"])
        self.s2 = self.lib.add_song(wav, "Beta Song", tags=["rock"])
        k = os.path.join(self.tmpdir, "k.wav")
        _make_wav(k)
        for s in [self.s1, self.s2]:
            self.lib.mark_prepared(s.song_id, {"bpm": 120}, k)

        self.queue = SingQueue(self.lib, auto_pick_enabled=True,
                               auto_pick_random_chance=1.0)
        self.mock_sing = MagicMock()
        self.handler = SingCommandHandler(
            library=self.lib,
            queue=self.queue,
            sing_mode_factory=lambda: self.mock_sing,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cmd_list(self):
        reply = self.handler.handle("list", "user1")
        self.assertIn("Alpha Song", reply)
        self.assertIn("Beta Song", reply)

    def test_cmd_queue_empty(self):
        reply = self.handler.handle("queue", "user1")
        self.assertIn("empty", reply.lower())

    def test_cmd_now_playing_nothing(self):
        reply = self.handler.handle("np", "user1")
        self.assertIn("Nothing", reply)

    def test_cmd_request_not_found(self):
        reply = self.handler.handle("Nonexistent Song", "user1")
        self.assertIn("Couldn't find", reply)

    def test_cmd_stop(self):
        reply = self.handler.handle("stop", "user1")
        self.assertIn("Stopped", reply)

    def test_cmd_skip_nothing_playing(self):
        reply = self.handler.handle("skip", "user1")
        self.assertIn("Nothing", reply)

    def test_auto_pick_while_busy_does_not_replace_now_playing(self):
        current = self.queue.next()
        self.handler._busy = True
        reply = self.handler.handle("", "user1")
        self.assertIs(self.queue.now_playing, current)
        self.assertEqual(self.queue.size, 1)
        self.assertIn("Queued", reply)


if __name__ == "__main__":
    unittest.main()
