"""Focused tests for TTS text handling."""

import unittest

from tts_backend import _strip_leading_style_directives


class TestTTSStyleSanitizer(unittest.TestCase):
    def test_removes_neutral_style_hint_from_spoken_text(self):
        text = "[Speak naturally with balanced pacing and clear tone] Hello there."
        self.assertEqual(
            _strip_leading_style_directives(text),
            "Hello there.",
        )

    def test_removes_sing_style_hint_from_lyric_text(self):
        text = "[Sing this line melodically, in a high register] Lonely"
        self.assertEqual(_strip_leading_style_directives(text), "Lonely")

    def test_keeps_normal_bracketed_user_text(self):
        text = "[aside] this is part of the message"
        self.assertEqual(_strip_leading_style_directives(text), text)


if __name__ == "__main__":
    unittest.main()
