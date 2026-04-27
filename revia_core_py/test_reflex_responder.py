import unittest

from reflex_responder import get_reflex_reply


class _Memory:
    def get_short_term(self, limit=12):
        return [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "I'm here."},
        ]


class TestReflexResponder(unittest.TestCase):
    def test_greeting_uses_profile_greeting(self):
        reply = get_reflex_reply("hello", profile={"greeting": "I'm here. What happened?"})

        self.assertIsNotNone(reply)
        self.assertEqual(reply.reason, "simple_greeting")
        self.assertEqual(reply.text, "I'm here. What happened?")

    def test_quiet_request_marks_quiet(self):
        reply = get_reflex_reply("wait")

        self.assertIsNotNone(reply)
        self.assertTrue(reply.quiet_request)

    def test_repeat_uses_latest_assistant_message(self):
        reply = get_reflex_reply("repeat", memory_store=_Memory())

        self.assertIsNotNone(reply)
        self.assertEqual(reply.reason, "repeat_last_assistant")
        self.assertEqual(reply.text, "I'm here.")


if __name__ == "__main__":
    unittest.main()
