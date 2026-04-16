import unittest

from persona_manager import normalize_profile
from profile_engine import ProfileEngine
from prompt_assembly import CharacterProfileManager


class TestPersonaPipeline(unittest.TestCase):
    def test_normalize_profile_builds_canonical_persona_definition(self):
        profile = normalize_profile(
            {
                "character_name": "Nyx",
                "persona": "A sharp modular assistant for engineering work.",
                "traits": "analytical, direct, helpful",
                "response_style": "Technical",
                "verbosity": "Verbose",
                "greeting": "Nyx online.",
                "character_prompt": "You are Nyx, a precise engineering persona.",
            }
        )

        self.assertEqual(profile["character_name"], "Nyx")
        self.assertEqual(profile["persona_definition"]["name"], "Nyx")
        self.assertEqual(
            profile["persona_definition"]["interaction_style"]["response_style"],
            "Technical",
        )
        self.assertEqual(
            profile["persona_definition"]["interaction_style"]["verbosity"],
            "Verbose",
        )
        self.assertIn("analytical", profile["persona_definition"]["traits"])

    def test_character_context_uses_active_persona_name(self):
        manager = CharacterProfileManager(log_fn=lambda _msg: None)
        context = manager.build_character_context(
            {
                "character_name": "Nyx",
                "persona_preset": "serious",
                "persona": "A focused technical reviewer persona.",
                "character_prompt": "You are Nyx, a focused technical reviewer.",
                "persona_definition": {
                    "style_prompt": "Answer with sharp technical clarity.",
                    "collaboration_prompt": "Work like a senior reviewer and ask pointed follow-up questions.",
                },
            },
            include_greeting_instruction=True,
        )

        self.assertIn("Nyx", context)
        self.assertIn("Answer with sharp technical clarity.", context)
        self.assertIn("senior reviewer", context)
        self.assertNotIn("Sound like Revia", context)
        self.assertNotIn("you are Revia.", context.lower())

    def test_profile_engine_applies_behavior_preset_for_persona(self):
        engine = ProfileEngine(log_fn=lambda _msg: None)
        profile = engine.load(
            {
                "persona_preset": "serious",
                "character_name": "Nyx",
                "character_prompt": "You are Nyx, a focused technical reviewer.",
            }
        )

        self.assertEqual(profile["persona_preset"], "serious")
        self.assertAlmostEqual(profile["behavior"]["verbosity"], 0.85)
        self.assertAlmostEqual(profile["emotion"]["emotion_intensity"], 0.30)


if __name__ == "__main__":
    unittest.main()
