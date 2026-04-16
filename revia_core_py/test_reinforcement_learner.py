"""Tests for the ReinforcementLearner module."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from reinforcement_learner import (
    ReinforcementLearner,
    RewardSignal,
    _TUNABLE_PARAMS,
    _WARMUP_INTERACTIONS,
)


class TestRewardSignal(unittest.TestCase):
    def test_composite_reward_default(self):
        """Default signal (all zeros) should produce ~0 reward."""
        sig = RewardSignal()
        self.assertAlmostEqual(sig.composite_reward(), 0.0, places=2)

    def test_positive_reward(self):
        """High AVS + engagement should yield positive reward."""
        sig = RewardSignal(
            avs_composite=0.9,
            user_followed_up=True,
            user_msg_length=30,
            emotion_delta=0.3,
        )
        reward = sig.composite_reward()
        self.assertGreater(reward, 0.3)

    def test_negative_reward(self):
        """Interruption + loop + correction should yield negative reward."""
        sig = RewardSignal(
            avs_composite=0.2,
            was_interrupted=True,
            loop_detected=True,
            was_corrected=True,
            idle_abandoned=True,
        )
        reward = sig.composite_reward()
        self.assertLess(reward, 0.0)

    def test_reward_clamped(self):
        """Reward should always be in [-1, 1]."""
        sig = RewardSignal(
            avs_composite=1.0,
            user_followed_up=True,
            user_msg_length=100,
            emotion_delta=1.0,
        )
        self.assertLessEqual(sig.composite_reward(), 1.0)
        self.assertGreaterEqual(sig.composite_reward(), -1.0)

    def test_to_dict(self):
        sig = RewardSignal(avs_composite=0.75)
        d = sig.to_dict()
        self.assertIn("composite_reward", d)
        self.assertIn("avs_composite", d)


class TestReinforcementLearner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.rl = ReinforcementLearner(data_dir=self.tmpdir, log_fn=lambda msg: None)

    def test_get_params_returns_all(self):
        params = self.rl.get_params()
        for name in _TUNABLE_PARAMS:
            self.assertIn(name, params)

    def test_params_within_bounds(self):
        params = self.rl.get_params()
        for name, val in params.items():
            lo, hi = _TUNABLE_PARAMS[name]
            self.assertGreaterEqual(val, lo, f"{name} below min")
            self.assertLessEqual(val, hi, f"{name} above max")

    def test_warmup_no_adjustment(self):
        """During warmup, params should stay at defaults."""
        initial = self.rl.get_params()
        for i in range(_WARMUP_INTERACTIONS - 1):
            self.rl.record_reward(RewardSignal(avs_composite=0.9, user_followed_up=True))
        after = self.rl.get_params()
        # During warmup, values should be unchanged
        for name in _TUNABLE_PARAMS:
            self.assertAlmostEqual(initial[name], after[name], places=4)

    def test_learning_updates_params(self):
        """After warmup, recording rewards should shift parameters."""
        initial = self.rl.get_params()
        # Fill warmup
        for _ in range(_WARMUP_INTERACTIONS + 5):
            self.rl.record_reward(RewardSignal(
                avs_composite=0.9,
                user_followed_up=True,
                user_msg_length=25,
                emotion_delta=0.2,
            ))
        stats = self.rl.get_stats()
        self.assertTrue(stats["warmup_complete"])
        self.assertGreater(stats["average_reward"], 0)

    def test_save_and_load(self):
        """Learned parameters should persist across save/load."""
        for _ in range(15):
            self.rl.record_reward(RewardSignal(avs_composite=0.8))
        self.rl.save()

        # Verify file exists
        param_file = Path(self.tmpdir) / "rl_parameters.json"
        self.assertTrue(param_file.exists())

        # Load into new instance
        rl2 = ReinforcementLearner(data_dir=self.tmpdir, log_fn=lambda msg: None)
        rl2.load()
        self.assertEqual(rl2._interaction_count, self.rl._interaction_count)

    def test_reset(self):
        for _ in range(20):
            self.rl.record_reward(RewardSignal(avs_composite=0.9))
        self.rl.reset()
        stats = self.rl.get_stats()
        self.assertEqual(stats["interaction_count"], 0)
        self.assertAlmostEqual(stats["cumulative_reward"], 0.0)

    def test_sync_from_profile(self):
        profile = {"temperature": 0.95, "verbosity": 0.8, "humor_frequency": 0.5}
        self.rl.sync_from_profile(profile)
        params = self.rl.get_params()
        self.assertAlmostEqual(params["temperature"], 0.95, places=2)
        self.assertAlmostEqual(params["verbosity"], 0.8, places=2)

    def test_get_stats(self):
        stats = self.rl.get_stats()
        self.assertIn("enabled", stats)
        self.assertIn("interaction_count", stats)
        self.assertIn("parameters", stats)
        self.assertIn("temperature", stats["parameters"])

    def test_reward_trend(self):
        trend = self.rl.get_reward_trend()
        self.assertIn("trend", trend)
        self.assertIn("avg_recent", trend)

    def test_disabled_skips_updates(self):
        self.rl.enabled = False
        self.rl.record_reward(RewardSignal(avs_composite=0.9))
        self.assertEqual(self.rl._interaction_count, 0)

    def test_recent_rewards(self):
        self.rl.record_reward(RewardSignal(avs_composite=0.7))
        recent = self.rl.recent_rewards(5)
        self.assertEqual(len(recent), 1)
        self.assertAlmostEqual(recent[0]["avs_composite"], 0.7)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
