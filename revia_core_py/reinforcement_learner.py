"""
Reinforcement Learning Engine (RLE) — REVIA
=============================================
Lightweight online RL system that adjusts Revia's behavioral parameters based
on implicit reward signals gathered during conversations.

Architecture
------------
This is NOT a deep-RL / neural-network approach (that would require GPU training
loops and is overkill for tuning ~15 scalar parameters).  Instead we use:

  • **Contextual Multi-Armed Bandit** with Thompson Sampling
    — Each "arm" is a behavioral parameter configuration
    — Context = (emotion_state, topic_category, user_engagement_level)
    — Reward = composite signal from AVS score + user engagement + interruption penalty

  • **Exponential-decay memory** so Revia adapts to shifting user preferences
    over time (recent interactions weighted more heavily).

Reward Signals (implicit — no explicit thumbs-up needed)
---------------------------------------------------------
  +  High AVS composite score on accepted reply       (quality)
  +  User continues the conversation (follow-up msg)  (engagement)
  +  User laughs / positive emotion shift detected     (satisfaction)
  +  Long user messages (high effort = high interest)  (investment)
  -  User interrupts mid-response (barge-in)           (dissatisfaction)
  -  ALE detects loop / repetition                     (staleness)
  -  User sends "stop", "shut up", corrections         (rejection)
  -  Short user replies after long Revia output        (over-talking)
  -  Conversation abandoned (long idle after response)  (disengagement)

Tunable Parameters (the "action space")
----------------------------------------
  temperature        — LLM sampling temperature
  verbosity          — target response length multiplier
  humor_frequency    — how often to inject humor/quirks
  formality          — formal vs casual tone
  emoji_density      — emoji usage (0 = none, 1 = lots)
  proactivity        — how eager to initiate conversation
  interrupt_sensitivity — how easily interrupted by user
  topic_depth        — how deep vs surface-level to go

Storage
-------
  All learned parameters persist to a JSONL file so learning survives restarts.
  A rolling window of the last N interactions is kept in memory for fast updates.

Thread Safety
-------------
  All mutable state is protected by a single `threading.Lock`.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Parameters the RL engine can tune, with their allowed [min, max] ranges
_TUNABLE_PARAMS: dict[str, tuple[float, float]] = {
    "temperature":           (0.40, 1.30),
    "verbosity":             (0.20, 1.00),
    "humor_frequency":       (0.00, 0.80),
    "formality":             (0.10, 0.90),
    "emoji_density":         (0.00, 0.60),
    "proactivity":           (0.10, 0.90),
    "interrupt_sensitivity":  (0.20, 0.90),
    "topic_depth":           (0.20, 1.00),
}

# Default parameter values (profile-based - overridden on first load)
_DEFAULT_PARAMS: dict[str, float] = {
    "temperature":           0.80,
    "verbosity":             0.55,
    "humor_frequency":       0.30,
    "formality":             0.40,
    "emoji_density":         0.05,
    "proactivity":           0.50,
    "interrupt_sensitivity":  0.55,
    "topic_depth":           0.60,
}

# Reward signal weights
_REWARD_WEIGHTS: dict[str, float] = {
    "avs_quality":          0.30,   # AVS composite score → quality of output
    "user_engagement":      0.25,   # user follow-up, message length
    "emotion_shift":        0.15,   # positive emotion change = good
    "interruption_penalty": -0.15,  # barge-in = bad (negative weight)
    "loop_penalty":         -0.10,  # ALE loop detection = bad
    "correction_penalty":   -0.05,  # user corrected Revia = bad
}

# Learning rate for exponential moving average
_LEARNING_RATE = 0.08

# Exploration rate (epsilon-greedy) - chance of trying a random perturbation
_EXPLORATION_RATE = 0.12

# Decay factor for old experiences (per-interaction)
_DECAY_FACTOR = 0.995

# Maximum interaction history kept in memory
_MAX_HISTORY = 500

# Minimum interactions before RL starts adjusting (warmup)
_WARMUP_INTERACTIONS = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RewardSignal:
    """A single reward observation from one interaction."""
    timestamp:            float = 0.0
    avs_composite:        float = 0.0   # [0, 1] — answer quality score
    user_msg_length:      int   = 0     # word count of user's follow-up
    user_followed_up:     bool  = False  # did the user send another message?
    emotion_delta:        float = 0.0   # change in valence (-1 to +1)
    was_interrupted:      bool  = False  # user barged in
    loop_detected:        bool  = False  # ALE flagged repetition
    was_corrected:        bool  = False  # user sent a correction
    idle_abandoned:       bool  = False  # conversation went idle after response
    context_emotion:      str   = "neutral"
    context_topic:        str   = "general"
    context_engagement:   str   = "medium"  # low / medium / high
    params_used:          dict  = field(default_factory=dict)

    def composite_reward(self) -> float:
        """Compute a scalar reward in [-1, 1] from all signals."""
        r = 0.0
        # Quality
        r += _REWARD_WEIGHTS["avs_quality"] * self.avs_composite
        # Engagement
        engagement = 0.0
        if self.user_followed_up:
            engagement += 0.5
            if self.user_msg_length > 20:
                engagement += 0.3  # long follow-up = high interest
            elif self.user_msg_length > 8:
                engagement += 0.1
        if self.idle_abandoned:
            engagement -= 0.6
        r += _REWARD_WEIGHTS["user_engagement"] * max(-1, min(1, engagement))
        # Emotion shift
        r += _REWARD_WEIGHTS["emotion_shift"] * max(-1, min(1, self.emotion_delta))
        # Penalties (weights are already negative)
        if self.was_interrupted:
            r += _REWARD_WEIGHTS["interruption_penalty"]
        if self.loop_detected:
            r += _REWARD_WEIGHTS["loop_penalty"]
        if self.was_corrected:
            r += _REWARD_WEIGHTS["correction_penalty"]
        return max(-1.0, min(1.0, r))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["composite_reward"] = round(self.composite_reward(), 4)
        return d


@dataclass
class ParameterEstimate:
    """Thompson Sampling posterior for a single parameter."""
    name:    str
    mu:      float = 0.5   # estimated mean reward at current value
    sigma:   float = 0.15  # uncertainty
    value:   float = 0.5   # current parameter value
    min_val: float = 0.0
    max_val: float = 1.0
    n:       int   = 0     # number of updates

    def sample(self) -> float:
        """Draw from the posterior (Gaussian approximation)."""
        return random.gauss(self.mu, self.sigma)

    def update(self, reward: float, lr: float = _LEARNING_RATE):
        """Bayesian-ish update: shift mu toward observed reward, shrink sigma."""
        self.n += 1
        self.mu  = self.mu + lr * (reward - self.mu)
        # Sigma shrinks with observations but never below a floor
        self.sigma = max(0.02, self.sigma * (1 - lr * 0.5))


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ReinforcementLearner:
    """
    Online reinforcement learning engine for Revia's behavioral parameters.

    Usage::

        rl = ReinforcementLearner(data_dir="revia_core_py/data")
        rl.load()

        # Before generating a reply, get RL-adjusted params:
        params = rl.get_params()
        # params = {"temperature": 0.82, "verbosity": 0.6, ...}

        # After the reply + user reaction, record reward:
        rl.record_reward(RewardSignal(
            avs_composite=0.78,
            user_followed_up=True,
            user_msg_length=25,
            ...
        ))
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        profile_engine=None,
        log_fn=None,
    ):
        self._lock = threading.Lock()
        self._data_dir = Path(data_dir)
        self._pe = profile_engine
        self._log = log_fn or _log.info
        self._history: deque[RewardSignal] = deque(maxlen=_MAX_HISTORY)
        self._params: dict[str, ParameterEstimate] = {}
        self._interaction_count = 0
        self._cumulative_reward = 0.0
        self._last_reward: float = 0.0
        self._enabled = True
        self._session_start = time.monotonic()

        # Initialize parameter estimates from defaults
        for name, (lo, hi) in _TUNABLE_PARAMS.items():
            default_val = _DEFAULT_PARAMS.get(name, (lo + hi) / 2)
            self._params[name] = ParameterEstimate(
                name=name,
                mu=0.5,
                sigma=0.15,
                value=default_val,
                min_val=lo,
                max_val=hi,
            )

    # Public API

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = bool(val)

    def get_params(self) -> dict[str, float]:
        """Return current RL-adjusted parameter values.

        During warmup, returns profile defaults.  After warmup,
        may apply exploration (small random perturbation) with probability
        _EXPLORATION_RATE.
        """
        with self._lock:
            if not self._enabled or self._interaction_count < _WARMUP_INTERACTIONS:
                return {name: est.value for name, est in self._params.items()}

            result = {}
            exploring = random.random() < _EXPLORATION_RATE
            for name, est in self._params.items():
                if exploring:
                    # Explore: perturb current value by a small random amount
                    delta = random.gauss(0, 0.05)
                    new_val = est.value + delta
                else:
                    # Exploit: use current best estimate
                    new_val = est.value
                result[name] = max(est.min_val, min(est.max_val, new_val))
            return result

    def record_reward(self, signal: RewardSignal):
        """Record a reward signal and update parameter estimates.

        Call this after each completed interaction (user message → Revia reply
        → user reaction observed).
        """
        with self._lock:
            if not self._enabled:
                return

            signal.timestamp = signal.timestamp or time.time()
            signal.params_used = {n: e.value for n, e in self._params.items()}
            reward = signal.composite_reward()
            self._history.append(signal)
            self._interaction_count += 1
            self._cumulative_reward += reward
            self._last_reward = reward

            # Skip parameter updates during warmup
            if self._interaction_count < _WARMUP_INTERACTIONS:
                self._log(
                    f"[RL] Warmup {self._interaction_count}/{_WARMUP_INTERACTIONS} "
                    f"| reward={reward:.3f}"
                )
                return

            # Update each parameter estimate
            for name, est in self._params.items():
                est.update(reward)

                # Adaptive step: if reward is positive, nudge the parameter
                # in the direction that produced it; if negative, nudge away
                if abs(reward) > 0.05:
                    # Use Thompson Sampling: sample from posterior, blend toward
                    # the sampled direction
                    sampled = est.sample()
                    direction = 1.0 if sampled > est.mu else -1.0
                    step = _LEARNING_RATE * reward * direction * 0.3
                    est.value = max(
                        est.min_val,
                        min(est.max_val, est.value + step)
                    )

            self._log(
                f"[RL] Update #{self._interaction_count} | reward={reward:.3f} "
                f"| cumulative={self._cumulative_reward:.2f} "
                f"| temp={self._params['temperature'].value:.3f} "
                f"| verbose={self._params['verbosity'].value:.3f}"
            )

            # Auto-save periodically
            if self._interaction_count % 10 == 0:
                self._save_unlocked()

    @staticmethod
    def _safe_float(value, default: float) -> float:
        """Convert a profile value to float, returning *default* for non-numeric
        strings like ``"Normal"`` or ``"High"``."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def sync_from_profile(self, profile: dict):
        """Sync initial parameter values from the active profile.

        Called once at startup or on profile swap.  Only overwrites params
        that haven't been significantly learned yet (low observation count).
        """
        _sf = self._safe_float
        with self._lock:
            mapping = {
                "temperature":          _sf(profile.get("temperature"),          0.8),
                "verbosity":            _sf(profile.get("verbosity"),            0.55),
                "humor_frequency":      _sf(profile.get("humor_frequency"),      0.3),
                "formality":            _sf(profile.get("formality"),            0.4),
                "emoji_density":        _sf(profile.get("emoji_density"),        0.05),
                "proactivity":          _sf(profile.get("proactivity"),          0.5),
                "interrupt_sensitivity": _sf(profile.get("interrupt_sensitivity"), 0.55),
                "topic_depth":          _sf(profile.get("topic_depth"),          0.6),
            }
            for name, prof_val in mapping.items():
                if name in self._params:
                    est = self._params[name]
                    # Only override if we haven't learned much yet
                    if est.n < 5:
                        est.value = max(est.min_val, min(est.max_val, prof_val))
            self._log("[RL] Synced parameters from profile")

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of RL statistics for the telemetry dashboard."""
        with self._lock:
            params_snapshot = {}
            for name, est in self._params.items():
                params_snapshot[name] = {
                    "value":  round(est.value, 4),
                    "mu":     round(est.mu, 4),
                    "sigma":  round(est.sigma, 4),
                    "n":      est.n,
                }
            avg_reward = (
                self._cumulative_reward / self._interaction_count
                if self._interaction_count > 0 else 0.0
            )
            return {
                "enabled":           self._enabled,
                "interaction_count": self._interaction_count,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "average_reward":    round(avg_reward, 4),
                "last_reward":       round(self._last_reward, 4),
                "warmup_complete":   self._interaction_count >= _WARMUP_INTERACTIONS,
                "exploration_rate":  _EXPLORATION_RATE,
                "learning_rate":     _LEARNING_RATE,
                "parameters":        params_snapshot,
            }

    def recent_rewards(self, n: int = 20) -> list[dict]:
        """Return the last N reward signals for debugging."""
        with self._lock:
            recent = list(self._history)[-n:]
            return [s.to_dict() for s in recent]

    # Persistence

    def load(self):
        """Load learned parameters from disk."""
        path = self._data_dir / "rl_parameters.json"
        if not path.exists():
            self._log("[RL] No saved parameters found, starting fresh")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._interaction_count = int(data.get("interaction_count", 0))
                self._cumulative_reward = float(data.get("cumulative_reward", 0.0))
                saved_params = data.get("parameters", {})
                for name, est_data in saved_params.items():
                    if name in self._params:
                        est = self._params[name]
                        est.value = float(est_data.get("value", est.value))
                        est.mu    = float(est_data.get("mu", est.mu))
                        est.sigma = float(est_data.get("sigma", est.sigma))
                        est.n     = int(est_data.get("n", est.n))
            self._log(
                f"[RL] Loaded parameters | interactions={self._interaction_count} "
                f"| cumulative_reward={self._cumulative_reward:.2f}"
            )
        except Exception as exc:
            self._log(f"[RL] Failed to load parameters: {exc}")

    def save(self):
        """Persist learned parameters to disk."""
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self):
        """Internal save — caller must hold self._lock."""
        path = self._data_dir / "rl_parameters.json"
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "interaction_count":  self._interaction_count,
                "cumulative_reward":  round(self._cumulative_reward, 4),
                "last_reward":        round(self._last_reward, 4),
                "saved_at":           time.strftime("%Y-%m-%dT%H:%M:%S"),
                "parameters": {
                    name: {
                        "value": round(est.value, 6),
                        "mu":    round(est.mu, 6),
                        "sigma": round(est.sigma, 6),
                        "n":     est.n,
                    }
                    for name, est in self._params.items()
                },
            }
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_path.replace(path)
        except Exception as exc:
            self._log(f"[RL] Failed to save parameters: {exc}")

    # History analysis

    def get_reward_trend(self, window: int = 20) -> dict[str, float]:
        """Compute recent reward trend for dashboard display."""
        with self._lock:
            recent = list(self._history)[-window:]
            if len(recent) < 2:
                return {"trend": 0.0, "avg_recent": 0.0, "avg_overall": 0.0}
            rewards = [s.composite_reward() for s in recent]
            half = len(rewards) // 2
            first_half = sum(rewards[:half]) / max(1, half)
            second_half = sum(rewards[half:]) / max(1, len(rewards) - half)
            avg_overall = (
                self._cumulative_reward / self._interaction_count
                if self._interaction_count > 0 else 0.0
            )
            return {
                "trend":       round(second_half - first_half, 4),
                "avg_recent":  round(sum(rewards) / len(rewards), 4),
                "avg_overall": round(avg_overall, 4),
            }

    def reset(self):
        """Reset all learned parameters to defaults.  Use with caution."""
        with self._lock:
            self._history.clear()
            self._interaction_count = 0
            self._cumulative_reward = 0.0
            self._last_reward = 0.0
            for name, (lo, hi) in _TUNABLE_PARAMS.items():
                default_val = _DEFAULT_PARAMS.get(name, (lo + hi) / 2)
                est = self._params[name]
                est.value = default_val
                est.mu = 0.5
                est.sigma = 0.15
                est.n = 0
            self._log("[RL] Parameters reset to defaults")
