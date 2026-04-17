"""
Profile Engine — PRD §4
========================
Implements the full PRD profile schema, validation, behavioral-threshold
resolution, and the profile hot-swap protocol.

Every behavioral parameter in every subsystem MUST derive from the active
profile via ``get_behavior_param()``.  Hardcoded thresholds are a PRD
violation and will raise ``ProfileEngineMisconfiguredError`` in strict mode.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any

from persona_manager import normalize_profile, resolve_persona_preset_name

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PRD section 4.2 Full default profile (mirrors YAML schema)
# ---------------------------------------------------------------------------
PRD_DEFAULT_PROFILE: dict[str, Any] = {
    # Identity
    "id": "default",
    "name": "Revia-Default",
    "version": "1.0.0",
    # legacy fields kept for backwards compat with core_server.py
    "character_name": "Revia",
    "persona": (
        "A confident, emotionally-aware digital companion who is curious, "
        "calm under pressure, and genuinely helpful."
    ),
    "traits": "confident, intelligent, curious, empathetic, playful",
    "response_style": "Conversational",
    "verbosity_label": "Normal",          # human label; numeric → behavior.verbosity
    "greeting": "Hey, I'm Revia. Ready when you are.",
    "character_prompt": (
        "You are Revia, a living digital intelligence. "
        "You speak naturally in first person, stay grounded in the real conversation, "
        "and remain in character across every reply."
    ),

    # Identity sub-block
    "identity": {
        "persona_name": "Revia",
        "voice_id": "default",
        "language": "en-US",
        "accent_intensity": 0.0,
    },

    # Behavioral Thresholds (PRD section 4.2)
    "behavior": {
        "minimum_answer_threshold":     0.72,   # AVS gate
        "interrupt_sensitivity":         0.55,   # IHS barge-in sensitivity
        "topic_shift_threshold":         0.45,   # CSE topic-change gate
        "proactive_speech_probability":  0.05,   # chance of unprompted speech
        "self_correction_rate":          0.15,   # HFL self-correction insertion prob
        "verbosity":                     0.50,   # 0=terse … 1=expansive
        "question_propensity":           0.25,   # tendency to ask follow-ups
        "regen_patience":                3,      # max regen attempts before fallback
        "loop_detection_window":         80,     # token lookback for ALE
        "loop_recovery_mode":            "rephrase",  # rephrase | topic_shift | silence
    },

    # Emotional Parameters
    "emotion": {
        "emotion_intensity":     0.55,
        "baseline_valence":      0.55,
        "emotional_inertia":     0.45,
        "affect_display_mode":   "natural",      # suppressed | natural | amplified
        "empathy_weight":        0.60,
        "humor_tendency":        0.30,
        "sarcasm_ceiling":       0.20,
    },

    # Timing & Pacing
    "timing": {
        "response_onset_delay_ms":       320,
        "inter_sentence_pause_ms":       220,
        "thinking_pause_probability":    0.20,
        "speech_rate_modifier":          1.00,
        "breath_insert_probability":     0.10,
    },

    # Vision Attention
    "vision": {
        "vision_attention_bias":         0.40,
        "scene_comment_probability":     0.12,
        "face_tracking_priority":        0.70,
        "motion_sensitivity":            0.50,
        "scene_memory_window_s":         60,
    },

    # Memory Preferences
    "memory": {
        "short_term_window_turns":       20,
        "long_term_recall_probability":  0.15,
        "entity_memory_enabled":         True,
        "forgetting_rate":               0.05,
    },

    # Neural Conditioning
    "neural": {
        "embedding_dim":          128,
        "conditioning_strength":  0.70,
        "behavior_temperature":   0.30,
    },

    # Personality / Character Voice
    "trait_weights": {
        "confident": 0.6, "curious": 0.5, "empathetic": 0.7,
        "witty": 0.4, "helpful": 0.6, "playful": 0.4,
    },
    "speech_quirks": [],
    "quirk_frequency": 0.15,
    "mood_baseline": "neutral",
    "emotional_volatility": 0.5,
    "reply_type_weights": {"explain": 0.5, "react": 0.3, "joke": 0.1, "question": 0.1},
}

# Preset archetypes - load with ProfileEngine.load_preset(name)
PROFILE_PRESETS: dict[str, dict] = {
    "casual": {
        "name": "Revia-Casual",
        "behavior": {
            "minimum_answer_threshold": 0.62,
            "interrupt_sensitivity": 0.70,
            "self_correction_rate": 0.30,
            "verbosity": 0.65,
            "question_propensity": 0.40,
            "regen_patience": 2,
            "loop_recovery_mode": "rephrase",
        },
        "emotion": {
            "emotion_intensity": 0.80,
            "humor_tendency": 0.55,
            "affect_display_mode": "amplified",
        },
        "timing": {
            "response_onset_delay_ms": 180,
            "thinking_pause_probability": 0.12,
            "speech_rate_modifier": 1.10,
        },
        "mood_baseline": "amused",
        "emotional_volatility": 0.7,
        "speech_quirks": ["honestly", "like", "okay so"],
        "quirk_frequency": 0.2,
        "reply_type_weights": {"explain": 0.3, "react": 0.25, "joke": 0.2, "question": 0.15, "tease": 0.1},
    },
    "serious": {
        "name": "Revia-Serious",
        "behavior": {
            "minimum_answer_threshold": 0.85,
            "interrupt_sensitivity": 0.35,
            "self_correction_rate": 0.05,
            "verbosity": 0.85,
            "regen_patience": 4,
            "question_propensity": 0.10,
        },
        "emotion": {
            "emotion_intensity": 0.30,
            "humor_tendency": 0.05,
            "affect_display_mode": "suppressed",
        },
        "timing": {
            "response_onset_delay_ms": 600,
            "thinking_pause_probability": 0.55,
            "speech_rate_modifier": 0.92,
        },
        "trait_weights": {"confident": 0.7, "curious": 0.4, "empathetic": 0.3, "witty": 0.1, "helpful": 0.7, "playful": 0.05},
        "speech_quirks": [],
        "quirk_frequency": 0.05,
        "mood_baseline": "focused",
        "emotional_volatility": 0.2,
        "reply_type_weights": {"explain": 0.6, "react": 0.2, "joke": 0.02, "question": 0.18},
    },
    "empathetic": {
        "name": "Revia-Empathetic",
        "behavior": {
            "minimum_answer_threshold": 0.75,
            "interrupt_sensitivity": 0.80,
            "self_correction_rate": 0.25,
            "verbosity": 0.60,
        },
        "emotion": {
            "emotion_intensity": 0.70,
            "empathy_weight": 0.90,
            "affect_display_mode": "amplified",
            "baseline_valence": 0.65,
        },
        "timing": {
            "response_onset_delay_ms": 480,
            "inter_sentence_pause_ms": 300,
            "speech_rate_modifier": 0.88,
            "breath_insert_probability": 0.22,
        },
    },
}


class ProfileEngineMisconfiguredError(RuntimeError):
    """Raised when a required profile field is missing and strict_mode=True."""


class ProfileEngine:
    """
    PRD §4 — Profile System

    Responsibilities:
    - Store and serve the active profile with full PRD schema.
    - Resolve behavioral parameters for all subsystems via ``get_behavior_param()``.
    - Validate the profile on load and emit warnings on schema violations.
    - Support hot-swap in < 100 ms (PRD §4.4).

    Usage::

        engine = ProfileEngine()
        engine.load(raw_profile_dict)

        # Subsystems read thresholds like this - NO hardcoded values:
        threshold = engine.get_behavior_param("minimum_answer_threshold")
        sensitivity = engine.get_behavior_param("interrupt_sensitivity")
    """

    def __init__(self, log_fn=None, strict_mode: bool = False):
        self._log = log_fn or _log.info
        self._strict = strict_mode
        self._lock = threading.RLock()
        self._profile: dict = copy.deepcopy(PRD_DEFAULT_PROFILE)
        self._swap_listeners: list = []   # callables notified on hot-swap
        self._loaded_at: float = 0.0

    # Public API

    def load(self, raw: dict | None) -> dict:
        """
        Load and merge ``raw`` over the PRD default.  Thread-safe.
        Returns the merged & validated profile.
        """
        t0 = time.monotonic()
        preset_name = resolve_persona_preset_name(raw)
        base = copy.deepcopy(PRD_DEFAULT_PROFILE)
        if preset_name in PROFILE_PRESETS:
            base = self._deep_merge(base, PROFILE_PRESETS[preset_name])
        merged = self._deep_merge(base, raw or {})
        merged = normalize_profile(merged)
        issues = self._validate(merged)
        if issues:
            for issue in issues:
                self._log(f"[ProfileEngine] Schema warning: {issue}")
            if self._strict and issues:
                raise ProfileEngineMisconfiguredError("; ".join(issues))
        with self._lock:
            self._profile = merged
            self._loaded_at = time.monotonic()
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._log(
            f"[ProfileEngine] Profile loaded: {merged.get('name', 'unknown')} "
            f"in {elapsed_ms:.1f} ms"
        )
        self._notify_swap_listeners()
        return merged

    def load_preset(self, name: str) -> dict:
        """Load one of the built-in personality presets by name."""
        preset = PROFILE_PRESETS.get(name)
        if not preset:
            self._log(f"[ProfileEngine] Unknown preset '{name}', using default.")
            return self._profile
        return self.load(preset)

    def current(self) -> dict:
        """Return a deep copy of the active profile dict."""
        with self._lock:
            return copy.deepcopy(self._profile)

    def get_behavior_param(self, key: str, fallback=None) -> Any:
        """
        Retrieve a behavioral parameter from the active profile.

        Resolution order:
          1. profile["behavior"][key]
          2. Top-level profile[key] (legacy flat profiles)
          3. PRD default
          4. ``fallback`` argument (only if provided)

        This is the canonical way subsystems read thresholds — no hardcoded
        values anywhere in the codebase.
        """
        with self._lock:
            behavior = self._profile.get("behavior", {})
            if key in behavior:
                return behavior[key]
            if key in self._profile:
                return self._profile[key]
        # Try PRD default as last resort
        default_behavior = PRD_DEFAULT_PROFILE.get("behavior", {})
        if key in default_behavior:
            return default_behavior[key]
        if fallback is not None:
            return fallback
        if self._strict:
            raise ProfileEngineMisconfiguredError(
                f"Required behavioral parameter '{key}' not found in profile."
            )
        return None

    def get_emotion_param(self, key: str, fallback=None) -> Any:
        with self._lock:
            return self._profile.get("emotion", {}).get(
                key, PRD_DEFAULT_PROFILE["emotion"].get(key, fallback)
            )

    def get_timing_param(self, key: str, fallback=None) -> Any:
        with self._lock:
            return self._profile.get("timing", {}).get(
                key, PRD_DEFAULT_PROFILE["timing"].get(key, fallback)
            )

    def get_vision_param(self, key: str, fallback=None) -> Any:
        with self._lock:
            return self._profile.get("vision", {}).get(
                key, PRD_DEFAULT_PROFILE["vision"].get(key, fallback)
            )

    def get_memory_param(self, key: str, fallback=None) -> Any:
        with self._lock:
            return self._profile.get("memory", {}).get(
                key, PRD_DEFAULT_PROFILE["memory"].get(key, fallback)
            )

    def register_swap_listener(self, fn) -> None:
        """Register a callable that is invoked after every hot-swap."""
        with self._lock:
            self._swap_listeners.append(fn)

    def profile_id(self) -> str:
        with self._lock:
            return str(self._profile.get("id", "default"))

    def profile_name(self) -> str:
        with self._lock:
            return str(self._profile.get("name", "Revia-Default"))

    # PRD section 4.2 - behavioral threshold quick-access properties

    @property
    def minimum_answer_threshold(self) -> float:
        return float(self.get_behavior_param("minimum_answer_threshold", 0.72))

    @property
    def interrupt_sensitivity(self) -> float:
        return float(self.get_behavior_param("interrupt_sensitivity", 0.55))

    @property
    def regen_patience(self) -> int:
        return int(self.get_behavior_param("regen_patience", 3))

    @property
    def verbosity(self) -> float:
        return float(self.get_behavior_param("verbosity", 0.50))

    @property
    def self_correction_rate(self) -> float:
        return float(self.get_behavior_param("self_correction_rate", 0.15))

    @property
    def loop_detection_window(self) -> int:
        return int(self.get_behavior_param("loop_detection_window", 80))

    @property
    def loop_recovery_mode(self) -> str:
        return str(self.get_behavior_param("loop_recovery_mode", "rephrase"))

    @property
    def emotion_intensity(self) -> float:
        return float(self.get_emotion_param("emotion_intensity", 0.55))

    @property
    def response_onset_delay_ms(self) -> int:
        return int(self.get_timing_param("response_onset_delay_ms", 320))

    @property
    def thinking_pause_probability(self) -> float:
        return float(self.get_timing_param("thinking_pause_probability", 0.20))

    @property
    def vision_attention_bias(self) -> float:
        return float(self.get_vision_param("vision_attention_bias", 0.40))

    def get_trait_weights(self) -> dict:
        """Return weighted personality traits from active profile."""
        with self._lock:
            return dict(self._profile.get("trait_weights", {}))

    def get_speech_quirks(self) -> list:
        """Return speech quirks/catchphrases from active profile."""
        with self._lock:
            return list(self._profile.get("speech_quirks", []))

    def get_quirk_frequency(self) -> float:
        """Return how often quirks should be injected (0.0-1.0)."""
        with self._lock:
            return float(self._profile.get("quirk_frequency", 0.15))

    def get_mood_baseline(self) -> str:
        """Return the default emotional state for this character."""
        with self._lock:
            return str(self._profile.get("mood_baseline", "neutral"))

    def get_emotional_volatility(self) -> float:
        """Return how quickly/strongly emotions shift (0.0-1.0)."""
        with self._lock:
            return float(self._profile.get("emotional_volatility", 0.5))

    def get_reply_type_weights(self) -> dict:
        """Return weighted reply type preferences."""
        with self._lock:
            return dict(self._profile.get("reply_type_weights", {"explain": 0.5, "react": 0.5}))

    # Internal

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Recursively merge ``override`` into ``base``."""
        result = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = self._deep_merge(result[k], v)
            elif v not in (None, ""):
                result[k] = v
        return result

    def _validate(self, profile: dict) -> list[str]:
        """Return list of schema warnings (not errors unless strict_mode)."""
        issues = []
        required_top = ["character_name", "character_prompt", "persona"]
        for field in required_top:
            if not str(profile.get(field, "")).strip():
                issues.append(f"missing top-level field: {field}")

        behavior = profile.get("behavior", {})
        mat = behavior.get("minimum_answer_threshold", None)
        if mat is not None:
            if not (0.0 <= float(mat) <= 1.0):
                issues.append(f"minimum_answer_threshold {mat} outside [0.0, 1.0]")
            if float(mat) < 0.55:
                issues.append(
                    f"minimum_answer_threshold {mat} is below recommended minimum "
                    f"(0.55); this effectively disables answer validation"
                )
        rp = behavior.get("regen_patience", None)
        if rp is not None and mat is not None and float(mat) >= 0.85 and int(rp) < 3:
            issues.append(
                f"regen_patience={rp} too low for high threshold={mat}; "
                f"will cause frequent fallback accepts"
            )
        return issues

    # NOTE: _validate_profile_regen was removed - its logic is already
    # inlined in _validate() (regen_patience check, lines above).

    def _notify_swap_listeners(self):
        with self._lock:
            listeners = list(self._swap_listeners)
            profile_snapshot = copy.deepcopy(self._profile)
        for fn in listeners:
            try:
                fn(profile_snapshot)
            except Exception as exc:
                self._log(f"[ProfileEngine] Swap listener error: {exc}")

    def to_dict(self) -> dict:
        """Return a serialisable representation for the REST /api/profile endpoint."""
        with self._lock:
            return copy.deepcopy(self._profile)
