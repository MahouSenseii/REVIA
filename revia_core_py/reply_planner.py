"""
Reply Planning System (RPS) — PRD §9
=======================================
4-stage pipeline that turns a user utterance into a validated, HFL-processed
reply candidate ready for delivery.

Stages (PRD §9)
----------------
  1. Intent Parse        — extract primary intent + entity slots
  2. Emotional Routing   — determine required tone / affect mode
  3. Candidate Generation— call LLM backend; collect up to regen_patience+1 candidates
  4. AVS Ranking         — score via AVS; apply ALE check; HFL post-process winner

The planner is intentionally *synchronous* so the async pipeline can wrap it
in a thread without dealing with nested event-loops.  All subsystem instances
are injected for testability.

All thresholds come from ProfileEngine.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from .answer_validation import AnswerValidationSystem, AVSResult
    from .anti_loop_engine   import AntiLoopEngine, ALEReport
    from .human_feel_layer   import HumanFeelLayer, HFLResult
except ImportError:
    # Direct (non-package) execution context — used by core_server.py and tests
    from answer_validation import AnswerValidationSystem, AVSResult    # type: ignore[no-redef]
    from anti_loop_engine   import AntiLoopEngine, ALEReport            # type: ignore[no-redef]
    from human_feel_layer   import HumanFeelLayer, HFLResult            # type: ignore[no-redef]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IntentFrame:
    """Lightweight intent/entity extraction result."""
    primary_intent: str              = "general_query"
    entities:       dict[str, str]   = field(default_factory=dict)
    question_words: list[str]        = field(default_factory=list)
    sentiment:      str              = "neutral"   # positive / neutral / negative

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "entities":       self.entities,
            "question_words": self.question_words,
            "sentiment":      self.sentiment,
        }


@dataclass
class ReplyPlan:
    """Complete record for one full RPS planning cycle."""
    user_utterance:   str
    intent:           IntentFrame
    emotion_label:    str
    candidates:       list[str]               = field(default_factory=list)
    avs_results:      list[AVSResult]         = field(default_factory=list)
    ale_reports:      list[ALEReport]         = field(default_factory=list)
    hfl_result:       HFLResult | None        = None
    final_reply:      str                     = ""
    fallback_accept:  bool                    = False
    total_elapsed_ms: float                   = 0.0
    notes:            list[str]               = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_utterance":   self.user_utterance[:120],
            "intent":           self.intent.to_dict(),
            "emotion_label":    self.emotion_label,
            "candidates_count": len(self.candidates),
            "avs_results":      [r.to_dict() for r in self.avs_results],
            "ale_triggered":    any(r.triggered for r in self.ale_reports),
            "hfl":              self.hfl_result.to_dict() if self.hfl_result else {},
            "final_reply_len":  len(self.final_reply.split()),
            "fallback_accept":  self.fallback_accept,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "notes":            self.notes,
        }


# ---------------------------------------------------------------------------
# Reply planner
# ---------------------------------------------------------------------------

LLMGenerateFn = Callable[[str, str, str], str]
"""
Signature: (system_prompt: str, user_text: str, repair_hint: str) -> reply: str

``repair_hint`` is empty on first attempt; on regen it contains the ALE/AVS
repair phrase so the LLM can adjust.
"""


class ReplyPlanner:
    """
    PRD §9 — Reply Planning System

    Usage::

        def my_llm(system_prompt, user_text, repair_hint) -> str:
            # Call your LLM backend here
            ...

        planner = ReplyPlanner(
            profile_engine = pe,
            llm_fn         = my_llm,
            avs            = AnswerValidationSystem(pe),
            ale            = AntiLoopEngine(pe),
            hfl            = HumanFeelLayer(pe),
        )

        plan = planner.plan(
            user_utterance = user_text,
            system_prompt  = system_prompt,
            emotion_label  = "neutral",
            recent_replies = last_n_replies,
        )
        deliver(plan.final_reply)
        set_prosody(plan.hfl_result.prosody)
    """

    def __init__(
        self,
        profile_engine=None,
        llm_fn: LLMGenerateFn | None = None,
        avs: AnswerValidationSystem | None = None,
        ale: AntiLoopEngine | None = None,
        hfl: HumanFeelLayer | None = None,
    ):
        self._pe  = profile_engine
        self._llm = llm_fn or self._stub_llm
        self._avs = avs or AnswerValidationSystem(profile_engine)
        self._ale = ale or AntiLoopEngine(profile_engine)
        self._hfl = hfl or HumanFeelLayer(profile_engine)

        # Populate ALE personality whitelist from profile
        if profile_engine:
            quirks = profile_engine.get_speech_quirks()
            if quirks:
                self._ale.set_personality_whitelist(quirks)
                _log.debug(f"[RPS] Loaded {len(quirks)} personality quirks into ALE whitelist")

    # ── Public API ────────────────────────────────────────────────────────

    def plan(
        self,
        user_utterance: str,
        system_prompt: str = "",
        emotion_label: str = "neutral",
        recent_replies: list[str] | None = None,
    ) -> ReplyPlan:
        """
        Execute the full 4-stage RPS pipeline.

        Returns a :class:`ReplyPlan` whose ``final_reply`` is ready for TTS.
        """
        t0 = time.monotonic()
        recent_replies = recent_replies or []

        # ── Stage 1: Intent Parse ─────────────────────────────────────────
        intent = self._parse_intent(user_utterance)
        reply_type = self._select_reply_type()
        _log.debug("[RPS] intent=%s sentiment=%s reply_type=%s", intent.primary_intent, intent.sentiment, reply_type)

        # ── Stage 2: Emotional Routing ────────────────────────────────────
        # Blend user-sentiment signal with the incoming emotion_label
        resolved_emotion = self._route_emotion(intent.sentiment, emotion_label)
        _log.debug("[RPS] resolved_emotion=%s", resolved_emotion)

        plan = ReplyPlan(
            user_utterance = user_utterance,
            intent         = intent,
            emotion_label  = resolved_emotion,
        )

        # ── Stages 3 + 4: Generate → AVS + ALE → regen loop ──────────────
        patience      = self._get_regen_patience()
        repair_hint   = ""
        prev_candidate: str | None = None

        for attempt in range(patience + 1):  # 0..patience inclusive
            # Stage 3: Candidate generation
            candidate = self._llm(system_prompt, user_utterance, repair_hint)
            plan.candidates.append(candidate)

            _log.debug(
                "[RPS] attempt=%d candidate_words=%d",
                attempt, len(candidate.split()),
            )

            # ALE check first (catch stuck generation before scoring)
            ale_report = self._ale.check(
                reply          = candidate,
                recent_replies = recent_replies,
                previous_reply = prev_candidate,
            )
            plan.ale_reports.append(ale_report)

            if ale_report.triggered:
                plan.notes.append(
                    f"ALE triggered on attempt {attempt}: mode={ale_report.recovery_mode}"
                )
                if ale_report.recovery_mode == "silence":
                    # Hard stop — return empty reply immediately
                    plan.final_reply      = ""
                    plan.total_elapsed_ms = (time.monotonic() - t0) * 1000
                    plan.notes.append("ALE silence mode → empty reply")
                    return plan
                # Build repair hint for next LLM call
                repair_hint   = ale_report.repair_phrase
                prev_candidate = candidate
                continue   # skip AVS, regen immediately

            # AVS scoring
            avs_result = self._avs.validate(
                reply          = candidate,
                user_utterance = user_utterance,
                emotion_label  = resolved_emotion,
                recent_replies = recent_replies,
                regen_attempt  = attempt,
            )
            plan.avs_results.append(avs_result)

            if avs_result.passed:
                # Winner — apply HFL and wrap up
                plan = self._finalise(plan, candidate, resolved_emotion, t0)
                return plan

            # Not passed — check patience
            if attempt < patience:
                plan.notes.append(
                    f"AVS fail attempt {attempt}: composite={avs_result.scores.composite:.3f} "
                    f"< threshold={avs_result.threshold:.3f}"
                )
                repair_hint    = (
                    f"[Your previous reply scored {avs_result.scores.composite:.2f} "
                    f"(need ≥ {avs_result.threshold:.2f}).  Please improve it by: "
                    f"covering the user's question more directly, checking for "
                    f"contradictions, and aiming for ~{int(50 + self._get_verbosity() * 150)} words.]"
                )
                prev_candidate = candidate

        # All attempts exhausted — fallback accept via AVS.select_best()
        if plan.avs_results:
            best = self._avs.select_best(plan.avs_results)
            plan.fallback_accept = True
            plan.notes.append("Fallback accept after all regen attempts exhausted")
            plan = self._finalise(plan, best.reply, resolved_emotion, t0)
        elif plan.candidates:
            # ALE kept triggering; use FIRST candidate (most diverse from prior outputs)
            plan = self._finalise(plan, plan.candidates[0], resolved_emotion, t0)
            plan.fallback_accept = True
        else:
            plan.final_reply      = ""
            plan.total_elapsed_ms = (time.monotonic() - t0) * 1000

        return plan

    # ── Internal helpers ──────────────────────────────────────────────────

    def _finalise(
        self,
        plan: ReplyPlan,
        raw_reply: str,
        emotion_label: str,
        t0: float,
    ) -> ReplyPlan:
        """Apply HFL and set final_reply + elapsed time."""
        hfl_result        = self._hfl.process(raw_reply, emotion_label)
        plan.hfl_result   = hfl_result
        plan.final_reply  = hfl_result.processed
        plan.total_elapsed_ms = (time.monotonic() - t0) * 1000
        return plan

    # ── Stage 1: Intent parsing ───────────────────────────────────────────

    def _select_reply_type(self) -> str:
        """Select a reply type strategy based on profile weights."""
        if self._pe:
            weights = self._pe.get_reply_type_weights()
        else:
            weights = {"explain": 0.5, "react": 0.5}

        import random
        reply_types = list(weights.keys())
        reply_weights = [weights.get(rt, 0.0) for rt in reply_types]

        # Normalize weights to sum to 1.0
        total = sum(reply_weights)
        if total > 0:
            reply_weights = [w / total for w in reply_weights]
        else:
            reply_weights = [1.0 / len(reply_types)] * len(reply_types)

        selected = random.choices(reply_types, weights=reply_weights, k=1)[0]
        _log.debug(f"[RPS] Selected reply_type={selected} from weights={weights}")
        return selected

    @staticmethod
    def _parse_intent(utterance: str) -> IntentFrame:
        """
        Lightweight rule-based intent extraction.

        In production, replace with a proper NLU / embedding classifier.
        """
        import re

        lower = utterance.lower().strip()
        frame = IntentFrame()

        # Question words
        q_words = [w for w in ["what", "how", "why", "when", "where", "who", "which",
                                "can you", "could you", "would you", "should"] if w in lower]
        frame.question_words = q_words

        # Sentiment
        positive_re = re.compile(r"\b(great|love|thanks|happy|excited|please|awesome)\b")
        negative_re = re.compile(r"\b(hate|angry|frustrated|terrible|bad|wrong|upset)\b")
        if positive_re.search(lower):
            frame.sentiment = "positive"
        elif negative_re.search(lower):
            frame.sentiment = "negative"
        else:
            frame.sentiment = "neutral"

        # Primary intent
        if re.search(r"\b(remind|set|schedule|alarm|timer)\b", lower):
            frame.primary_intent = "set_reminder"
        elif re.search(r"\b(play|music|song|queue|next)\b", lower):
            frame.primary_intent = "media_control"
        elif re.search(r"\b(what('?s| is)|tell me|explain|define|describe)\b", lower):
            frame.primary_intent = "information_request"
        elif re.search(r"\b(how (do|can|should|would)|steps|instructions)\b", lower):
            frame.primary_intent = "procedural_request"
        elif re.search(r"\b(hello|hi|hey|how are you|what's up)\b", lower):
            frame.primary_intent = "greeting"
        elif re.search(r"\b(thank(s| you)|appreciate)\b", lower):
            frame.primary_intent = "gratitude"
        elif re.search(r"\b(joke|funny|laugh|humor|entertain)\b", lower):
            frame.primary_intent = "entertainment"
        elif q_words:
            frame.primary_intent = "general_question"
        else:
            frame.primary_intent = "general_query"

        return frame

    # ── Stage 2: Emotional routing ────────────────────────────────────────

    @staticmethod
    def _route_emotion(user_sentiment: str, incoming_label: str) -> str:
        """
        Blend user sentiment with the EmotionNet-detected emotion label to
        produce a final tone for HFL prosody computation.
        """
        # User sentiment overrides only for strong signals
        if user_sentiment == "positive" and incoming_label in ("neutral", ""):
            return "happy"
        if user_sentiment == "negative" and incoming_label in ("neutral", ""):
            return "concerned"
        # Otherwise trust the EmotionNet label
        return incoming_label or "neutral"

    # ── Fallback stub LLM ─────────────────────────────────────────────────

    @staticmethod
    def _stub_llm(system_prompt: str, user_text: str, repair_hint: str) -> str:
        """
        Placeholder used when no real LLM function is injected.
        Returns a minimal response so the pipeline doesn't crash.
        """
        _log.warning("[RPS] Using stub LLM — no real LLM backend injected")
        return f"[Stub reply for: {user_text[:40]}]"

    # ── Profile accessors ─────────────────────────────────────────────────

    def _get_regen_patience(self) -> int:
        if self._pe:
            return int(self._pe.regen_patience)
        return 3

    def _get_verbosity(self) -> float:
        if self._pe:
            return float(self._pe.verbosity)
        return 0.50
