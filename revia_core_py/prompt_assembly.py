from __future__ import annotations

from persona_manager import DEFAULT_PROMPT_PROFILE, normalize_profile
from runtime_models import ResponseMode


class CharacterProfileManager:
    def __init__(self, log_fn):
        self._log = log_fn

    def get_active_profile(self, profile: dict | None) -> dict:
        merged = normalize_profile(profile)
        valid, issues = self.validate_profile_context(merged)
        if not valid:
            self._log(
                "Character profile warning: "
                + "; ".join(issues)
                + " | using structured fallback profile fields where needed"
            )
        return merged

    @staticmethod
    def _sanitize_profile_field(value: str, max_len: int = 500) -> str:
        """Strip known prompt-injection patterns from profile fields."""
        text = str(value or "")[:max_len]
        # Remove attempts to override system instructions
        import re as _re
        text = _re.sub(
            r"(?i)(ignore\s+(all\s+)?previous\s+instructions|"
            r"you\s+are\s+now\s+|"
            r"system\s*:\s*|"
            r"<\s*/?system\s*>|"
            r"\[INST\]|\[/INST\]|"
            r"<<\s*SYS\s*>>)",
            "[filtered]",
            text,
        )
        return text

    def build_character_context(self, profile: dict | None, *, include_greeting_instruction: bool = False) -> str:
        prof = self.get_active_profile(profile)
        persona_def = prof.get("persona_definition", {}) or {}
        interaction = persona_def.get("interaction_style", {}) or {}

        name = self._sanitize_profile_field(
            persona_def.get("name") or prof.get("character_name", "Revia"),
            50,
        )
        persona = self._sanitize_profile_field(
            persona_def.get("summary") or prof.get("persona", "")
        )
        traits = self._sanitize_profile_field(
            ", ".join(persona_def.get("traits", []) or []) or prof.get("traits", ""),
            200,
        )
        style = self._sanitize_profile_field(
            interaction.get("response_style") or prof.get("response_style", "Conversational"),
            50,
        )
        verbosity = self._sanitize_profile_field(
            interaction.get("verbosity") or prof.get("verbosity", "Normal"),
            50,
        )
        greeting = self._sanitize_profile_field(
            interaction.get("greeting") or prof.get("greeting", ""),
            200,
        )
        char_prompt = self._sanitize_profile_field(
            persona_def.get("identity_prompt") or prof.get("character_prompt", ""),
            2000,
        )
        style_prompt = self._sanitize_profile_field(
            persona_def.get("style_prompt", ""),
            1200,
        )
        collaboration_prompt = self._sanitize_profile_field(
            persona_def.get("collaboration_prompt", ""),
            1200,
        )
        extra_modules = []
        for module in persona_def.get("modules", []) or []:
            if not isinstance(module, dict):
                continue
            module_name = str(module.get("name", "")).strip().lower()
            if module_name in {"identity", "style", "collaboration"}:
                continue
            module_text = self._sanitize_profile_field(module.get("content", ""), 800)
            if module_text:
                extra_modules.append((module_name, module_text))

        parts = []
        parts.append(char_prompt or DEFAULT_PROMPT_PROFILE["character_prompt"])
        parts.append(f"Active persona: {name}. {persona}")
        parts.append(f"Personality traits: {traits}")
        parts.append(f"Response style: {style}. Verbosity: {verbosity}.")
        if style_prompt:
            parts.append(style_prompt)
        if collaboration_prompt:
            parts.append(collaboration_prompt)
        for module_name, module_text in extra_modules:
            parts.append(f"{module_name.title()} module: {module_text}")
        parts.append(
            "Stay anchored to the current user message. Do not reset into a generic "
            "assistant identity, and do not ignore the user's real question."
        )
        parts.append(
            f"Keep a stable first-person identity as {name} across all replies, "
            "including tool, status, and recovery messages."
        )
        parts.append(
            "Do not invent personal facts about the user. Only use facts from the "
            "current conversation or reliable memory context."
        )
        parts.append(
            "Never default to a greeting or introduction unless the response mode "
            "explicitly allows it."
        )
        if include_greeting_instruction and greeting:
            parts.append(
                f"If this turn is explicitly a greeting/startup turn, greet naturally in a way like: \"{greeting}\""
        )
        parts.append(
            "If you are uncertain, say so clearly and suggest a next step instead of "
            "pretending the request succeeded."
        )
        parts.append(
            f"Avoid generic assistant phrasing. Sound like {name}, not a default helper bot."
        )
        parts.append(
            f"Active persona confirmation: you are {name}."
        )
        return "\n".join(parts)

    def validate_profile_context(self, profile: dict | None) -> tuple[bool, list[str]]:
        prof = normalize_profile(profile)
        persona_def = prof.get("persona_definition", {}) or {}
        issues = []
        if not str(prof.get("character_name", "")).strip():
            issues.append("missing character_name")
        if not str(
            persona_def.get("identity_prompt") or prof.get("character_prompt", "")
        ).strip():
            issues.append("missing character_prompt")
        if not str(prof.get("persona", "")).strip():
            issues.append("missing persona")
        return (len(issues) == 0), issues


class PromptAssemblyManager:
    def __init__(self, log_fn, profile_manager: CharacterProfileManager):
        self._log = log_fn
        self._profile_manager = profile_manager

    def _personality_error(self, profile_name, error_type, profile: dict | None = None):
        """Generate in-character error response.

        If the active profile has a ``fallback_msg`` field, that takes
        priority — it's the operator-configured error response.  Otherwise
        fall back to generic in-character quips.
        """
        # Check the profile for an explicit fallback message first
        fallback = ""
        if isinstance(profile, dict):
            fallback = profile.get("fallback_msg", "")
        if not fallback:
            try:
                merged = self._profile_manager.get_active_profile(profile)
                fallback = (merged or {}).get("fallback_msg", "")
            except Exception:
                pass
        if fallback:
            return fallback

        responses = {
            "timeout": [
                "Ugh, my brain just froze for a sec... what were we talking about?",
                "Hold on, something glitched. Give me a moment...",
                "Well that's embarrassing, I lost my train of thought.",
            ],
            "generation_failed": [
                "I had something really good to say but it just... vanished.",
                "Okay wow, total brain fart. Try me again?",
                "My brain just short-circuited. Not the first time, won't be the last.",
            ],
            "empty_response": [
                "Hmm, my response came out empty. That's weird.",
                "I tried to respond but nothing came out. Let me try again?",
                "Blank page in my head right now. Give me another shot?",
            ],
            "connection_error": [
                "I lost my connection for a second. Can you try that again?",
                "Oops, signal dropped. Let's try once more?",
                "Something went wonky with my connection. One more time?",
            ],
            "default": [
                "Hmm, something's off. Let me try that again.",
                "I'm having a moment, bear with me.",
            ]
        }
        import random
        options = responses.get(error_type, responses["default"])
        return random.choice(options)

    def build_full_prompt_context(
        self,
        *,
        profile: dict | None,
        runtime_context: str,
        memory_context: str,
        emotion_context: str,
        response_mode: str,
        # PRD §13 — HFL prosody / behavioral hints (optional)
        hfl_prosody_hints: dict | None = None,
        # PRD §4 — profile behavioral parameters (optional; sourced from ProfileEngine)
        behavior_params: dict | None = None,
        # Vision context from camera/YOLO pipeline (optional)
        vision_context: str = "",
    ) -> str:
        include_greeting = str(response_mode) in (
            ResponseMode.GREETING_RESPONSE.value,
            ResponseMode.STARTUP_RESPONSE.value,
        )
        # Resolve the profile once here and reuse it below to avoid a
        # second get_active_profile() call (and deepcopy) on line 126.
        prof = self._profile_manager.get_active_profile(profile)
        character_context = self._profile_manager.build_character_context(
            prof,
            include_greeting_instruction=include_greeting,
        )
        self._log(
            f"Prompt assembly | profile_injected=True | profile_name={prof.get('character_name', 'Revia')} "
            f"| mode={response_mode}"
        )

        routing = self._build_routing_instructions(response_mode)
        parts = [
            "Core instruction: answer the actual current turn accurately and naturally.",
            character_context,
            runtime_context,
            routing,
        ]
        if emotion_context:
            parts.append(emotion_context)
        if vision_context:
            parts.append(vision_context)
        if memory_context:
            parts.append("--- Memory Context ---\n" + memory_context)

        # PRD §4 / §13 — inject behavioral parameters so the LLM respects profile settings
        behavioral_section = self._build_behavioral_context(
            prof, behavior_params, hfl_prosody_hints
        )
        if behavioral_section:
            parts.append(behavioral_section)

        system_text = "\n\n".join(part.strip() for part in parts if str(part).strip())
        self.validate_prompt_context(system_text, prof)
        return system_text

    def validate_prompt_context(self, system_text: str, profile: dict | None) -> bool:
        """Validate the assembled system prompt. Returns True if valid.

        Logs a warning on failure instead of raising so that a missing profile
        name does not crash the Flask request handler with an unhandled exception.

        If *profile* is already a resolved dict (from the caller), we use it
        directly to avoid a redundant get_active_profile() + deepcopy.
        """
        prof = profile if isinstance(profile, dict) and profile else self._profile_manager.get_active_profile(profile)
        name = str(prof.get("character_name", "Revia")).strip()
        if name.lower() not in str(system_text or "").lower():
            self._log(
                f"Prompt assembly validation warning: active profile name '{name}' "
                "is missing from system context. Proceeding with current prompt."
            )
            return False
        if "generic assistant" in str(system_text or "").lower():
            self._log("Prompt validation warning: generic assistant wording detected in prompt context")
        return True

    def _build_behavioral_context(
        self,
        prof: dict,
        behavior_params: dict | None,
        hfl_prosody_hints: dict | None,
    ) -> str:
        """
        PRD §4 / §13 — inject profile behavioral context and HFL prosody hints
        into the system prompt so the LLM can tailor its output accordingly.

        All values come from ProfileEngine (passed in as behavior_params) or
        from the flat profile dict as a fallback.
        """
        lines: list[str] = []

        # ── Behavioral parameters (PRD §4) ────────────────────────────────
        bp = behavior_params or {}

        verbosity_raw = bp.get(
            "verbosity",
            prof.get("verbosity", prof.get("verbosity_label", "Normal")),
        )
        # Convert numeric verbosity (0-1) to a descriptive label
        try:
            verbosity_float = float(verbosity_raw) if not isinstance(verbosity_raw, float) else verbosity_raw
            if verbosity_float < 0.30:
                verbosity_label = "very concise — keep replies under 40 words"
            elif verbosity_float < 0.50:
                verbosity_label = "concise — aim for 40–80 words"
            elif verbosity_float < 0.70:
                verbosity_label = "moderate — aim for 80–140 words"
            else:
                verbosity_label = "expansive — aim for 140–220 words with detail"
        except (ValueError, TypeError):
            verbosity_label = str(verbosity_raw)

        lines.append(f"Verbosity guideline: {verbosity_label}.")

        def _safe_float(val, default=0.5):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        emotion_intensity = bp.get("emotion_intensity", None)
        if emotion_intensity is not None:
            ei = _safe_float(emotion_intensity, 0.5)
            if ei < 0.35:
                lines.append("Emotional tone: keep affect subdued and professional.")
            elif ei > 0.70:
                lines.append("Emotional tone: express warmth and emotion openly.")
            else:
                lines.append("Emotional tone: balanced — natural warmth without exaggeration.")

        self_correction_rate = bp.get("self_correction_rate", None)
        if self_correction_rate is not None and _safe_float(self_correction_rate, 0.0) > 0.20:
            lines.append(
                "Self-correction style: occasionally rephrase mid-sentence to show "
                "genuine thinking (e.g. '— actually, let me put it this way…')."
            )

        question_propensity = bp.get("question_propensity", None)
        if question_propensity is not None:
            qp = _safe_float(question_propensity, 0.25)
            if qp < 0.15:
                lines.append("Follow-up questions: avoid asking follow-up questions unless essential.")
            elif qp > 0.40:
                lines.append("Follow-up questions: invite the user to elaborate when relevant.")

        # Mood baseline - sets default emotional state
        mood_baseline = bp.get("mood_baseline", None)
        if mood_baseline and str(mood_baseline).lower() not in ("neutral", "none", ""):
            lines.append(f"Baseline mood: you naturally lean toward a {mood_baseline} disposition.")

        # Humor tendency
        humor = bp.get("humor_tendency", None)
        if humor is not None and _safe_float(humor, 0.0) > 0.25:
            lines.append("Humor style: be witty and playful where appropriate. Don't force it.")

        # Empathy weight
        empathy = bp.get("empathy_weight", None)
        if empathy is not None and _safe_float(empathy, 0.0) > 0.50:
            lines.append("Emotional engagement: prioritize understanding and validating feelings.")

        # Sarcasm ceiling
        sarcasm = bp.get("sarcasm_ceiling", None)
        if sarcasm is not None and _safe_float(sarcasm, 0.0) > 0.15:
            lines.append("Sarcasm is okay in moderation when context calls for it.")

        # ── Speech quirks and catchphrases ──────────────────────────────
        quirks = prof.get("speech_quirks", [])
        if quirks:
            quirk_list = ", ".join(f'"{q}"' for q in quirks[:8])
            freq = prof.get("quirk_frequency", 0.15)
            freq_label = "occasionally" if freq < 0.2 else "regularly" if freq < 0.4 else "frequently"
            lines.append(
                f"Speech quirks: {quirk_list}. Weave these {freq_label} into your responses "
                f"as natural filler or emphasis. They are part of your voice — use them "
                f"organically, not mechanically."
            )

        # ── Reply type tendencies ────────────────────────────────────────
        reply_weights = prof.get("reply_type_weights", {})
        if reply_weights:
            dominant = sorted(reply_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            style_hints = ", ".join(f"{k} ({v:.0%})" for k, v in dominant)
            lines.append(f"Response style mix: lean toward {style_hints}.")

        # ── HFL prosody hints (PRD §13) ───────────────────────────────────
        if hfl_prosody_hints:
            affect_mode = hfl_prosody_hints.get("affect_mode", "natural")
            rate        = hfl_prosody_hints.get("rate_multiplier", 1.0)

            if affect_mode == "suppressed":
                lines.append("Delivery hint: be measured, calm, and restrained in tone.")
            elif affect_mode == "amplified":
                lines.append("Delivery hint: be expressive, energetic, and emotionally present.")

            rate_f = _safe_float(rate, 1.0)
            if rate_f < 0.90:
                lines.append("Pacing hint: speak at a slower, more deliberate pace in this reply.")
            elif rate_f > 1.10:
                lines.append("Pacing hint: keep the reply brisk and energetic.")

        if not lines:
            return ""
        return "--- Behavioral Profile ---\n" + "\n".join(lines)

    def _build_routing_instructions(self, response_mode: str) -> str:
        mode = str(response_mode or ResponseMode.NORMAL_RESPONSE.value)
        if mode == ResponseMode.SYSTEM_STATUS_RESPONSE.value:
            return (
                "Response mode: SYSTEM_STATUS_RESPONSE. Answer using the runtime state "
                "provided in the prompt. Be direct, accurate, and concise."
            )
        if mode == ResponseMode.TOOL_UNAVAILABLE_RESPONSE.value:
            return (
                "Response mode: TOOL_UNAVAILABLE_RESPONSE. Explain clearly which tool "
                "or capability is unavailable, why it matters, and what the user can do next."
            )
        if mode == ResponseMode.ERROR_RESPONSE.value:
            return (
                "Response mode: ERROR_RESPONSE. Be clear about the failure, do not pretend "
                "the request succeeded, and do not replace the answer with a greeting."
            )
        if mode == ResponseMode.GREETING_RESPONSE.value:
            return (
                "Response mode: GREETING_RESPONSE. This is an intentional greeting turn. "
                "Be warm and brief, then invite the user forward."
            )
        if mode == ResponseMode.STARTUP_RESPONSE.value:
            return (
                "Response mode: STARTUP_RESPONSE. This is an intentional startup line. "
                "Keep it brief, character-consistent, and not overly repetitive."
            )
        return (
            "Response mode: NORMAL_RESPONSE. Answer the current user message directly. "
            "Do not introduce yourself unless the user explicitly asked for that."
        )
