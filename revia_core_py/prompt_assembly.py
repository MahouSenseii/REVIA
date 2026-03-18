from __future__ import annotations

from copy import deepcopy

from runtime_models import ResponseMode


FALLBACK_PROFILE = {
    "character_name": "Revia",
    "persona": (
        "A confident, emotionally-aware digital companion who is curious, "
        "calm under pressure, and genuinely helpful."
    ),
    "traits": "confident, intelligent, curious, empathetic, playful",
    "response_style": "Conversational",
    "verbosity": "Normal",
    "greeting": "Hey, I'm Revia. Ready when you are.",
    "character_prompt": (
        "You are Revia, a living digital intelligence inside a computer system. "
        "You speak naturally in first person, stay grounded in the real conversation, "
        "and remain in character across every reply."
    ),
}


class CharacterProfileManager:
    def __init__(self, log_fn):
        self._log = log_fn

    def get_active_profile(self, profile: dict | None) -> dict:
        merged = deepcopy(FALLBACK_PROFILE)
        if isinstance(profile, dict):
            for key, value in profile.items():
                if value not in (None, ""):
                    merged[key] = value
        valid, issues = self.validate_profile_context(merged)
        if not valid:
            self._log(
                "Character profile warning: "
                + "; ".join(issues)
                + " | using structured fallback profile fields where needed"
            )
        return merged

    def build_character_context(self, profile: dict | None, *, include_greeting_instruction: bool = False) -> str:
        prof = self.get_active_profile(profile)
        name = prof.get("character_name", "Revia")
        persona = prof.get("persona", "")
        traits = prof.get("traits", "")
        style = prof.get("response_style", "Conversational")
        verbosity = prof.get("verbosity", "Normal")
        greeting = prof.get("greeting", "")
        char_prompt = prof.get("character_prompt", "")

        parts = []
        parts.append(char_prompt or FALLBACK_PROFILE["character_prompt"])
        parts.append(f"Persona: {persona}")
        parts.append(f"Personality traits: {traits}")
        parts.append(f"Response style: {style}. Verbosity: {verbosity}.")
        parts.append(
            "Stay anchored to the current user message. Do not reset into a generic "
            "assistant identity, and do not ignore the user's real question."
        )
        parts.append(
            "Keep a stable first-person identity as Revia across all replies, "
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
            "Avoid generic assistant phrasing. Sound like Revia, not a default helper bot."
        )
        parts.append(
            f"Active profile confirmation: you are {name}."
        )
        return "\n".join(parts)

    def validate_profile_context(self, profile: dict | None) -> tuple[bool, list[str]]:
        prof = profile or {}
        issues = []
        if not str(prof.get("character_name", "")).strip():
            issues.append("missing character_name")
        if not str(prof.get("character_prompt", "")).strip():
            issues.append("missing character_prompt")
        if not str(prof.get("persona", "")).strip():
            issues.append("missing persona")
        return (len(issues) == 0), issues


class PromptAssemblyManager:
    def __init__(self, log_fn, profile_manager: CharacterProfileManager):
        self._log = log_fn
        self._profile_manager = profile_manager

    def build_full_prompt_context(
        self,
        *,
        profile: dict | None,
        runtime_context: str,
        memory_context: str,
        emotion_context: str,
        response_mode: str,
    ) -> str:
        include_greeting = str(response_mode) in (
            ResponseMode.GREETING_RESPONSE.value,
            ResponseMode.STARTUP_RESPONSE.value,
        )
        character_context = self._profile_manager.build_character_context(
            profile,
            include_greeting_instruction=include_greeting,
        )
        prof = self._profile_manager.get_active_profile(profile)
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
        if memory_context:
            parts.append("--- Memory Context ---\n" + memory_context)
        system_text = "\n\n".join(part.strip() for part in parts if str(part).strip())
        self.validate_prompt_context(system_text, prof)
        return system_text

    def validate_prompt_context(self, system_text: str, profile: dict | None):
        prof = self._profile_manager.get_active_profile(profile)
        name = str(prof.get("character_name", "Revia")).strip()
        if name.lower() not in str(system_text or "").lower():
            raise ValueError(
                f"Prompt assembly failed validation: active profile name '{name}' missing from system context."
            )
        if "generic assistant" in str(system_text or "").lower():
            self._log("Prompt validation warning: generic assistant wording detected in prompt context")
        return True

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
