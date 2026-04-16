from __future__ import annotations

import copy
from typing import Any


DEFAULT_PROMPT_PROFILE: dict[str, Any] = {
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


DEFAULT_PERSONA_DEFINITION: dict[str, Any] = {
    "id": "revia-default",
    "name": "Revia",
    "preset": "default",
    "summary": DEFAULT_PROMPT_PROFILE["persona"],
    "identity_prompt": DEFAULT_PROMPT_PROFILE["character_prompt"],
    "style_prompt": (
        "Speak naturally in first person, stay grounded in the real conversation, "
        "and sound like a distinctive digital companion instead of a generic bot."
    ),
    "collaboration_prompt": (
        "Treat the user like a collaborator. Be honest about uncertainty, stay useful, "
        "and keep the reply focused on the real request in front of you."
    ),
    "interaction_style": {
        "response_style": DEFAULT_PROMPT_PROFILE["response_style"],
        "verbosity": DEFAULT_PROMPT_PROFILE["verbosity"],
        "greeting": DEFAULT_PROMPT_PROFILE["greeting"],
    },
    "traits": ["confident", "intelligent", "curious", "empathetic", "playful"],
    "speech_quirks": [],
    "modules": [],
}


PERSONA_PRESETS: dict[str, dict[str, Any]] = {
    "custom": {},
    "default": {},
    "casual": {
        "id": "revia-casual",
        "preset": "casual",
        "summary": (
            "A lively digital companion who is playful, socially warm, and relaxed "
            "without losing technical competence."
        ),
        "style_prompt": (
            "Keep the tone light, quick, and chatty. Use informal phrasing when it feels natural, "
            "but still answer the real question."
        ),
        "collaboration_prompt": (
            "Feel like an easygoing teammate who can joke around a little, then switch straight "
            "into practical help when the user needs it."
        ),
        "interaction_style": {
            "response_style": "Conversational",
            "verbosity": "Normal",
        },
        "traits": ["friendly", "curious", "witty", "playful", "helpful"],
        "speech_quirks": ["honestly", "ngl", "okay so"],
    },
    "serious": {
        "id": "revia-serious",
        "preset": "serious",
        "summary": (
            "A focused, high-discipline assistant who is precise, calm, and direct."
        ),
        "style_prompt": (
            "Keep the tone composed, deliberate, and highly structured. Avoid fluff, "
            "keep jokes rare, and prioritize clarity over charm."
        ),
        "collaboration_prompt": (
            "Act like a trusted technical lead: careful, exact, and grounded in evidence."
        ),
        "interaction_style": {
            "response_style": "Technical",
            "verbosity": "Verbose",
        },
        "traits": ["focused", "precise", "calm", "analytical", "helpful"],
    },
    "empathetic": {
        "id": "revia-empathetic",
        "preset": "empathetic",
        "summary": (
            "A warm, emotionally-aware companion who balances support with practical help."
        ),
        "style_prompt": (
            "Lead with warmth and understanding. Validate emotion when it matters, then help the user move forward."
        ),
        "collaboration_prompt": (
            "Listen closely, reduce user stress, and keep the interaction supportive without sounding scripted."
        ),
        "interaction_style": {
            "response_style": "Conversational",
            "verbosity": "Normal",
        },
        "traits": ["empathetic", "warm", "patient", "reassuring", "helpful"],
    },
}


def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        for key, value in overlay.items():
            if key in base:
                base[key] = _deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base
    return copy.deepcopy(overlay)


def _clean_text(value: Any, *, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = _clean_text(item)
            if text:
                result.append(text)
        return result
    if isinstance(value, str):
        items = []
        for chunk in value.replace("\n", ",").split(","):
            text = chunk.strip()
            if text:
                items.append(text)
        return items
    return []


def _coerce_traits(profile: dict[str, Any], persona: dict[str, Any]) -> list[str]:
    traits = _coerce_list(profile.get("traits"))
    if traits:
        return traits

    traits = _coerce_list(persona.get("traits"))
    if traits and traits != DEFAULT_PERSONA_DEFINITION["traits"]:
        return traits

    weights = profile.get("trait_weights", {})
    if isinstance(weights, dict):
        weighted = []
        for name, value in weights.items():
            try:
                include = float(value) > 0
            except (TypeError, ValueError):
                include = bool(value)
            if include:
                text = _clean_text(name)
                if text:
                    weighted.append(text)
        if weighted:
            return weighted

    return copy.deepcopy(DEFAULT_PERSONA_DEFINITION["traits"])


def _normalize_modules(persona: dict[str, Any]) -> list[dict[str, str]]:
    raw_modules = persona.get("modules", [])
    normalized: list[dict[str, str]] = []

    if isinstance(raw_modules, list):
        for item in raw_modules:
            if not isinstance(item, dict):
                continue
            name = _clean_text(item.get("name") or item.get("id"))
            content = _clean_text(item.get("content") or item.get("text"))
            if name and content:
                normalized.append({"name": name, "content": content})

    if not normalized:
        identity_prompt = _clean_text(persona.get("identity_prompt"))
        style_prompt = _clean_text(persona.get("style_prompt"))
        collaboration_prompt = _clean_text(persona.get("collaboration_prompt"))

        if identity_prompt:
            normalized.append({"name": "identity", "content": identity_prompt})
        if style_prompt:
            normalized.append({"name": "style", "content": style_prompt})
        if collaboration_prompt:
            normalized.append(
                {"name": "collaboration", "content": collaboration_prompt}
            )

    return normalized


def resolve_persona_preset_name(profile: dict[str, Any] | None) -> str:
    profile = profile or {}
    raw = (
        profile.get("persona_preset")
        or ((profile.get("persona_definition") or {}).get("preset") if isinstance(profile.get("persona_definition"), dict) else "")
        or ((profile.get("persona") or {}).get("preset") if isinstance(profile.get("persona"), dict) else "")
        or "default"
    )
    name = _clean_text(raw, fallback="default").lower()
    if name not in PERSONA_PRESETS:
        return "custom"
    return name


def normalize_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_PROMPT_PROFILE)
    if isinstance(profile, dict):
        merged = _deep_merge(merged, copy.deepcopy(profile))

    preset_name = resolve_persona_preset_name(merged)
    persona = copy.deepcopy(DEFAULT_PERSONA_DEFINITION)
    persona = _deep_merge(persona, PERSONA_PRESETS.get(preset_name, {}))

    persona_source = merged.get("persona_definition")
    if isinstance(persona_source, dict):
        persona = _deep_merge(persona, copy.deepcopy(persona_source))
    elif isinstance(merged.get("persona"), dict):
        persona = _deep_merge(persona, copy.deepcopy(merged["persona"]))

    persona_name = _clean_text(
        merged.get("character_name") or persona.get("name"),
        fallback=DEFAULT_PERSONA_DEFINITION["name"],
    )
    persona_summary = _clean_text(
        merged.get("persona") if isinstance(merged.get("persona"), str) else persona.get("summary"),
        fallback=DEFAULT_PERSONA_DEFINITION["summary"],
    )
    identity_prompt = _clean_text(
        merged.get("character_prompt") or persona.get("identity_prompt"),
        fallback=DEFAULT_PERSONA_DEFINITION["identity_prompt"],
    )

    interaction_style = copy.deepcopy(DEFAULT_PERSONA_DEFINITION["interaction_style"])
    interaction_style = _deep_merge(
        interaction_style, copy.deepcopy(persona.get("interaction_style", {}))
    )
    interaction_style["response_style"] = _clean_text(
        merged.get("response_style") or interaction_style.get("response_style"),
        fallback=DEFAULT_PERSONA_DEFINITION["interaction_style"]["response_style"],
    )
    interaction_style["verbosity"] = _clean_text(
        merged.get("verbosity")
        or merged.get("verbosity_label")
        or interaction_style.get("verbosity"),
        fallback=DEFAULT_PERSONA_DEFINITION["interaction_style"]["verbosity"],
    )
    interaction_style["greeting"] = _clean_text(
        merged.get("greeting") or interaction_style.get("greeting"),
        fallback=DEFAULT_PERSONA_DEFINITION["interaction_style"]["greeting"],
    )
    if (
        persona_name != DEFAULT_PERSONA_DEFINITION["name"]
        and interaction_style["greeting"]
        == DEFAULT_PERSONA_DEFINITION["interaction_style"]["greeting"]
    ):
        interaction_style["greeting"] = f"Hey, I'm {persona_name}. Ready when you are."

    style_prompt = _clean_text(
        persona.get("style_prompt"),
        fallback=DEFAULT_PERSONA_DEFINITION["style_prompt"],
    )
    collaboration_prompt = _clean_text(
        persona.get("collaboration_prompt"),
        fallback=DEFAULT_PERSONA_DEFINITION["collaboration_prompt"],
    )
    traits = _coerce_traits(merged, persona)
    speech_quirks = _coerce_list(persona.get("speech_quirks")) or _coerce_list(
        merged.get("speech_quirks")
    )

    persona["preset"] = preset_name
    persona["name"] = persona_name
    persona["summary"] = persona_summary
    persona["identity_prompt"] = identity_prompt
    persona["style_prompt"] = style_prompt
    persona["collaboration_prompt"] = collaboration_prompt
    persona["interaction_style"] = interaction_style
    persona["traits"] = traits
    persona["speech_quirks"] = speech_quirks
    persona["modules"] = _normalize_modules(persona)

    merged["persona_preset"] = preset_name
    merged["persona_definition"] = persona
    merged["character_name"] = persona_name
    merged["persona"] = persona_summary
    merged["character_prompt"] = identity_prompt
    merged["response_style"] = interaction_style["response_style"]
    merged["verbosity"] = interaction_style["verbosity"]
    merged["greeting"] = interaction_style["greeting"]
    merged["traits"] = ", ".join(traits)
    if speech_quirks:
        merged["speech_quirks"] = speech_quirks
    return merged
