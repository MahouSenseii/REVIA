from __future__ import annotations


def build_self_initiation_prompt(candidate, state, memories, scored_candidate) -> str:
    memory_hint = ""
    if getattr(memories, "summary", ""):
        memory_hint = f"\nAutonomy memory context: {memories.summary}"
    if getattr(memories, "avoid", None):
        memory_hint += "\nAvoid bringing up: " + " | ".join(memories.avoid[-2:])

    return (
        "You are deciding to speak without a direct user message.\n"
        "Only speak if the line is genuinely worth saying. Silence was already scored, "
        "so keep this short and intentional.\n"
        f"Candidate type: {candidate.type}\n"
        f"Topic: {candidate.topic or 'none'}\n"
        f"Reason: {candidate.reason}\n"
        f"Candidate instruction: {candidate.text}\n"
        f"User mood: {state.user_mood}\n"
        f"Autonomy score: {scored_candidate.final_score:.2f}\n"
        f"{memory_hint}\n"
        "Write exactly what Revia should say, in her natural voice. "
        "One or two sentences. Do not mention scoring, candidates, timers, or autonomy internals "
        "unless the topic itself is REVIA's architecture."
    )
