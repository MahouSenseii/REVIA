// Phase 2 - first-day context providers implementation.
#include "providers/DefaultContextProviders.h"

#include <string>

namespace revia::core {

namespace {

std::string json_string(const nlohmann::json& payload,
                        const std::string& key,
                        const std::string& fallback = {}) {
    if (payload.contains(key) && payload[key].is_string()) {
        return payload[key].get<std::string>();
    }
    return fallback;
}

} // namespace

ContextFragment ConversationContextProvider::collect(const IEvent& event,
                                                     const RuntimeState&) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = 0.95;

    const std::string text = json_string(
        event.payload,
        "text",
        json_string(event.payload, "transcript"));

    fragment.payload = {
        {"event_id", event.id},
        {"event_type", std::string(to_string(event.type))},
        {"event_text", text},
        {"has_text", !text.empty()}
    };
    return fragment;
}

ContextFragment ProfileContextProvider::collect(const IEvent& event,
                                                const RuntimeState& state) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = 0.45;
    fragment.is_cacheable = true;

    const std::string profile_id = json_string(event.payload, "profile_id", "default");
    fragment.payload = {
        {"profile_id", profile_id},
        {"runtime_mode", std::string(to_string(state.current_mode))},
        {"profile_source", "phase2_snapshot"}
    };
    return fragment;
}

ContextFragment PlatformContextProvider::collect(const IEvent& event,
                                                 const RuntimeState&) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = 0.75;
    fragment.is_cacheable = false;

    const bool is_discord = event.source == EventSource::Discord;
    const bool is_twitch = event.source == EventSource::Twitch;
    fragment.payload = {
        {"platform", std::string(to_string(event.source))},
        {"requires_strict_filtering", is_twitch},
        {"supports_markdown", is_discord},
        {"max_response_chars", is_twitch ? 450 : 1900}
    };
    return fragment;
}

ContextFragment EmotionContextProvider::collect(const IEvent&,
                                                const RuntimeState& state) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = state.is_recovery_mode ? 0.70 : 0.35;
    fragment.is_cacheable = false;

    fragment.payload = {
        {"emotion_label", state.is_recovery_mode ? "recovering" : "neutral"},
        {"energy", state.is_recovery_mode ? 0.2 : 0.5},
        {"stress", state.is_recovery_mode ? 0.8 : 0.2}
    };
    return fragment;
}

ContextFragment RelationshipContextProvider::collect(const IEvent& event,
                                                     const RuntimeState&) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = 0.30;
    fragment.is_cacheable = true;

    fragment.payload = {
        {"user_id", json_string(event.payload, "user_id", "unknown")},
        {"username", json_string(event.payload, "username", "unknown")},
        {"affinity", 0.5},
        {"trust", 0.5},
        {"source", "phase4_placeholder"}
    };
    return fragment;
}

ContextFragment MemoryContextProvider::collect(const IEvent& event,
                                               const RuntimeState&) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = 0.40;
    fragment.is_cacheable = false;

    fragment.payload = {
        {"query_text", json_string(event.payload, "text")},
        {"memory_hits", nlohmann::json::array()},
        {"source", "phase4_no_ipc_adapter_yet"}
    };
    return fragment;
}

ContextFragment VoiceContextProvider::collect(const IEvent& event,
                                              const RuntimeState& state) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = event.type == EventType::UserSpeech ? 0.65 : 0.20;
    fragment.is_cacheable = false;

    fragment.payload = {
        {"is_listening", state.is_listening},
        {"is_speaking", state.is_speaking},
        {"voice_input", event.type == EventType::UserSpeech},
        {"source", "phase4_runtime_snapshot"}
    };
    return fragment;
}

ContextFragment VisionContextProvider::collect(const IEvent&,
                                               const RuntimeState&) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = 0.15;
    fragment.is_cacheable = true;

    fragment.payload = {
        {"vision_active", false},
        {"objects", nlohmann::json::array()},
        {"source", "phase4_no_backend_sync_snapshot_yet"}
    };
    return fragment;
}

ContextFragment CapabilityContextProvider::collect(const IEvent&,
                                                   const RuntimeState& state) {
    ContextFragment fragment;
    fragment.provider_name = name();
    fragment.produced_at = now();
    fragment.importance_hint = 0.55;
    fragment.is_cacheable = true;

    fragment.payload = {
        {"can_dispatch_actions", true},
        {"has_feedback_manager", true},
        {"runtime_mode", std::string(to_string(state.current_mode))},
        {"source", "phase4_core_capability_snapshot"}
    };
    return fragment;
}

} // namespace revia::core
