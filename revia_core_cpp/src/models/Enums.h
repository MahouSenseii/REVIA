// Phase 0 — Core enums. POD only. No behavior yet.
// Single-ownership: all core enums live here and only here.
#pragma once

#include <string>
#include <string_view>

namespace revia::core {

enum class EventType {
    UserText,
    UserSpeech,
    SilenceTimeout,
    PlatformEvent,
    ModelCompletion,
    FilterResult,
    MemoryUpdate,
    EmotionShift,
    Interruption,
    ConfigChange,
    Unknown
};

enum class EventSource {
    Discord,
    Twitch,
    LocalSTT,
    ControllerUI,
    InternalTimer,
    InternalModel,
    Plugin,
    Unknown
};

enum class CoreState {
    Booting,
    Idle,
    Listening,
    Thinking,
    Responding,
    Interrupted,
    Waiting,
    Proactive,
    Recovering,
    Disabled
};

enum class RuntimeMode {
    Normal,
    Focus,
    Performance,
    Recovery,
    Debug
};

enum class ActionType {
    SpeakResponse,
    SendTextResponse,
    AskClarifyingQuestion,
    WaitSilently,
    FollowUp,
    IgnoreEvent,
    TriggerFallback,
    UpdateInternalStateOnly,
    InterruptCurrentOutput,
    QueueProactiveMessage
};

enum class SafetyVerdict {
    Allowed,
    AllowedWithModification,
    Blocked,
    RequiresFallback
};

// ---------- Free functions for stringification (used by StructuredLogger) ----------

constexpr std::string_view to_string(EventType v) {
    switch (v) {
        case EventType::UserText:        return "UserText";
        case EventType::UserSpeech:      return "UserSpeech";
        case EventType::SilenceTimeout:  return "SilenceTimeout";
        case EventType::PlatformEvent:   return "PlatformEvent";
        case EventType::ModelCompletion: return "ModelCompletion";
        case EventType::FilterResult:    return "FilterResult";
        case EventType::MemoryUpdate:    return "MemoryUpdate";
        case EventType::EmotionShift:    return "EmotionShift";
        case EventType::Interruption:    return "Interruption";
        case EventType::ConfigChange:    return "ConfigChange";
        case EventType::Unknown:         return "Unknown";
    }
    return "Unknown";
}

constexpr std::string_view to_string(EventSource v) {
    switch (v) {
        case EventSource::Discord:       return "Discord";
        case EventSource::Twitch:        return "Twitch";
        case EventSource::LocalSTT:      return "LocalSTT";
        case EventSource::ControllerUI:  return "ControllerUI";
        case EventSource::InternalTimer: return "InternalTimer";
        case EventSource::InternalModel: return "InternalModel";
        case EventSource::Plugin:        return "Plugin";
        case EventSource::Unknown:       return "Unknown";
    }
    return "Unknown";
}

constexpr std::string_view to_string(CoreState v) {
    switch (v) {
        case CoreState::Booting:     return "Booting";
        case CoreState::Idle:        return "Idle";
        case CoreState::Listening:   return "Listening";
        case CoreState::Thinking:    return "Thinking";
        case CoreState::Responding:  return "Responding";
        case CoreState::Interrupted: return "Interrupted";
        case CoreState::Waiting:     return "Waiting";
        case CoreState::Proactive:   return "Proactive";
        case CoreState::Recovering:  return "Recovering";
        case CoreState::Disabled:    return "Disabled";
    }
    return "Unknown";
}

constexpr std::string_view to_string(RuntimeMode v) {
    switch (v) {
        case RuntimeMode::Normal:      return "Normal";
        case RuntimeMode::Focus:       return "Focus";
        case RuntimeMode::Performance: return "Performance";
        case RuntimeMode::Recovery:    return "Recovery";
        case RuntimeMode::Debug:       return "Debug";
    }
    return "Unknown";
}

constexpr std::string_view to_string(ActionType v) {
    switch (v) {
        case ActionType::SpeakResponse:           return "SpeakResponse";
        case ActionType::SendTextResponse:        return "SendTextResponse";
        case ActionType::AskClarifyingQuestion:   return "AskClarifyingQuestion";
        case ActionType::WaitSilently:            return "WaitSilently";
        case ActionType::FollowUp:                return "FollowUp";
        case ActionType::IgnoreEvent:             return "IgnoreEvent";
        case ActionType::TriggerFallback:         return "TriggerFallback";
        case ActionType::UpdateInternalStateOnly: return "UpdateInternalStateOnly";
        case ActionType::InterruptCurrentOutput:  return "InterruptCurrentOutput";
        case ActionType::QueueProactiveMessage:   return "QueueProactiveMessage";
    }
    return "Unknown";
}

constexpr std::string_view to_string(SafetyVerdict v) {
    switch (v) {
        case SafetyVerdict::Allowed:                 return "Allowed";
        case SafetyVerdict::AllowedWithModification: return "AllowedWithModification";
        case SafetyVerdict::Blocked:                 return "Blocked";
        case SafetyVerdict::RequiresFallback:        return "RequiresFallback";
    }
    return "Unknown";
}

} // namespace revia::core
