// Phase 0 — IEvent contract. Matches REVIA_CORE.md §Example Interfaces.
#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include "models/Enums.h"
#include "models/RuntimeState.h"   // Timestamp

namespace revia::core {

using json = nlohmann::json;

// IEvent is a struct (POD-ish) rather than a class with virtuals because the
// spec models events as value types. If a future event subtype genuinely needs
// polymorphism we can promote this to a class; keeping it a struct keeps the
// EventBus copy-semantics simple.
struct IEvent {
    std::string id;                 // UUID-v4 string; EventBus fills if empty
    EventType   type   = EventType::Unknown;
    EventSource source = EventSource::Unknown;

    Timestamp   created_at = Timestamp{};

    // Typed payload; each EventType defines its own expected fields.
    // Examples:
    //   UserText      -> { "text": "hi", "user_id": "..." }
    //   UserSpeech    -> { "transcript": "hi", "confidence": 0.87 }
    //   Interruption  -> { "reason": "user_talked_over" }
    json payload = json::object();

    // Optional correlation id for tracing related events (e.g. a UserText
    // plus the ModelCompletion it triggered share a correlation_id).
    std::string correlation_id;

    virtual ~IEvent() = default;
};

} // namespace revia::core
