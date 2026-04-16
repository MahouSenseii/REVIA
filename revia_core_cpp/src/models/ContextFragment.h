// Phase 0 — Context fragment produced by a single IContextProvider.
// ContextManager collects fragments into a ContextPackage.
#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include "models/RuntimeState.h"  // Timestamp

namespace revia::core {

using json = nlohmann::json;

struct ContextFragment {
    std::string provider_name;   // e.g. "ConversationContextProvider"
    Timestamp   produced_at = Timestamp{};

    // Free-form payload; each provider defines its own shape.
    // ContextRanker reads a hint (see below) and does NOT need the full payload
    // to score importance.
    json payload = json::object();

    // Ranker hint. Providers may set this; if left at the default the ranker
    // will compute its own score from the payload.
    //
    //   0.0  = irrelevant (will be dropped)
    //   1.0  = maximally relevant
    double importance_hint = -1.0;  // negative sentinel = no hint

    // Set by the provider when the fragment was cheap to compute. Lets the
    // ContextManager cache it across multiple events if desired.
    bool is_cacheable = false;
};

} // namespace revia::core
