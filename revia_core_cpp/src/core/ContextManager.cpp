// Phase 2 - ContextManager implementation.
#include "core/ContextManager.h"

#include <exception>
#include <stdexcept>
#include <utility>

#include "core/StructuredLogger.h"

namespace revia::core {

ContextManager::ContextManager(ContextRanker ranker)
    : ranker_(std::move(ranker)) {}

void ContextManager::register_provider(std::unique_ptr<IContextProvider> provider) {
    if (!provider) {
        throw std::invalid_argument("ContextManager::register_provider received null provider");
    }

    StructuredLogger::instance().info("context.provider_registered", {
        {"provider", provider->name()},
        {"budget_ms_hint", provider->budget_ms_hint()}
    });
    providers_.push_back(std::move(provider));
}

ContextPackage ContextManager::build_context(const IEvent& input_event,
                                             const RuntimeState& runtime_state) {
    ContextPackage package;
    package.current_event = &input_event;
    package.state_snapshot = runtime_state;

    StructuredLogger::instance().debug("context.build_started", {
        {"event_id", input_event.id},
        {"event_type", std::string(to_string(input_event.type))},
        {"provider_count", providers_.size()}
    });

    for (const auto& provider : providers_) {
        try {
            ContextFragment fragment = provider->collect(input_event, runtime_state);
            if (fragment.provider_name.empty()) {
                fragment.provider_name = provider->name();
            }
            if (fragment.produced_at == Timestamp{}) {
                fragment.produced_at = now();
            }
            if (fragment.payload.empty() && fragment.importance_hint <= 0.0) {
                StructuredLogger::instance().debug("context.provider_empty", {
                    {"event_id", input_event.id},
                    {"provider", provider->name()}
                });
                continue;
            }

            package.add(std::move(fragment));
            StructuredLogger::instance().debug("context.provider_collected", {
                {"event_id", input_event.id},
                {"provider", provider->name()}
            });
        } catch (const std::exception& exc) {
            StructuredLogger::instance().warn("context.provider_failed", {
                {"event_id", input_event.id},
                {"provider", provider->name()},
                {"error", exc.what()}
            });
        } catch (...) {
            StructuredLogger::instance().warn("context.provider_failed", {
                {"event_id", input_event.id},
                {"provider", provider->name()},
                {"error", "unknown exception"}
            });
        }
    }

    package.ranked_signals = ranker_.score(package, runtime_state);
    StructuredLogger::instance().info("context.build_completed", {
        {"event_id", input_event.id},
        {"fragments", package.fragments.size()},
        {"ranked_signals", package.ranked_signals.size()}
    });

    return package;
}

} // namespace revia::core
