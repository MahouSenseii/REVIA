// Phase 2 - first-day context providers.
//
// These are intentionally thin C++ providers. They give the orchestrator a
// ranked context package without pulling ownership from Python adapters yet.
#pragma once

#include <string>

#include "interfaces/IContextProvider.h"
#include "interfaces/IEvent.h"

namespace revia::core {

class ConversationContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "ConversationContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

class ProfileContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "ProfileContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

class PlatformContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "PlatformContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

class EmotionContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "EmotionContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

class RelationshipContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "RelationshipContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

class MemoryContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "MemoryContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

class VoiceContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "VoiceContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

class VisionContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "VisionContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

class CapabilityContextProvider final : public IContextProvider {
public:
    [[nodiscard]] std::string name() const override {
        return "CapabilityContextProvider";
    }

    [[nodiscard]] ContextFragment collect(const IEvent& event,
                                          const RuntimeState& state) override;
};

} // namespace revia::core
