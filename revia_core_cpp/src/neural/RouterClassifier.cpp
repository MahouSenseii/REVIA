#include "neural/RouterClassifier.h"
#include <chrono>
#include <algorithm>
#include <thread>
#include <random>

namespace revia {

// Thread-local Mersenne Twister — avoids the data races inherent in std::rand().
static thread_local std::mt19937 tl_rng{std::random_device{}()};

RouterOutput RouterClassifier::classify(const std::string& text, const std::string&,
                                         const EmotionOutput&) {
    auto start = std::chrono::high_resolution_clock::now();
    RouterOutput out;

    if (!enabled) { out.mode = "chat"; out.confidence = 0.5f; return out; }

    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("search") != std::string::npos || lower.find("find") != std::string::npos
        || lower.find("look up") != std::string::npos) {
        out = {"memory_query", 0.85f, "rag_search", true, 0.9f, 0.0};
    } else if (lower.find("run") != std::string::npos || lower.find("execute") != std::string::npos
               || lower.find("open") != std::string::npos) {
        out = {"command", 0.80f, "system_exec", false, 0.1f, 0.0};
    } else if (lower.find("see") != std::string::npos || lower.find("look at") != std::string::npos
               || lower.find("camera") != std::string::npos || lower.find("image") != std::string::npos) {
        out = {"vision_query", 0.78f, "vision_capture", false, 0.15f, 0.0};
    } else if (lower.find("remember") != std::string::npos || lower.find("recall") != std::string::npos) {
        out = {"memory_query", 0.82f, "memory_recall", true, 0.85f, 0.0};
    } else if (lower.find("use tool") != std::string::npos || lower.find("call") != std::string::npos) {
        out = {"tool_call", 0.75f, "", false, 0.1f, 0.0};
    } else if (lower.find("say") != std::string::npos || lower.find("speak") != std::string::npos) {
        out = {"voice_only", 0.80f, "tts", false, 0.1f, 0.0};
    } else {
        out = {"chat", 0.92f, "", false, 0.2f, 0.0};
    }

    // Confidence threshold: fall back to chat if uncertain
    if (out.confidence < 0.65f) {
        out.mode = "chat";
        out.suggested_tool = "";
    }

    auto end = std::chrono::high_resolution_clock::now();
    out.inference_ms = std::chrono::duration<double, std::milli>(end - start).count();
    last_inference_ms = out.inference_ms;
    return out;
}

} // namespace revia
