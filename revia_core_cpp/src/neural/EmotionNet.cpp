#include "neural/EmotionNet.h"
#include <chrono>
#include <algorithm>
#include <thread>
#include <random>

namespace revia {

// Thread-local Mersenne Twister — each thread gets its own seeded instance,
// avoiding data races that arise from the shared mutable state of std::rand().
static thread_local std::mt19937 tl_rng{std::random_device{}()};

EmotionOutput EmotionNet::infer(const std::string& text) {
    auto start = std::chrono::high_resolution_clock::now();
    EmotionOutput out;

    if (!enabled) { out.label = "Disabled"; return out; }

    std::string lower = text;
    // Use ASCII-safe tolower that only affects A-Z characters
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    });

    // TODO: Replace keyword matching with ONNX Runtime model for proper emotion classification
    // Model: fine-tuned DistilBERT or similar, targeting <5ms inference with TensorRT

    if (lower.find("happy") != std::string::npos || lower.find("great") != std::string::npos
        || lower.find("awesome") != std::string::npos || lower.find("love") != std::string::npos) {
        out = {0.8f, 0.6f, 0.5f, "Happy", 0.82f, 0.0};
    } else if (lower.find("angry") != std::string::npos || lower.find("hate") != std::string::npos
               || lower.find("furious") != std::string::npos) {
        out = {-0.7f, 0.8f, 0.7f, "Angry", 0.78f, 0.0};
    } else if (lower.find("sad") != std::string::npos || lower.find("depressed") != std::string::npos
               || lower.find("sorry") != std::string::npos) {
        out = {-0.6f, 0.3f, 0.2f, "Sad", 0.75f, 0.0};
    } else if (lower.find("scared") != std::string::npos || lower.find("afraid") != std::string::npos) {
        out = {-0.5f, 0.7f, 0.1f, "Fear", 0.72f, 0.0};
    } else if (lower.find("?") != std::string::npos) {
        out = {0.1f, 0.4f, 0.3f, "Curious", 0.68f, 0.0};
    } else if (lower.find("!") != std::string::npos) {
        out = {0.3f, 0.6f, 0.5f, "Excited", 0.65f, 0.0};
    } else {
        out = {0.0f, 0.2f, 0.4f, "Neutral", 0.90f, 0.0};
    }

    auto end = std::chrono::high_resolution_clock::now();
    out.inference_ms = std::chrono::duration<double, std::milli>(end - start).count();
    last_inference_ms = out.inference_ms;
    return out;
}

} // namespace revia
