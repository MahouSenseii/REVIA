#include "neural/RouterClassifier.h"
#include <chrono>
#include <algorithm>
#include <thread>
#include <random>

namespace revia {

// Thread-local Mersenne Twister — avoids the data races inherent in std::rand().
static thread_local std::mt19937 tl_rng{std::random_device{}()};

RouterOutput RouterClassifier::classify(const std::string& text,
                                         const std::string& context,
                                         const EmotionOutput& emotion) {
    auto start = std::chrono::high_resolution_clock::now();
    RouterOutput out;

    if (!enabled) { out.mode = "chat"; out.confidence = 0.5f; return out; }

    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) -> unsigned char {
                       return (c < 128) ? static_cast<unsigned char>(std::tolower(c)) : c;
                   });

    // Whole-word check: prevents "search" from matching inside "researching",
    // "open" from matching "openly", etc.  A word boundary is a position where
    // the adjacent character is non-alpha (or the string edge).
    auto has_word = [&lower](const std::string& word) -> bool {
        std::size_t pos = lower.find(word);
        while (pos != std::string::npos) {
            bool left_ok  = (pos == 0)
                            || !std::isalpha(static_cast<unsigned char>(lower[pos - 1]));
            bool right_ok = (pos + word.size() >= lower.size())
                            || !std::isalpha(static_cast<unsigned char>(lower[pos + word.size()]));
            if (left_ok && right_ok) return true;
            pos = lower.find(word, pos + 1);
        }
        return false;
    };

    if (has_word("search") || has_word("find") || has_word("look up")) {
        out = {"memory_query", 0.85f, "rag_search", true, 0.9f, 0.0};
    } else if (has_word("run") || has_word("execute") || has_word("open")) {
        out = {"command", 0.80f, "system_exec", false, 0.1f, 0.0};
    } else if (has_word("see") || has_word("look at") || has_word("camera") || has_word("image")) {
        out = {"vision_query", 0.78f, "vision_capture", false, 0.15f, 0.0};
    } else if (has_word("remember") || has_word("recall")) {
        out = {"memory_query", 0.82f, "memory_recall", true, 0.85f, 0.0};
    } else if (has_word("use tool") || has_word("call")) {
        out = {"tool_call", 0.75f, "", false, 0.1f, 0.0};
    } else if (has_word("say") || has_word("speak")) {
        out = {"voice_only", 0.80f, "tts", false, 0.1f, 0.0};
    } else {
        out = {"chat", 0.92f, "", false, 0.2f, 0.0};
    }

    // Use the emotion parameter: a curious emotional state raises confidence on
    // vision and memory queries; an excited state slightly lowers it (could be
    // exclamatory chat rather than a deliberate query).
    if (!emotion.label.empty() && emotion.label != "Disabled") {
        if (emotion.label == "Curious") {
            if (out.mode == "vision_query" || out.mode == "memory_query") {
                out.confidence = std::min(1.0f, out.confidence + 0.05f);
            }
        } else if (emotion.label == "Excited") {
            // Excited speech is more likely to be chat than a structured query.
            if (out.mode != "chat") {
                out.confidence = std::max(0.0f, out.confidence - 0.04f);
            }
        }
    }

    // Use the context parameter: if additional context is provided and it is
    // non-trivial in length, the user is likely asking a complex question —
    // slightly lower confidence in non-chat routing to prefer careful handling.
    if (context.size() > 30 && out.mode != "chat") {
        out.confidence = std::max(0.0f, out.confidence - 0.03f);
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
