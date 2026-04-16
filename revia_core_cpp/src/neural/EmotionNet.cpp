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

    // Whole-word check: prevents "happy" matching "unhappily", "sad" matching "sadly", etc.
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

    // Count keyword hits per emotion class; confidence is derived from hit density rather
    // than being hardcoded, so multi-signal inputs produce a more reliable score.
    struct EmotionClass {
        const char* label;
        float valence, arousal, dominance;
        std::initializer_list<const char*> keywords;
    };

    static const EmotionClass kClasses[] = {
        {"Happy",   0.8f,  0.6f,  0.5f, {"happy", "great", "awesome", "love", "joy", "excited",
                                          "wonderful", "fantastic", "glad", "pleased", "thrilled"}},
        {"Angry",  -0.7f,  0.8f,  0.7f, {"angry", "hate", "furious", "rage", "mad", "annoyed",
                                          "irritated", "frustrated", "livid"}},
        {"Sad",    -0.6f,  0.3f,  0.2f, {"sad", "depressed", "sorry", "unhappy", "miserable",
                                          "upset", "heartbroken", "disappointed", "grief", "crying"}},
        {"Fear",   -0.5f,  0.7f,  0.1f, {"scared", "afraid", "fear", "terrified", "worried",
                                          "nervous", "anxious", "panic", "dread"}},
        {"Curious", 0.2f,  0.4f,  0.4f, {"curious", "wonder", "interesting", "question",
                                          "how", "why", "what", "explain", "tell me"}},
        {"Excited", 0.6f,  0.7f,  0.6f, {"excited", "amazing", "incredible", "unbelievable",
                                          "can't wait", "so cool", "wow", "omg"}},
    };

    int   best_hits  = 0;
    int   total_hits = 0;
    const EmotionClass* best = nullptr;

    for (const auto& cls : kClasses) {
        int hits = 0;
        for (const char* kw : cls.keywords)
            if (has_word(kw)) ++hits;
        total_hits += hits;
        if (hits > best_hits) { best_hits = hits; best = &cls; }
    }

    // Punctuation signals (weak, don't override lexical evidence)
    bool has_question = lower.find('?') != std::string::npos;
    bool has_exclaim  = lower.find('!') != std::string::npos;

    if (best && best_hits > 0) {
        // Base confidence: 0.60 for 1 hit, +0.08 per additional hit, capped at 0.92.
        // Reduce by 0.08 when a second emotion class also has hits (ambiguity / mixed signals).
        int competing = 0;
        for (const auto& cls : kClasses)
            if (&cls != best) {
                int h = 0;
                for (const char* kw : cls.keywords) if (has_word(kw)) ++h;
                if (h > 0) ++competing;
            }
        float conf = std::min(0.92f, 0.60f + (best_hits - 1) * 0.08f);
        if (competing > 0) conf = std::max(0.45f, conf - 0.08f * competing);
        out = {best->valence, best->arousal, best->dominance, best->label, conf, 0.0};
    } else if (has_question) {
        out = {0.1f, 0.4f, 0.3f, "Curious", 0.55f, 0.0};
    } else if (has_exclaim) {
        out = {0.3f, 0.6f, 0.5f, "Excited", 0.50f, 0.0};
    } else {
        out = {0.0f, 0.2f, 0.4f, "Neutral", 0.90f, 0.0};
    }

    auto end = std::chrono::high_resolution_clock::now();
    out.inference_ms = std::chrono::duration<double, std::milli>(end - start).count();
    last_inference_ms = out.inference_ms;
    return out;
}

} // namespace revia
