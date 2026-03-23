#pragma once
#include <string>
#include <atomic>
#include "telemetry/telemetry.h"

namespace revia {

class RouterClassifier {
public:
    RouterClassifier() = default;
    RouterOutput classify(const std::string& text, const std::string& context,
                          const EmotionOutput& emotion);
    std::atomic<bool> enabled{true};
    std::atomic<double> last_inference_ms{0.0};
};

} // namespace revia
