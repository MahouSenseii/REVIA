#pragma once
#include <string>
#include "telemetry/telemetry.h"

namespace revia {

class RouterClassifier {
public:
    RouterClassifier() = default;
    RouterOutput classify(const std::string& text, const std::string& context,
                          const EmotionOutput& emotion);
    bool enabled = true;
    double last_inference_ms = 0;
};

} // namespace revia
