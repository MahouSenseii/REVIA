#pragma once
#include <string>
#include <atomic>
#include "telemetry/telemetry.h"

namespace revia {

class EmotionNet {
public:
    EmotionNet() = default;
    EmotionOutput infer(const std::string& text);
    std::atomic<bool> enabled{true};
    std::atomic<double> last_inference_ms{0.0};
};

} // namespace revia
