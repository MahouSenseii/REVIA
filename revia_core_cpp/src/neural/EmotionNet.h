#pragma once
#include <string>
#include "telemetry/telemetry.h"

namespace revia {

class EmotionNet {
public:
    EmotionNet() = default;
    EmotionOutput infer(const std::string& text);
    bool enabled = true;
    double last_inference_ms = 0;
};

} // namespace revia
