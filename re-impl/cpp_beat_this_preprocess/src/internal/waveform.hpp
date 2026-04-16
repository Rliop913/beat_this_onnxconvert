#pragma once

#include "beat_this_preprocess/preprocessor.hpp"

#include <vector>

namespace beat_this::preprocess::internal {

[[nodiscard]] std::vector<float> PrepareMonoWaveform(AudioBufferView audio);

}  // namespace beat_this::preprocess::internal
