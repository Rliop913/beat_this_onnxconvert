#pragma once

#include "beat_this_preprocess/preprocessor.hpp"

#include <span>

namespace beat_this::preprocess::internal {

[[nodiscard]] Spectrogram ComputeLogMelSpectrogram(std::span<const float> mono_waveform);

}  // namespace beat_this::preprocess::internal
