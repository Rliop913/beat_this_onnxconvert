#pragma once

#include <span>
#include <vector>

namespace beat_this::preprocess::internal {

[[nodiscard]] std::vector<double> ResampleMonoWaveformSoxr(std::span<const double> mono,
                                                           int input_sample_rate,
                                                           int output_sample_rate);

}  // namespace beat_this::preprocess::internal
