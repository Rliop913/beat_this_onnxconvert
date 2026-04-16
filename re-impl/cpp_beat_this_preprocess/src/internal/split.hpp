#pragma once

#include "beat_this_preprocess/preprocessor.hpp"

#include <vector>

namespace beat_this::inference::internal {

struct SpectrogramChunk {
  int start_frame = 0;
  int num_frames = 0;
  int num_bins = 0;
  std::vector<float> values;
};

[[nodiscard]] std::vector<SpectrogramChunk> SplitSpectrogram(
    const beat_this::preprocess::Spectrogram& spectrogram);

}  // namespace beat_this::inference::internal

