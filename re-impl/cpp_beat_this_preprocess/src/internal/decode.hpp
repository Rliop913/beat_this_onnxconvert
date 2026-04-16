#pragma once

#include <filesystem>
#include <vector>

namespace beat_this::preprocess::internal {

struct DecodedAudio {
  int sample_rate = 0;
  int num_channels = 0;
  std::vector<float> interleaved_samples;
};

[[nodiscard]] DecodedAudio DecodeFileToInterleavedF32(const std::filesystem::path& path);

}  // namespace beat_this::preprocess::internal
