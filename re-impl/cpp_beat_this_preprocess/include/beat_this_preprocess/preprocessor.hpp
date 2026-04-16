#pragma once

#include <filesystem>
#include <span>
#include <vector>

namespace beat_this::preprocess {

enum class ChannelLayout {
  kInterleavedFrames,
};

struct AudioBufferView {
  std::span<const float> samples;
  int sample_rate = 22050;
  int num_channels = 1;
  ChannelLayout layout = ChannelLayout::kInterleavedFrames;
};

struct Spectrogram {
  int num_frames = 0;
  int num_bins = 128;
  std::vector<float> values;

  [[nodiscard]] bool empty() const noexcept { return values.empty(); }
};

class BeatThisPreprocessor {
 public:
  BeatThisPreprocessor() = default;

  [[nodiscard]] Spectrogram ProcessWaveform(AudioBufferView audio) const;
  [[nodiscard]] Spectrogram ProcessFile(const std::filesystem::path& path) const;
};

}  // namespace beat_this::preprocess
