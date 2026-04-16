#include "internal/waveform.hpp"

#include "internal/common.hpp"
#include "internal/soxr_resampler.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace beat_this::preprocess::internal {

namespace {

void ValidateAudioBuffer(const AudioBufferView audio) {
  if (audio.layout != ChannelLayout::kInterleavedFrames) {
    ThrowInvalid("only interleaved audio buffers are supported");
  }
  if (audio.sample_rate <= 0) {
    ThrowInvalid("sample_rate must be positive");
  }
  if (audio.num_channels <= 0) {
    ThrowInvalid("num_channels must be positive");
  }
  if (audio.samples.empty()) {
    ThrowInvalid("audio buffer must be non-empty");
  }
  if ((audio.samples.size() % static_cast<std::size_t>(audio.num_channels)) != 0U) {
    ThrowInvalid("audio buffer size must be divisible by num_channels");
  }
  for (const float sample : audio.samples) {
    if (!std::isfinite(sample)) {
      ThrowInvalid("audio buffer must not contain NaN or Inf");
    }
  }
}

std::vector<double> NormalizeInterleavedToMono(const std::span<const float> interleaved,
                                               const int channels) {
  if (channels <= 0) {
    ThrowInvalid("channels must be positive");
  }
  if ((interleaved.size() % static_cast<std::size_t>(channels)) != 0U) {
    ThrowInvalid("interleaved audio size must be divisible by channel count");
  }

  const std::size_t num_frames = interleaved.size() / static_cast<std::size_t>(channels);
  std::vector<double> mono(num_frames, 0.0);

  for (std::size_t frame = 0; frame < num_frames; ++frame) {
    double sum = 0.0;
    const std::size_t frame_base = frame * static_cast<std::size_t>(channels);
    for (int channel = 0; channel < channels; ++channel) {
      sum += static_cast<double>(interleaved[frame_base + static_cast<std::size_t>(channel)]);
    }
    mono[frame] = sum / static_cast<double>(channels);
  }

  return mono;
}

std::vector<float> CastWaveformToFloat32(const std::span<const double> waveform) {
  std::vector<float> output(waveform.size(), 0.0F);
  for (std::size_t idx = 0; idx < waveform.size(); ++idx) {
    output[idx] = static_cast<float>(waveform[idx]);
  }
  return output;
}

}  // namespace

std::vector<float> PrepareMonoWaveform(const AudioBufferView audio) {
  ValidateAudioBuffer(audio);
  const std::vector<double> mono = NormalizeInterleavedToMono(audio.samples, audio.num_channels);
  const std::vector<double> resampled =
      ResampleMonoWaveformSoxr(mono, audio.sample_rate, kTargetSampleRate);
  return CastWaveformToFloat32(resampled);
}

}  // namespace beat_this::preprocess::internal
