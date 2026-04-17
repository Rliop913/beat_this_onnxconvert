#include "internal/frontend.hpp"

#include "internal/common.hpp"

#include "stft_wrapper/ConfigurableSerialStft.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace beat_this::preprocess::internal {

namespace {

std::vector<float> ReflectPad(const std::span<const float> input) {
  if (kPad == 0) {
    return std::vector<float>(input.begin(), input.end());
  }
  if (input.size() <= 1U) {
    ThrowInvalid("reflect padding requires at least two samples");
  }
  if (static_cast<std::size_t>(kPad) >= input.size()) {
    ThrowInvalid("reflect padding requires input longer than pad");
  }

  std::vector<float> padded(input.size() + (2U * static_cast<std::size_t>(kPad)), 0.0F);
  const std::size_t left_pad = static_cast<std::size_t>(kPad);

  for (std::size_t idx = 0; idx < left_pad; ++idx) {
    padded[idx] = input[left_pad - idx];
  }
  std::copy(input.begin(), input.end(), padded.begin() + static_cast<std::ptrdiff_t>(left_pad));
  for (std::size_t idx = 0; idx < left_pad; ++idx) {
    padded[left_pad + input.size() + idx] = input[input.size() - 2U - idx];
  }

  return padded;
}

PDJE_PARALLEL::SerialStftConfig CreateFrontendConfig() {
  return PDJE_PARALLEL::SerialStftConfig{
      .sample_rate = kTargetSampleRate,
      .window_size_exp = 10U,
      .hop_length = static_cast<unsigned int>(kHopLength),
      .mel_bins = static_cast<unsigned int>(kNumMels),
      .f_min_hz = kFMinHz,
      .f_max_hz = kFMaxHz,
      .mel_formula = PDJE_PARALLEL::MelFormula::Slaney,
      .mel_norm = PDJE_PARALLEL::MelNorm::None,
      .window_mode = PDJE_PARALLEL::SerialWindowMode::HanningPeriodic,
      .remove_dc = false,
      .normalize_by_sqrt_window = true,
      .output_mode = PDJE_PARALLEL::SerialOutputMode::Mel,
  };
}

const PDJE_PARALLEL::SerialStftConfig& GetFrontendConfig() {
  static const PDJE_PARALLEL::SerialStftConfig config = CreateFrontendConfig();
  return config;
}

PDJE_PARALLEL::ConfigurableSerialStft& GetFrontendEngine() {
  static PDJE_PARALLEL::ConfigurableSerialStft engine;
  return engine;
}

}  // namespace

Spectrogram ComputeLogMelSpectrogram(const std::span<const float> mono_waveform) {
  if (mono_waveform.empty()) {
    ThrowInvalid("mono waveform must be non-empty");
  }
  for (const float sample : mono_waveform) {
    if (!std::isfinite(sample)) {
      ThrowInvalid("mono waveform must not contain NaN or Inf");
    }
  }

  const std::vector<float> padded =
      kCenterFrames ? ReflectPad(mono_waveform) : std::vector<float>(mono_waveform.begin(), mono_waveform.end());
  if (padded.size() < static_cast<std::size_t>(kNfft)) {
    ThrowRuntime("padded waveform shorter than FFT size");
  }

  Spectrogram spectrogram;
  const auto linear_mel = GetFrontendEngine().Compute(padded, GetFrontendConfig());
  if (linear_mel.num_bins != kNumMels) {
    ThrowRuntime("configured serial STFT returned unexpected mel bin count");
  }
  spectrogram.num_frames = linear_mel.num_frames;
  spectrogram.num_bins = linear_mel.num_bins;
  spectrogram.values = linear_mel.values;
  for (float& value : spectrogram.values) {
    value = std::log1p(kLogMultiplier * std::max(value, 0.0F));
  }

  return spectrogram;
}

}  // namespace beat_this::preprocess::internal
