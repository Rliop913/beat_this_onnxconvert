#include "internal/frontend.hpp"

#include "internal/common.hpp"

#include "STFT/GenOut/SERIAL/STFT_MAIN_SERIAL.hpp"
#include "stft_wrapper/MelFilterBank.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace beat_this::preprocess::internal {

namespace {

struct FrontendPlan {
  std::vector<float> window;
  std::vector<float> mel_filter_bank;
  float magnitude_normalization = 1.0F;
};

struct FrontendScratch {
  std::vector<float> frame;
  std::vector<float> real;
  std::vector<float> imag;
  std::vector<float> magnitude;
  std::vector<float> mel;
};

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

std::vector<float> BuildPeriodicHannWindow() {
  std::vector<float> window(static_cast<std::size_t>(kNfft), 0.0F);
  const float denom = static_cast<float>(kNfft);
  for (int idx = 0; idx < kNfft; ++idx) {
    const float phase = (2.0F * std::numbers::pi_v<float> * static_cast<float>(idx)) / denom;
    window[static_cast<std::size_t>(idx)] = 0.5F - (0.5F * std::cos(phase));
  }
  return window;
}

std::vector<float> BuildTorchaudioCompatibleMelFilterBank() {
  std::vector<float> filter_bank(static_cast<std::size_t>(kNumMels) * kFftBins, 0.0F);

  const float nyquist = static_cast<float>(kTargetSampleRate / 2);
  const float mel_min = PDJE_PARALLEL::detail::HzToMel(kFMinHz, PDJE_PARALLEL::MelFormula::Slaney);
  const float mel_max = PDJE_PARALLEL::detail::HzToMel(kFMaxHz, PDJE_PARALLEL::MelFormula::Slaney);
  const float mel_step = (mel_max - mel_min) / static_cast<float>(kNumMels + 1);

  std::vector<float> f_pts(static_cast<std::size_t>(kNumMels + 2), 0.0F);
  for (int idx = 0; idx < kNumMels + 2; ++idx) {
    const float mel_value = mel_min + (static_cast<float>(idx) * mel_step);
    f_pts[static_cast<std::size_t>(idx)] =
        PDJE_PARALLEL::detail::MelToHz(mel_value, PDJE_PARALLEL::MelFormula::Slaney);
  }

  std::vector<float> f_diff(static_cast<std::size_t>(kNumMels + 1), 0.0F);
  for (int idx = 0; idx < kNumMels + 1; ++idx) {
    f_diff[static_cast<std::size_t>(idx)] =
        f_pts[static_cast<std::size_t>(idx + 1)] - f_pts[static_cast<std::size_t>(idx)];
  }

  const float freq_step = nyquist / static_cast<float>(kFftBins - 1U);
  for (std::size_t fft_bin = 0; fft_bin < kFftBins; ++fft_bin) {
    const float bin_hz = static_cast<float>(fft_bin) * freq_step;
    for (int mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
      const float down_slope =
          (bin_hz - f_pts[static_cast<std::size_t>(mel_bin)]) /
          f_diff[static_cast<std::size_t>(mel_bin)];
      const float up_slope =
          (f_pts[static_cast<std::size_t>(mel_bin + 2)] - bin_hz) /
          f_diff[static_cast<std::size_t>(mel_bin + 1)];
      const float weight = std::max(0.0F, std::min(down_slope, up_slope));
      filter_bank[static_cast<std::size_t>(mel_bin) * kFftBins + fft_bin] = weight;
    }
  }

  return filter_bank;
}

FrontendPlan CreateFrontendPlan() {
  FrontendPlan plan{
      .window = BuildPeriodicHannWindow(),
      .mel_filter_bank = BuildTorchaudioCompatibleMelFilterBank(),
      .magnitude_normalization = std::sqrt(static_cast<float>(kNfft)),
  };
  if (plan.window.size() != static_cast<std::size_t>(kNfft) ||
      plan.mel_filter_bank.size() != (static_cast<std::size_t>(kNumMels) * kFftBins)) {
    ThrowRuntime("failed to initialize Beat This frontend plan");
  }
  return plan;
}

const FrontendPlan& GetFrontendPlan() {
  static const FrontendPlan plan = CreateFrontendPlan();
  return plan;
}

void ComputeMagnitudeBins(const FrontendPlan& plan, FrontendScratch& scratch) {
  for (std::size_t idx = 0; idx < scratch.frame.size(); ++idx) {
    scratch.real[idx] = scratch.frame[idx] * plan.window[idx];
  }
  std::fill(scratch.imag.begin(), scratch.imag.end(), 0.0F);

  Stockhoptimized10(scratch.real.data(),
                    scratch.imag.data(),
                    static_cast<unsigned int>(kNfft / 2));

  for (std::size_t bin = 0; bin < kFftBins; ++bin) {
    const float re = scratch.real[bin];
    const float im = scratch.imag[bin];
    scratch.magnitude[bin] = std::sqrt((re * re) + (im * im)) / plan.magnitude_normalization;
  }
}

void ComputeLogMelFrame(const FrontendPlan& plan, FrontendScratch& scratch) {
  for (int mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
    const std::size_t filter_base = static_cast<std::size_t>(mel_bin) * kFftBins;
    float sum = 0.0F;
    for (std::size_t fft_bin = 0; fft_bin < kFftBins; ++fft_bin) {
      sum += scratch.magnitude[fft_bin] * plan.mel_filter_bank[filter_base + fft_bin];
    }
    scratch.mel[static_cast<std::size_t>(mel_bin)] = std::max(sum, 0.0F);
  }

  for (std::size_t idx = 0; idx < scratch.mel.size(); ++idx) {
    scratch.mel[idx] = std::log1p(kLogMultiplier * scratch.mel[idx]);
  }
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

  const FrontendPlan& plan = GetFrontendPlan();
  const std::vector<float> padded =
      kCenterFrames ? ReflectPad(mono_waveform) : std::vector<float>(mono_waveform.begin(), mono_waveform.end());
  const std::size_t n_fft = static_cast<std::size_t>(kNfft);
  const std::size_t hop = static_cast<std::size_t>(kHopLength);
  if (padded.size() < n_fft) {
    ThrowRuntime("padded waveform shorter than FFT size");
  }

  const std::size_t num_frames = 1U + ((padded.size() - n_fft) / hop);
  std::vector<float> values(num_frames * static_cast<std::size_t>(kNumMels), 0.0F);

  FrontendScratch scratch{
      .frame = std::vector<float>(n_fft, 0.0F),
      .real = std::vector<float>(n_fft, 0.0F),
      .imag = std::vector<float>(n_fft, 0.0F),
      .magnitude = std::vector<float>(kFftBins, 0.0F),
      .mel = std::vector<float>(static_cast<std::size_t>(kNumMels), 0.0F),
  };

  for (std::size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
    const std::size_t start = frame_idx * hop;
    std::copy_n(padded.begin() + static_cast<std::ptrdiff_t>(start),
                static_cast<std::ptrdiff_t>(n_fft),
                scratch.frame.begin());

    ComputeMagnitudeBins(plan, scratch);
    ComputeLogMelFrame(plan, scratch);
    std::copy(scratch.mel.begin(),
              scratch.mel.end(),
              values.begin() +
                  static_cast<std::ptrdiff_t>(frame_idx * static_cast<std::size_t>(kNumMels)));
  }

  return Spectrogram{
      .num_frames = static_cast<int>(num_frames),
      .num_bins = kNumMels,
      .values = std::move(values),
  };
}

}  // namespace beat_this::preprocess::internal
