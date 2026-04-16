#include "beat_this_preprocess/preprocessor.hpp"
#include "internal/soxr_resampler.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using beat_this::preprocess::AudioBufferView;
using beat_this::preprocess::BeatThisPreprocessor;
using beat_this::preprocess::ChannelLayout;
using beat_this::preprocess::Spectrogram;
using beat_this::preprocess::internal::ResampleMonoWaveformSoxr;

void Require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void RequireAllFinite(const Spectrogram& spect) {
  for (const float value : spect.values) {
    Require(std::isfinite(value), "spectrogram contains non-finite values");
  }
}

template <typename Fn>
void RequireThrowsInvalid(Fn&& fn, const std::string& message) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename Fn>
void RequireThrowsRuntime(Fn&& fn, const std::string& message) {
  try {
    fn();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

void RequireNear(const Spectrogram& lhs,
                 const Spectrogram& rhs,
                 const float tolerance,
                 const std::string& message) {
  Require(lhs.num_frames == rhs.num_frames, message + ": frame count mismatch");
  Require(lhs.num_bins == rhs.num_bins, message + ": bin count mismatch");
  Require(lhs.values.size() == rhs.values.size(), message + ": flattened size mismatch");

  for (std::size_t idx = 0; idx < lhs.values.size(); ++idx) {
    if (std::abs(lhs.values[idx] - rhs.values[idx]) > tolerance) {
      throw std::runtime_error(message + ": value mismatch");
    }
  }
}

std::vector<float> MakeMonoSine(const int sample_rate, const int length, const float frequency) {
  std::vector<float> waveform(static_cast<std::size_t>(length), 0.0F);
  for (int idx = 0; idx < length; ++idx) {
    const float phase = (2.0F * std::numbers::pi_v<float> * frequency * static_cast<float>(idx)) /
                        static_cast<float>(sample_rate);
    waveform[static_cast<std::size_t>(idx)] = std::sin(phase);
  }
  return waveform;
}

std::vector<double> MakeMonoSineDouble(const int sample_rate, const int length, const double frequency) {
  std::vector<double> waveform(static_cast<std::size_t>(length), 0.0);
  for (int idx = 0; idx < length; ++idx) {
    const double phase = (2.0 * std::numbers::pi_v<double> * frequency * static_cast<double>(idx)) /
                         static_cast<double>(sample_rate);
    waveform[static_cast<std::size_t>(idx)] = std::sin(phase);
  }
  return waveform;
}

void RequireEqual(const std::vector<double>& lhs,
                  const std::vector<double>& rhs,
                  const std::string& message) {
  Require(lhs.size() == rhs.size(), message + ": size mismatch");
  for (std::size_t idx = 0; idx < lhs.size(); ++idx) {
    if (lhs[idx] != rhs[idx]) {
      throw std::runtime_error(message + ": value mismatch");
    }
  }
}

void RequireAllFinite(const std::vector<double>& waveform, const std::string& message) {
  for (const double sample : waveform) {
    Require(std::isfinite(sample), message);
  }
}

void TestSoxrCopyPathPreservesValues() {
  const std::vector<double> mono = {0.0, 0.5, -0.25, 1.0, -1.0, 0.125};
  const std::vector<double> resampled = ResampleMonoWaveformSoxr(mono, 22050, 22050);
  RequireEqual(resampled, mono, "soxr copy path should preserve values");
}

void TestSoxrResamplePathProducesFiniteOutput() {
  const std::vector<double> mono = MakeMonoSineDouble(44100, 44100, 440.0);
  const std::vector<double> resampled = ResampleMonoWaveformSoxr(mono, 44100, 22050);
  Require(!resampled.empty(), "soxr resample path should produce output");
  RequireAllFinite(resampled, "soxr resample path should produce finite output");
}

void TestSoxrResamplePathIsDeterministic() {
  const std::vector<double> mono = MakeMonoSineDouble(44100, 12000, 440.0);
  const std::vector<double> first = ResampleMonoWaveformSoxr(mono, 44100, 22050);
  const std::vector<double> second = ResampleMonoWaveformSoxr(mono, 44100, 22050);
  RequireEqual(first, second, "soxr resample path should be deterministic");
}

void TestSoxrRejectsInvalidInput() {
  RequireThrowsInvalid(
      [&]() {
        const std::vector<double> empty;
        static_cast<void>(ResampleMonoWaveformSoxr(empty, 22050, 22050));
      },
      "empty soxr input should throw invalid_argument");

  RequireThrowsInvalid(
      [&]() {
        const std::vector<double> mono = {0.0, 1.0};
        static_cast<void>(ResampleMonoWaveformSoxr(mono, 0, 22050));
      },
      "non-positive soxr input rate should throw invalid_argument");

  RequireThrowsInvalid(
      [&]() {
        const std::vector<double> mono = {0.0, 1.0};
        static_cast<void>(ResampleMonoWaveformSoxr(mono, 22050, 0));
      },
      "non-positive soxr output rate should throw invalid_argument");
}

void TestMonoWaveform() {
  const BeatThisPreprocessor preprocessor;
  const std::vector<float> mono = MakeMonoSine(22050, 22050, 440.0F);
  const Spectrogram spect = preprocessor.ProcessWaveform(AudioBufferView{
      .samples = std::span<const float>(mono),
      .sample_rate = 22050,
      .num_channels = 1,
      .layout = ChannelLayout::kInterleavedFrames,
  });

  Require(spect.num_bins == 128, "expected 128 mel bins");
  Require(spect.num_frames == 51, "expected floor(len/hop)+1 frames for 1 second clip");
  Require(!spect.values.empty(), "spectrogram must not be empty");
  RequireAllFinite(spect);
}

void TestStereoAveragingMatchesManualMono() {
  const BeatThisPreprocessor preprocessor;
  const std::vector<float> left = MakeMonoSine(22050, 4096, 220.0F);
  const std::vector<float> right = MakeMonoSine(22050, 4096, 330.0F);

  std::vector<float> stereo;
  stereo.reserve(left.size() * 2U);
  std::vector<float> mono(left.size(), 0.0F);

  for (std::size_t idx = 0; idx < left.size(); ++idx) {
    stereo.push_back(left[idx]);
    stereo.push_back(right[idx]);
    mono[idx] = (left[idx] + right[idx]) * 0.5F;
  }

  const Spectrogram stereo_spect = preprocessor.ProcessWaveform(AudioBufferView{
      .samples = std::span<const float>(stereo),
      .sample_rate = 22050,
      .num_channels = 2,
      .layout = ChannelLayout::kInterleavedFrames,
  });
  const Spectrogram mono_spect = preprocessor.ProcessWaveform(AudioBufferView{
      .samples = std::span<const float>(mono),
      .sample_rate = 22050,
      .num_channels = 1,
      .layout = ChannelLayout::kInterleavedFrames,
  });

  Require(stereo_spect.num_frames == mono_spect.num_frames, "frame count mismatch");
  Require(stereo_spect.values.size() == mono_spect.values.size(), "spectrogram size mismatch");
  RequireNear(stereo_spect, mono_spect, 1.0e-6F, "stereo average path diverged from manual mono path");
}

void TestResamplePath() {
  const BeatThisPreprocessor preprocessor;
  const std::vector<float> mono = MakeMonoSine(44100, 44100, 440.0F);
  const Spectrogram spect = preprocessor.ProcessWaveform(AudioBufferView{
      .samples = std::span<const float>(mono),
      .sample_rate = 44100,
      .num_channels = 1,
      .layout = ChannelLayout::kInterleavedFrames,
  });

  Require(spect.num_bins == 128, "expected 128 mel bins after resample");
  Require(spect.num_frames > 0, "resampled spectrogram should have frames");
  RequireAllFinite(spect);
}

void TestProcessWaveformIsDeterministic() {
  const BeatThisPreprocessor preprocessor;
  const std::vector<float> mono = MakeMonoSine(44100, 12000, 440.0F);

  const Spectrogram first = preprocessor.ProcessWaveform(AudioBufferView{
      .samples = std::span<const float>(mono),
      .sample_rate = 44100,
      .num_channels = 1,
      .layout = ChannelLayout::kInterleavedFrames,
  });
  const Spectrogram second = preprocessor.ProcessWaveform(AudioBufferView{
      .samples = std::span<const float>(mono),
      .sample_rate = 44100,
      .num_channels = 1,
      .layout = ChannelLayout::kInterleavedFrames,
  });

  Require(first.num_frames == second.num_frames, "deterministic waveform path frame count mismatch");
  Require(first.num_bins == second.num_bins, "deterministic waveform path bin count mismatch");
  Require(first.values == second.values, "deterministic waveform path values mismatch");
}

void TestInvalidWaveformInputs() {
  const BeatThisPreprocessor preprocessor;
  const std::vector<float> mono = MakeMonoSine(22050, 1024, 440.0F);

  RequireThrowsInvalid(
      [&]() {
        const std::vector<float> empty;
        static_cast<void>(preprocessor.ProcessWaveform(AudioBufferView{
            .samples = std::span<const float>(empty),
            .sample_rate = 22050,
            .num_channels = 1,
            .layout = ChannelLayout::kInterleavedFrames,
        }));
      },
      "empty waveform should throw invalid_argument");

  RequireThrowsInvalid(
      [&]() {
        static_cast<void>(preprocessor.ProcessWaveform(AudioBufferView{
            .samples = std::span<const float>(mono),
            .sample_rate = 0,
            .num_channels = 1,
            .layout = ChannelLayout::kInterleavedFrames,
        }));
      },
      "non-positive sample rate should throw invalid_argument");

  RequireThrowsInvalid(
      [&]() {
        static_cast<void>(preprocessor.ProcessWaveform(AudioBufferView{
            .samples = std::span<const float>(mono),
            .sample_rate = 22050,
            .num_channels = 0,
            .layout = ChannelLayout::kInterleavedFrames,
        }));
      },
      "non-positive channel count should throw invalid_argument");

  RequireThrowsInvalid(
      [&]() {
        std::vector<float> bad = mono;
        bad[0] = std::numeric_limits<float>::quiet_NaN();
        static_cast<void>(preprocessor.ProcessWaveform(AudioBufferView{
            .samples = std::span<const float>(bad),
            .sample_rate = 22050,
            .num_channels = 1,
            .layout = ChannelLayout::kInterleavedFrames,
        }));
      },
      "NaN waveform should throw invalid_argument");

  RequireThrowsInvalid(
      [&]() {
        std::vector<float> bad = mono;
        bad[0] = std::numeric_limits<float>::infinity();
        static_cast<void>(preprocessor.ProcessWaveform(AudioBufferView{
            .samples = std::span<const float>(bad),
            .sample_rate = 22050,
            .num_channels = 1,
            .layout = ChannelLayout::kInterleavedFrames,
        }));
      },
      "Inf waveform should throw invalid_argument");

  RequireThrowsInvalid(
      [&]() {
        const std::vector<float> bad = {0.0F, 1.0F, 2.0F};
        static_cast<void>(preprocessor.ProcessWaveform(AudioBufferView{
            .samples = std::span<const float>(bad),
            .sample_rate = 22050,
            .num_channels = 2,
            .layout = ChannelLayout::kInterleavedFrames,
        }));
      },
      "non-divisible interleaved buffer should throw invalid_argument");

  RequireThrowsInvalid(
      [&]() {
        const std::vector<float> too_short(512U, 0.0F);
        static_cast<void>(preprocessor.ProcessWaveform(AudioBufferView{
            .samples = std::span<const float>(too_short),
            .sample_rate = 22050,
            .num_channels = 1,
            .layout = ChannelLayout::kInterleavedFrames,
        }));
      },
      "short waveform should throw invalid_argument");
}

void TestMissingFileThrowsRuntime() {
  const BeatThisPreprocessor preprocessor;
  const std::filesystem::path missing_path =
      std::filesystem::path("tests") / "__missing_preprocess_fixture__.mp3";
  RequireThrowsRuntime(
      [&]() {
        static_cast<void>(preprocessor.ProcessFile(missing_path));
      },
      "missing file should throw runtime_error");
}

void TestProcessFile(const std::filesystem::path& audio_path) {
  const BeatThisPreprocessor preprocessor;
  const Spectrogram first = preprocessor.ProcessFile(audio_path);
  const Spectrogram second = preprocessor.ProcessFile(audio_path);

  Require(first.num_bins == 128, "file path should produce 128 mel bins");
  Require(first.num_frames > 0, "file path should produce frames");
  RequireAllFinite(first);
  RequireNear(first, second, 1.0e-6F, "file path should be repeatable");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    TestSoxrCopyPathPreservesValues();
    TestSoxrResamplePathProducesFiniteOutput();
    TestSoxrResamplePathIsDeterministic();
    TestSoxrRejectsInvalidInput();
    TestMonoWaveform();
    TestStereoAveragingMatchesManualMono();
    TestResamplePath();
    TestProcessWaveformIsDeterministic();
    TestInvalidWaveformInputs();
    TestMissingFileThrowsRuntime();

    if (argc >= 2) {
      TestProcessFile(argv[1]);
    }

    std::cout << "all preprocessing tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
