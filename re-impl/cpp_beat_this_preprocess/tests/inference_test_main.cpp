#include "beat_this_inference/inference.hpp"
#include "beat_this_preprocess/preprocessor.hpp"
#include "internal/aggregate.hpp"
#include "internal/split.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using beat_this::inference::BeatThisOnnxRunner;
using beat_this::inference::BeatTimestamps;
using beat_this::inference::FrameLogits;
using beat_this::inference::InferBeatNumbers;
using beat_this::inference::MinimalBeatPostprocessor;
using beat_this::inference::WriteBeatTsv;
using beat_this::inference::internal::AggregateChunkLogits;
using beat_this::inference::internal::ChunkFrameLogits;
using beat_this::inference::internal::SplitSpectrogram;
using beat_this::preprocess::AudioBufferView;
using beat_this::preprocess::ChannelLayout;
using beat_this::preprocess::Spectrogram;

constexpr double kDefaultFps = 50.0;

void Require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
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

void RequireAllFinite(const FrameLogits& logits, const std::string& message) {
  Require(logits.beat.size() == logits.downbeat.size(), message + ": beat/downbeat size mismatch");
  Require(logits.beat.size() == static_cast<std::size_t>(logits.num_frames),
          message + ": frame count mismatch");
  for (const float value : logits.beat) {
    Require(std::isfinite(value), message + ": non-finite beat logit");
  }
  for (const float value : logits.downbeat) {
    Require(std::isfinite(value), message + ": non-finite downbeat logit");
  }
}

void RequireNear(const double actual,
                 const double expected,
                 const double tolerance,
                 const std::string& message) {
  Require(std::abs(actual - expected) <= tolerance, message);
}

void RequireTimesEqual(const std::span<const double> actual,
                       const std::span<const double> expected,
                       const std::string& message) {
  Require(actual.size() == expected.size(), message + ": size mismatch");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    RequireNear(actual[index], expected[index], 1e-12, message + ": value mismatch");
  }
}

void RequireNumbersEqual(const std::span<const int> actual,
                         const std::span<const int> expected,
                         const std::string& message) {
  Require(actual.size() == expected.size(), message + ": size mismatch");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    Require(actual[index] == expected[index], message + ": value mismatch");
  }
}

void RequireTimestampsEqual(const BeatTimestamps& actual,
                            const BeatTimestamps& expected,
                            const std::string& message) {
  RequireTimesEqual(actual.beats, expected.beats, message + ": beats");
  RequireTimesEqual(actual.downbeats, expected.downbeats, message + ": downbeats");
}

std::vector<std::string> ReadNonEmptyLines(const std::filesystem::path& path) {
  std::ifstream input(path);
  Require(static_cast<bool>(input), "failed to open beats tsv");

  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

Spectrogram MakeDeterministicSpectrogram(const int num_frames) {
  Spectrogram spect{
      .num_frames = num_frames,
      .num_bins = 128,
      .values = std::vector<float>(static_cast<std::size_t>(num_frames) * 128U, 0.0F),
  };
  for (int frame = 0; frame < num_frames; ++frame) {
    for (int bin = 0; bin < spect.num_bins; ++bin) {
      const float value_index = static_cast<float>((frame * spect.num_bins) + bin);
      spect.values[static_cast<std::size_t>(frame * spect.num_bins + bin)] =
          std::sin(value_index / 31.0F) + (0.5F * std::cos(value_index / 17.0F));
    }
  }
  return spect;
}

std::vector<float> MakeStereoWaveform(const int sample_rate, const float duration_seconds) {
  const int num_frames = static_cast<int>(duration_seconds * static_cast<float>(sample_rate));
  std::vector<float> waveform;
  waveform.reserve(static_cast<std::size_t>(num_frames) * 2U);

  const int click_length = sample_rate / 100;
  std::vector<float> click(static_cast<std::size_t>(click_length), 0.0F);
  for (int idx = 0; idx < click_length; ++idx) {
    click[static_cast<std::size_t>(idx)] = 1.0F - (static_cast<float>(idx) / click_length);
  }

  for (int frame = 0; frame < num_frames; ++frame) {
    const float time = static_cast<float>(frame) / static_cast<float>(sample_rate);
    float left = 0.30F * std::sin(2.0F * std::numbers::pi_v<float> * 220.0F * time);
    left += 0.15F * std::sin(2.0F * std::numbers::pi_v<float> * 440.0F * time);

    float right = 0.28F * std::sin((2.0F * std::numbers::pi_v<float> * 330.0F * time) + 0.25F);
    right += 0.12F * std::cos(2.0F * std::numbers::pi_v<float> * 550.0F * time);

    const int half_second_index = static_cast<int>(time / 0.5F);
    const int pulse_start = half_second_index * sample_rate / 2;
    const int pulse_offset = frame - pulse_start;
    if (pulse_offset >= 0 && pulse_offset < click_length) {
      const float amplitude = (half_second_index % 4 == 0) ? 0.80F : 0.45F;
      left += amplitude * click[static_cast<std::size_t>(pulse_offset)];

      const int shifted_offset = pulse_offset - (click_length / 4);
      if (shifted_offset >= 0 && shifted_offset < click_length) {
        right += (amplitude * 0.85F) * click[static_cast<std::size_t>(shifted_offset)];
      }
    }

    waveform.push_back(left);
    waveform.push_back(right);
  }
  return waveform;
}

void TestSplitShortSpectrogram() {
  const Spectrogram spectrogram = MakeDeterministicSpectrogram(1000);
  const std::vector<beat_this::inference::internal::SpectrogramChunk> chunks =
      SplitSpectrogram(spectrogram);
  Require(chunks.size() == 1U, "short spectrogram should produce one chunk");
  Require(chunks[0].start_frame == -6, "short spectrogram start offset mismatch");
  Require(chunks[0].num_frames == 1012, "short spectrogram chunk length mismatch");
  Require(chunks[0].num_bins == 128, "short spectrogram chunk bins mismatch");
}

void TestSplitLongSpectrogram() {
  const Spectrogram spectrogram = MakeDeterministicSpectrogram(2000);
  const std::vector<beat_this::inference::internal::SpectrogramChunk> chunks =
      SplitSpectrogram(spectrogram);
  Require(chunks.size() == 2U, "long spectrogram should produce two chunks");
  Require(chunks[0].start_frame == -6, "first long chunk start mismatch");
  Require(chunks[1].start_frame == 506, "second long chunk start mismatch");
  Require(chunks[0].num_frames == 1500, "first long chunk length mismatch");
  Require(chunks[1].num_frames == 1500, "second long chunk length mismatch");
}

void TestAggregateKeepFirstSemantics() {
  const std::vector<ChunkFrameLogits> chunks = {
      ChunkFrameLogits{
          .start_frame = -6,
          .logits =
              FrameLogits{
                  .num_frames = 1500,
                  .beat = std::vector<float>(1500U, 1.0F),
                  .downbeat = std::vector<float>(1500U, 10.0F),
              },
      },
      ChunkFrameLogits{
          .start_frame = 506,
          .logits =
              FrameLogits{
                  .num_frames = 1500,
                  .beat = std::vector<float>(1500U, 2.0F),
                  .downbeat = std::vector<float>(1500U, 20.0F),
              },
      },
  };

  const FrameLogits aggregate = AggregateChunkLogits(chunks, 2000);
  Require(aggregate.num_frames == 2000, "aggregate frame count mismatch");
  for (int frame = 0; frame < 1488; ++frame) {
    Require(aggregate.beat[static_cast<std::size_t>(frame)] == 1.0F,
            "keep_first beat overwrite mismatch");
    Require(aggregate.downbeat[static_cast<std::size_t>(frame)] == 10.0F,
            "keep_first downbeat overwrite mismatch");
  }
  for (int frame = 1488; frame < 2000; ++frame) {
    Require(aggregate.beat[static_cast<std::size_t>(frame)] == 2.0F,
            "tail beat aggregate mismatch");
    Require(aggregate.downbeat[static_cast<std::size_t>(frame)] == 20.0F,
            "tail downbeat aggregate mismatch");
  }
}

void TestMissingModelThrowsRuntime() {
  const std::filesystem::path missing_path =
      std::filesystem::path("re-impl") / "models" / "__missing_model__.onnx";
  RequireThrowsRuntime(
      [&]() {
        static_cast<void>(BeatThisOnnxRunner(missing_path));
      },
      "missing onnx model should throw runtime_error");
}

void TestMinimalPostprocessorPreservesFractionalBeatCenters() {
  MinimalBeatPostprocessor postprocessor;
  FrameLogits logits{
      .num_frames = 32,
      .beat = std::vector<float>(32U, -1.0F),
      .downbeat = std::vector<float>(32U, -1.0F),
  };
  logits.beat[10] = 2.0F;
  logits.beat[11] = 2.0F;
  logits.beat[18] = 0.0F;
  logits.beat[24] = 0.25F;

  const BeatTimestamps timestamps = postprocessor.Process(logits);
  Require(timestamps.downbeats.empty(), "fractional center test should not emit downbeats");
  Require(timestamps.beats.size() == 2U, "fractional center test beat count mismatch");
  RequireNear(timestamps.beats[0], 10.5 / kDefaultFps, 1e-12,
              "adjacent beat peaks should preserve fractional frame center");
  RequireNear(timestamps.beats[1], 24.0 / kDefaultFps, 1e-12,
              "positive isolated beat peak should survive minimal postprocessing");
}

void TestMinimalPostprocessorSnapsAndDeduplicatesDownbeats() {
  MinimalBeatPostprocessor postprocessor;
  FrameLogits logits{
      .num_frames = 24,
      .beat = std::vector<float>(24U, -1.0F),
      .downbeat = std::vector<float>(24U, -1.0F),
  };
  logits.beat[5] = 2.0F;
  logits.beat[15] = 2.0F;

  logits.downbeat[4] = 3.0F;
  logits.downbeat[6] = 3.0F;
  logits.downbeat[16] = 3.0F;

  const BeatTimestamps timestamps = postprocessor.Process(logits);
  const std::vector<double> expected_beats = {5.0 / kDefaultFps, 15.0 / kDefaultFps};
  const std::vector<double> expected_downbeats = {5.0 / kDefaultFps, 15.0 / kDefaultFps};
  RequireTimesEqual(timestamps.beats, expected_beats,
                    "snapping test beat times");
  RequireTimesEqual(timestamps.downbeats, expected_downbeats,
                    "snapping test downbeat times");
}

void TestMinimalPostprocessorRejectsMalformedLogits() {
  const MinimalBeatPostprocessor postprocessor;
  const FrameLogits malformed{
      .num_frames = 4,
      .beat = std::vector<float>(3U, 0.0F),
      .downbeat = std::vector<float>(4U, 0.0F),
  };
  RequireThrowsInvalid(
      [&]() {
        static_cast<void>(postprocessor.Process(malformed));
      },
      "minimal postprocessor should reject malformed frame logits");
}

void TestInferBeatNumbersHandlesPickupMeasure() {
  const BeatTimestamps timestamps{
      .beats = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0},
      .downbeats = {1.5, 3.0},
  };
  const std::vector<int> expected = {2, 3, 1, 2, 3, 1};
  const std::vector<int> actual = InferBeatNumbers(timestamps);
  RequireNumbersEqual(actual, expected, "pickup measure beat number inference");
}

void TestInferBeatNumbersFallsBackWithoutDownbeats() {
  const BeatTimestamps timestamps{
      .beats = {0.5, 1.0, 1.5},
      .downbeats = {},
  };
  const std::vector<int> expected = {2, 3, 4};
  const std::vector<int> actual = InferBeatNumbers(timestamps);
  RequireNumbersEqual(actual, expected, "fallback beat number inference");
}

void TestInferBeatNumbersRejectsMissingBeatDownbeats() {
  const BeatTimestamps timestamps{
      .beats = {0.5, 1.0},
      .downbeats = {0.75},
  };
  RequireThrowsInvalid(
      [&]() {
        static_cast<void>(InferBeatNumbers(timestamps));
      },
      "beat number inference should reject downbeats that are not beats");
}

void TestWriteBeatTsvProducesPythonFormat() {
  const BeatTimestamps timestamps{
      .beats = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0},
      .downbeats = {1.5, 3.0},
  };
  const std::filesystem::path output_path =
      std::filesystem::path("test-artifacts") / "unit-test.beats";

  std::filesystem::remove(output_path);
  WriteBeatTsv(timestamps, output_path);

  const std::vector<std::string> expected = {
      "0.5\t2",
      "1.0\t3",
      "1.5\t1",
      "2.0\t2",
      "2.5\t3",
      "3.0\t1",
  };
  const std::vector<std::string> actual = ReadNonEmptyLines(output_path);
  Require(actual == expected, "beat tsv writer should match python line format");
}

void TestRunnerProcessSpectrogram(const std::filesystem::path& model_path) {
  BeatThisOnnxRunner runner(model_path);
  const Spectrogram spectrogram = MakeDeterministicSpectrogram(1700);

  const FrameLogits first = runner.ProcessSpectrogram(spectrogram);
  const FrameLogits second = runner.ProcessSpectrogram(spectrogram);

  Require(first.num_frames == 1700, "runner should preserve spectrogram frame count");
  Require(first.beat.size() == 1700U, "runner beat logits size mismatch");
  Require(first.downbeat.size() == 1700U, "runner downbeat logits size mismatch");
  Require(first.beat == second.beat, "runner should be deterministic for beat logits");
  Require(first.downbeat == second.downbeat, "runner should be deterministic for downbeat logits");
  RequireAllFinite(first, "runner spectrogram logits");

  const BeatTimestamps manual = MinimalBeatPostprocessor().Process(first);
  const BeatTimestamps convenience = runner.ProcessSpectrogramToBeats(spectrogram);
  RequireTimestampsEqual(convenience, manual, "runner spectrogram convenience beats");
}

void TestRunnerRejectsInvalidSpectrogram(const std::filesystem::path& model_path) {
  BeatThisOnnxRunner runner(model_path);
  const Spectrogram bad{
      .num_frames = 10,
      .num_bins = 128,
      .values = std::vector<float>(100U, 0.0F),
  };
  RequireThrowsInvalid(
      [&]() {
        static_cast<void>(runner.ProcessSpectrogram(bad));
      },
      "runner should reject malformed spectrogram storage");
}

void TestRunnerProcessWaveform(const std::filesystem::path& model_path) {
  BeatThisOnnxRunner runner(model_path);
  const std::vector<float> waveform = MakeStereoWaveform(44100, 32.0F);
  const AudioBufferView audio{
      .samples = std::span<const float>(waveform),
      .sample_rate = 44100,
      .num_channels = 2,
      .layout = ChannelLayout::kInterleavedFrames,
  };
  const FrameLogits logits = runner.ProcessWaveform(audio);

  Require(logits.num_frames > 1500, "waveform integration path should exercise chunk splitting");
  RequireAllFinite(logits, "runner waveform logits");

  const BeatTimestamps manual = MinimalBeatPostprocessor().Process(logits);
  const BeatTimestamps convenience = runner.ProcessWaveformToBeats(audio);
  Require(!convenience.beats.empty(), "runner waveform convenience beats should not be empty");
  RequireTimestampsEqual(convenience, manual, "runner waveform convenience beats");
}

void TestRunnerProcessFile(const std::filesystem::path& model_path,
                           const std::filesystem::path& audio_path) {
  BeatThisOnnxRunner runner(model_path);
  const FrameLogits logits = runner.ProcessFile(audio_path);
  Require(logits.num_frames > 0, "file integration path should produce logits");
  RequireAllFinite(logits, "runner file logits");

  const BeatTimestamps manual = MinimalBeatPostprocessor().Process(logits);
  const BeatTimestamps convenience = runner.ProcessFileToBeats(audio_path);
  Require(!convenience.beats.empty(), "runner file convenience beats should not be empty");
  RequireTimestampsEqual(convenience, manual, "runner file convenience beats");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    TestSplitShortSpectrogram();
    TestSplitLongSpectrogram();
    TestAggregateKeepFirstSemantics();
    TestMissingModelThrowsRuntime();
    TestMinimalPostprocessorPreservesFractionalBeatCenters();
    TestMinimalPostprocessorSnapsAndDeduplicatesDownbeats();
    TestMinimalPostprocessorRejectsMalformedLogits();
    TestInferBeatNumbersHandlesPickupMeasure();
    TestInferBeatNumbersFallsBackWithoutDownbeats();
    TestInferBeatNumbersRejectsMissingBeatDownbeats();
    TestWriteBeatTsvProducesPythonFormat();

    if (argc >= 2) {
      TestRunnerProcessSpectrogram(argv[1]);
      TestRunnerRejectsInvalidSpectrogram(argv[1]);
      TestRunnerProcessWaveform(argv[1]);
    } else {
      std::cout << "skipping model-dependent inference tests\n";
    }

    if (argc >= 3) {
      TestRunnerProcessFile(argv[1], argv[2]);
    }

    std::cout << "all inference tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
