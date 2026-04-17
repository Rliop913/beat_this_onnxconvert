#pragma once

#include "beat_this_preprocess/preprocessor.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace beat_this::inference {

struct FrameLogits {
  int num_frames = 0;
  std::vector<float> beat;
  std::vector<float> downbeat;

  [[nodiscard]] bool empty() const noexcept { return num_frames == 0; }
};

struct BeatTimestamps {
  std::vector<double> beats;
  std::vector<double> downbeats;

  [[nodiscard]] bool empty() const noexcept { return beats.empty() && downbeats.empty(); }
};

class MinimalBeatPostprocessor {
 public:
  explicit MinimalBeatPostprocessor(double fps = 50.0);

  [[nodiscard]] BeatTimestamps Process(const FrameLogits& logits) const;
  [[nodiscard]] double fps() const noexcept { return fps_; }

 private:
  double fps_ = 50.0;
};

[[nodiscard]] std::vector<int> InferBeatNumbers(const BeatTimestamps& timestamps);
void WriteBeatTsv(const BeatTimestamps& timestamps, const std::filesystem::path& output_path);

class BeatThisOnnxRunner {
 public:
  explicit BeatThisOnnxRunner(std::filesystem::path model_path);
  ~BeatThisOnnxRunner();

  BeatThisOnnxRunner(BeatThisOnnxRunner&&) noexcept;
  BeatThisOnnxRunner& operator=(BeatThisOnnxRunner&&) noexcept;

  BeatThisOnnxRunner(const BeatThisOnnxRunner&) = delete;
  BeatThisOnnxRunner& operator=(const BeatThisOnnxRunner&) = delete;

  [[nodiscard]] FrameLogits ProcessSpectrogram(
      const beat_this::preprocess::Spectrogram& spectrogram) const;
  [[nodiscard]] FrameLogits ProcessWaveform(
      beat_this::preprocess::AudioBufferView audio) const;
  [[nodiscard]] FrameLogits ProcessFile(const std::filesystem::path& path) const;
  [[nodiscard]] BeatTimestamps ProcessSpectrogramToBeats(
      const beat_this::preprocess::Spectrogram& spectrogram) const;
  [[nodiscard]] BeatTimestamps ProcessWaveformToBeats(
      beat_this::preprocess::AudioBufferView audio) const;
  [[nodiscard]] BeatTimestamps ProcessFileToBeats(const std::filesystem::path& path) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace beat_this::inference
