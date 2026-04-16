#include "beat_this_inference/inference.hpp"

#include "internal/aggregate.hpp"
#include "internal/common.hpp"
#include "internal/onnx_session.hpp"
#include "internal/split.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace beat_this::inference {

class BeatThisOnnxRunner::Impl {
 public:
  explicit Impl(std::filesystem::path model_path) : session(std::move(model_path)) {}

  beat_this::preprocess::BeatThisPreprocessor preprocessor;
  internal::OnnxSession session;
};

namespace {

using beat_this::preprocess::Spectrogram;
using beat_this::preprocess::internal::ThrowInvalid;

void ValidateSpectrogram(const Spectrogram& spectrogram) {
  if (spectrogram.num_frames <= 0) {
    ThrowInvalid("spectrogram must contain at least one frame");
  }
  if (spectrogram.num_bins != beat_this::preprocess::internal::kNumMels) {
    ThrowInvalid("spectrogram must contain 128 mel bins");
  }
  const std::size_t expected_values = static_cast<std::size_t>(spectrogram.num_frames) *
                                      static_cast<std::size_t>(spectrogram.num_bins);
  if (spectrogram.values.size() != expected_values) {
    ThrowInvalid("spectrogram flattened storage size does not match its shape");
  }
}

}  // namespace

BeatThisOnnxRunner::BeatThisOnnxRunner(std::filesystem::path model_path)
    : impl_(std::make_unique<Impl>(std::move(model_path))) {}

BeatThisOnnxRunner::~BeatThisOnnxRunner() = default;
BeatThisOnnxRunner::BeatThisOnnxRunner(BeatThisOnnxRunner&&) noexcept = default;
BeatThisOnnxRunner& BeatThisOnnxRunner::operator=(BeatThisOnnxRunner&&) noexcept = default;

FrameLogits BeatThisOnnxRunner::ProcessSpectrogram(
    const beat_this::preprocess::Spectrogram& spectrogram) const {
  ValidateSpectrogram(spectrogram);

  const std::vector<internal::SpectrogramChunk> chunks = internal::SplitSpectrogram(spectrogram);
  std::vector<internal::ChunkFrameLogits> chunk_logits;
  chunk_logits.reserve(chunks.size());

  for (const internal::SpectrogramChunk& chunk : chunks) {
    chunk_logits.push_back(internal::ChunkFrameLogits{
        .start_frame = chunk.start_frame,
        .logits = impl_->session.RunSpectrogramChunk(chunk.values, chunk.num_frames, chunk.num_bins),
    });
  }

  return internal::AggregateChunkLogits(chunk_logits, spectrogram.num_frames);
}

FrameLogits BeatThisOnnxRunner::ProcessWaveform(
    const beat_this::preprocess::AudioBufferView audio) const {
  const beat_this::preprocess::Spectrogram spectrogram = impl_->preprocessor.ProcessWaveform(audio);
  return ProcessSpectrogram(spectrogram);
}

FrameLogits BeatThisOnnxRunner::ProcessFile(const std::filesystem::path& path) const {
  const beat_this::preprocess::Spectrogram spectrogram = impl_->preprocessor.ProcessFile(path);
  return ProcessSpectrogram(spectrogram);
}

}  // namespace beat_this::inference
