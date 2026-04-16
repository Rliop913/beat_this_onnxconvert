#include "internal/split.hpp"

#include "internal/common.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

namespace beat_this::inference::internal {

namespace {

using beat_this::preprocess::Spectrogram;
using beat_this::preprocess::internal::ThrowInvalid;

std::vector<int> ComputeChunkStarts(const int num_frames) {
  const int step = beat_this::preprocess::internal::kChunkSize -
                   (2 * beat_this::preprocess::internal::kChunkBorderSize);
  std::vector<int> starts;
  for (int start = -beat_this::preprocess::internal::kChunkBorderSize;
       start < num_frames - beat_this::preprocess::internal::kChunkBorderSize;
       start += step) {
    starts.push_back(start);
  }
  if (num_frames > step && !starts.empty()) {
    starts.back() =
        num_frames - (beat_this::preprocess::internal::kChunkSize -
                      beat_this::preprocess::internal::kChunkBorderSize);
  }
  return starts;
}

}  // namespace

std::vector<SpectrogramChunk> SplitSpectrogram(const Spectrogram& spectrogram) {
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

  const std::vector<int> starts = ComputeChunkStarts(spectrogram.num_frames);
  std::vector<SpectrogramChunk> chunks;
  chunks.reserve(starts.size());

  for (const int start : starts) {
    const int content_begin = std::max(start, 0);
    const int content_end = std::min(
        start + beat_this::preprocess::internal::kChunkSize, spectrogram.num_frames);
    const int left_pad = std::max(0, -start);
    const int right_pad = std::max(
        0,
        std::min(beat_this::preprocess::internal::kChunkBorderSize,
                 start + beat_this::preprocess::internal::kChunkSize -
                     spectrogram.num_frames));
    const int chunk_num_frames = left_pad + (content_end - content_begin) + right_pad;
    if (chunk_num_frames <= 0) {
      ThrowInvalid("split_piece produced an empty chunk");
    }

    SpectrogramChunk chunk{
        .start_frame = start,
        .num_frames = chunk_num_frames,
        .num_bins = spectrogram.num_bins,
        .values = std::vector<float>(
            static_cast<std::size_t>(chunk_num_frames) *
                static_cast<std::size_t>(spectrogram.num_bins),
            0.0F),
    };

    if (content_end > content_begin) {
      const std::size_t source_offset =
          static_cast<std::size_t>(content_begin) *
          static_cast<std::size_t>(spectrogram.num_bins);
      const std::size_t copy_count =
          static_cast<std::size_t>(content_end - content_begin) *
          static_cast<std::size_t>(spectrogram.num_bins);
      const std::size_t destination_offset =
          static_cast<std::size_t>(left_pad) *
          static_cast<std::size_t>(spectrogram.num_bins);
      std::copy_n(spectrogram.values.begin() +
                      static_cast<std::ptrdiff_t>(source_offset),
                  static_cast<std::ptrdiff_t>(copy_count),
                  chunk.values.begin() +
                      static_cast<std::ptrdiff_t>(destination_offset));
    }

    chunks.push_back(std::move(chunk));
  }

  return chunks;
}

}  // namespace beat_this::inference::internal

