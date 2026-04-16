#include "internal/aggregate.hpp"

#include "internal/common.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

namespace beat_this::inference::internal {

namespace {

using beat_this::preprocess::internal::ThrowInvalid;

void ValidateChunkLogits(const ChunkFrameLogits& chunk) {
  if (chunk.logits.num_frames <= 0) {
    ThrowInvalid("chunk logits must contain at least one frame");
  }
  if (chunk.logits.beat.size() != chunk.logits.downbeat.size()) {
    ThrowInvalid("beat/downbeat chunk logits must have matching sizes");
  }
  if (chunk.logits.beat.size() != static_cast<std::size_t>(chunk.logits.num_frames)) {
    ThrowInvalid("chunk logits flattened size does not match its frame count");
  }
}

}  // namespace

FrameLogits AggregateChunkLogits(const std::span<const ChunkFrameLogits> chunks,
                                 const int full_num_frames) {
  if (full_num_frames <= 0) {
    ThrowInvalid("aggregate target must contain at least one frame");
  }
  if (chunks.empty()) {
    ThrowInvalid("aggregate requires at least one chunk");
  }

  FrameLogits result{
      .num_frames = full_num_frames,
      .beat = std::vector<float>(static_cast<std::size_t>(full_num_frames),
                                 beat_this::preprocess::internal::kPaddingLogit),
      .downbeat = std::vector<float>(static_cast<std::size_t>(full_num_frames),
                                     beat_this::preprocess::internal::kPaddingLogit),
  };

  for (auto chunk_it = chunks.rbegin(); chunk_it != chunks.rend(); ++chunk_it) {
    ValidateChunkLogits(*chunk_it);

    const int target_begin =
        std::max(0, chunk_it->start_frame + beat_this::preprocess::internal::kChunkBorderSize);
    const int target_end = std::min(
        full_num_frames,
        chunk_it->start_frame + beat_this::preprocess::internal::kChunkSize -
            beat_this::preprocess::internal::kChunkBorderSize);
    if (target_end <= target_begin) {
      continue;
    }

    const std::size_t copy_count = static_cast<std::size_t>(target_end - target_begin);
    const std::size_t source_offset =
        static_cast<std::size_t>(beat_this::preprocess::internal::kChunkBorderSize);
    if ((source_offset + copy_count) > chunk_it->logits.beat.size()) {
      ThrowInvalid("chunk logits are shorter than the aggregate target window");
    }

    std::copy_n(chunk_it->logits.beat.begin() + static_cast<std::ptrdiff_t>(source_offset),
                static_cast<std::ptrdiff_t>(copy_count),
                result.beat.begin() + static_cast<std::ptrdiff_t>(target_begin));
    std::copy_n(
        chunk_it->logits.downbeat.begin() + static_cast<std::ptrdiff_t>(source_offset),
        static_cast<std::ptrdiff_t>(copy_count),
        result.downbeat.begin() + static_cast<std::ptrdiff_t>(target_begin));
  }

  return result;
}

}  // namespace beat_this::inference::internal
