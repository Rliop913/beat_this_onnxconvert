#pragma once

#include "beat_this_inference/inference.hpp"

#include <span>

namespace beat_this::inference::internal {

struct ChunkFrameLogits {
  int start_frame = 0;
  FrameLogits logits;
};

[[nodiscard]] FrameLogits AggregateChunkLogits(
    std::span<const ChunkFrameLogits> chunks,
    int full_num_frames);

}  // namespace beat_this::inference::internal

