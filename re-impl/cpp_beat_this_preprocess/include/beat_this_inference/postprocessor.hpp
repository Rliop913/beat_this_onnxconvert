#pragma once

#include "beat_this_inference/inference.hpp"

#include <vector>

namespace beat_this::inference {

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

}  // namespace beat_this::inference
