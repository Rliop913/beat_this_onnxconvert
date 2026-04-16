#pragma once

#include "beat_this_inference/inference.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace Ort {
class Session;
}

namespace beat_this::inference::internal {

class OnnxSession {
 public:
  explicit OnnxSession(std::filesystem::path model_path);
  ~OnnxSession();

  OnnxSession(OnnxSession&&) noexcept;
  OnnxSession& operator=(OnnxSession&&) noexcept;

  OnnxSession(const OnnxSession&) = delete;
  OnnxSession& operator=(const OnnxSession&) = delete;

  [[nodiscard]] FrameLogits RunSpectrogramChunk(
      std::span<const float> values,
      int num_frames,
      int num_bins) const;

 private:
  std::filesystem::path model_path_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
};

}  // namespace beat_this::inference::internal

