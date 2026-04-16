#include "internal/onnx_session.hpp"

#include "internal/common.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace beat_this::inference::internal {

namespace {

using beat_this::preprocess::internal::ThrowInvalid;
using beat_this::preprocess::internal::ThrowRuntime;

constexpr const char* kExpectedInputName = "spect";
constexpr const char* kExpectedOutputName = "logits";

Ort::Env& GetOrtEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "beat_this_cpp");
  return env;
}

std::string ReadIoName(Ort::Session& session,
                       const bool input,
                       const std::size_t index,
                       Ort::AllocatorWithDefaultOptions& allocator) {
  Ort::AllocatedStringPtr name = input ? session.GetInputNameAllocated(index, allocator)
                                       : session.GetOutputNameAllocated(index, allocator);
  if (!name || name.get()[0] == '\0') {
    ThrowRuntime(input ? "onnx model input name is empty" : "onnx model output name is empty");
  }
  return std::string(name.get());
}

void RequireIoName(const std::string& actual, const char* expected, const char* kind) {
  if (actual != expected) {
    ThrowRuntime(std::string("unexpected onnx model ") + kind + " name: expected '" + expected +
                 "', got '" + actual + "'");
  }
}

}  // namespace

OnnxSession::OnnxSession(std::filesystem::path model_path) : model_path_(std::move(model_path)) {
  if (!std::filesystem::exists(model_path_)) {
    ThrowRuntime(std::string("onnx model was not found: ") + model_path_.string());
  }

  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);

  session_ = std::make_unique<Ort::Session>(GetOrtEnv(), model_path_.c_str(), options);

  Ort::AllocatorWithDefaultOptions allocator;
  if (session_->GetInputCount() != 1U) {
    ThrowRuntime("onnx model must expose exactly one input");
  }
  if (session_->GetOutputCount() != 1U) {
    ThrowRuntime("onnx model must expose exactly one output");
  }

  input_name_ = ReadIoName(*session_, true, 0U, allocator);
  output_name_ = ReadIoName(*session_, false, 0U, allocator);
  RequireIoName(input_name_, kExpectedInputName, "input");
  RequireIoName(output_name_, kExpectedOutputName, "output");
}

OnnxSession::~OnnxSession() = default;
OnnxSession::OnnxSession(OnnxSession&&) noexcept = default;
OnnxSession& OnnxSession::operator=(OnnxSession&&) noexcept = default;

FrameLogits OnnxSession::RunSpectrogramChunk(const std::span<const float> values,
                                             const int num_frames,
                                             const int num_bins) const {
  if (num_frames <= 0) {
    ThrowInvalid("onnx session input must contain at least one frame");
  }
  if (num_bins != beat_this::preprocess::internal::kNumMels) {
    ThrowInvalid("onnx session expects 128 mel bins");
  }
  const std::size_t expected_values =
      static_cast<std::size_t>(num_frames) * static_cast<std::size_t>(num_bins);
  if (values.size() != expected_values) {
    ThrowInvalid("onnx session input flattened size does not match its shape");
  }

  static const Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 3> input_shape = {1, static_cast<int64_t>(num_frames),
                                              static_cast<int64_t>(num_bins)};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      const_cast<float*>(values.data()),
      values.size(),
      input_shape.data(),
      input_shape.size());

  const std::array<const char*, 1> input_names = {input_name_.c_str()};
  const std::array<const char*, 1> output_names = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr},
                               input_names.data(),
                               &input_tensor,
                               input_names.size(),
                               output_names.data(),
                               output_names.size());
  if (outputs.size() != 1U || !outputs[0].IsTensor()) {
    ThrowRuntime("onnx runtime did not return the expected tensor output");
  }

  const Ort::TensorTypeAndShapeInfo output_info = outputs[0].GetTensorTypeAndShapeInfo();
  const std::vector<int64_t> output_shape = output_info.GetShape();
  if (output_shape.size() != 3U || output_shape[0] != 1 ||
      output_shape[1] != static_cast<int64_t>(num_frames) ||
      output_shape[2] != static_cast<int64_t>(beat_this::preprocess::internal::kModelOutputChannels)) {
    ThrowRuntime("onnx model returned an unexpected logits shape");
  }

  const std::size_t expected_output_values =
      static_cast<std::size_t>(num_frames) *
      static_cast<std::size_t>(beat_this::preprocess::internal::kModelOutputChannels);
  if (output_info.GetElementCount() != expected_output_values) {
    ThrowRuntime("onnx model returned an unexpected logits element count");
  }

  const float* output_data = outputs[0].GetTensorData<float>();
  FrameLogits logits{
      .num_frames = num_frames,
      .beat = std::vector<float>(static_cast<std::size_t>(num_frames), 0.0F),
      .downbeat = std::vector<float>(static_cast<std::size_t>(num_frames), 0.0F),
  };

  for (int frame = 0; frame < num_frames; ++frame) {
    const std::size_t offset = static_cast<std::size_t>(frame) *
                               static_cast<std::size_t>(
                                   beat_this::preprocess::internal::kModelOutputChannels);
    logits.beat[static_cast<std::size_t>(frame)] = output_data[offset];
    logits.downbeat[static_cast<std::size_t>(frame)] = output_data[offset + 1U];
  }

  return logits;
}

}  // namespace beat_this::inference::internal
