#include "internal/soxr_resampler.hpp"

#include "internal/common.hpp"

#include "soxr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace beat_this::preprocess::internal {

namespace {

constexpr std::size_t kResampleBufferFloorFrames = 1024U;
constexpr std::size_t kFlushChunkFrames = 1024U;

std::size_t CheckedAdd(const std::size_t lhs,
                       const std::size_t rhs,
                       const std::string_view context) {
  if (lhs > (std::numeric_limits<std::size_t>::max() - rhs)) {
    ThrowRuntime(std::string(context) + " overflowed");
  }
  return lhs + rhs;
}

std::size_t CheckedFrameCountFromDouble(const double value,
                                        const std::string_view context) {
  if (!std::isfinite(value) || value <= 0.0) {
    ThrowRuntime(std::string(context) + " must be finite and positive");
  }

  constexpr double kMaxSizeT = static_cast<double>(std::numeric_limits<std::size_t>::max());
  if (value > kMaxSizeT) {
    ThrowRuntime(std::string(context) + " exceeds supported frame count");
  }

  return static_cast<std::size_t>(std::ceil(value));
}

std::size_t NextPowerOfTwo(std::size_t value) {
  std::size_t result = kResampleBufferFloorFrames;
  while (result < value) {
    if (result > (std::numeric_limits<std::size_t>::max() >> 1U)) {
      ThrowRuntime("resampler output capacity overflowed");
    }
    result <<= 1U;
  }
  return result;
}

void EnsureOutputCapacity(std::vector<double>& output, const std::size_t required_frames) {
  if (required_frames <= output.size()) {
    return;
  }
  if (required_frames > output.max_size()) {
    ThrowRuntime("resampler output exceeds vector capacity");
  }
  output.resize(NextPowerOfTwo(required_frames), 0.0);
}

using SoxrHandle = std::unique_ptr<soxr, decltype(&soxr_delete)>;

SoxrHandle CreateSoxrResampler(const int input_sample_rate, const int output_sample_rate) {
  const soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);
  const soxr_quality_spec_t quality_spec = soxr_quality_spec(SOXR_HQ, 0);
  soxr_error_t error = nullptr;
  soxr_t raw = soxr_create(static_cast<double>(input_sample_rate),
                           static_cast<double>(output_sample_rate),
                           1,
                           &error,
                           &io_spec,
                           &quality_spec,
                           nullptr);
  if (error != nullptr || raw == nullptr) {
    ThrowRuntime(std::string("libsoxr streaming create failed: ") +
                 (error != nullptr ? error : "unknown error"));
  }
  return SoxrHandle(raw, &soxr_delete);
}

}  // namespace

std::vector<double> ResampleMonoWaveformSoxr(const std::span<const double> mono,
                                             const int input_sample_rate,
                                             const int output_sample_rate) {
  if (input_sample_rate <= 0 || output_sample_rate <= 0) {
    ThrowInvalid("sample rates must be positive");
  }
  if (mono.empty()) {
    ThrowInvalid("mono waveform must be non-empty");
  }
  if (input_sample_rate == output_sample_rate) {
    return std::vector<double>(mono.begin(), mono.end());
  }

  const double ratio =
      static_cast<double>(output_sample_rate) / static_cast<double>(input_sample_rate);
  const std::size_t div_len = static_cast<std::size_t>(
      std::max(1000.0,
               48000.0 * static_cast<double>(input_sample_rate) /
                   static_cast<double>(output_sample_rate)));
  SoxrHandle resampler = CreateSoxrResampler(input_sample_rate, output_sample_rate);
  const std::size_t initial_capacity = CheckedFrameCountFromDouble(
      soxr_delay(resampler.get()) + (static_cast<double>(mono.size()) * ratio) + 1.0,
      "initial resampler output capacity");

  std::vector<double> output;
  EnsureOutputCapacity(output, std::max(initial_capacity, kResampleBufferFloorFrames));

  std::size_t out_pos = 0;
  for (std::size_t idx = 0; idx < mono.size(); idx += div_len) {
    const std::size_t chunk_len = std::min(div_len, mono.size() - idx);
    std::size_t chunk_consumed = 0;

    while (chunk_consumed < chunk_len) {
      if (output.size() <= out_pos) {
        EnsureOutputCapacity(output, CheckedAdd(out_pos, kFlushChunkFrames, "resampler output position"));
      }

      std::size_t input_done = 0;
      std::size_t output_done = 0;
      const soxr_error_t error = soxr_process(resampler.get(),
                                              mono.data() + idx + chunk_consumed,
                                              chunk_len - chunk_consumed,
                                              &input_done,
                                              output.data() + out_pos,
                                              output.size() - out_pos,
                                              &output_done);
      if (error != nullptr) {
        ThrowRuntime(std::string("libsoxr streaming process failed: ") + error);
      }

      chunk_consumed += input_done;
      out_pos += output_done;

      if (chunk_consumed == chunk_len) {
        break;
      }
      if (input_done == 0 && output_done == 0) {
        ThrowRuntime("libsoxr streaming process stalled before consuming the full input chunk");
      }
      EnsureOutputCapacity(output,
                           CheckedAdd(output.size(), kFlushChunkFrames, "resampler output growth"));
    }
  }

  while (true) {
    if (output.size() <= out_pos) {
      EnsureOutputCapacity(output, CheckedAdd(out_pos, kFlushChunkFrames, "resampler flush output"));
    }

    std::size_t output_done = 0;
    const soxr_error_t error = soxr_process(
        resampler.get(), nullptr, 0, nullptr, output.data() + out_pos, output.size() - out_pos, &output_done);
    if (error != nullptr) {
      ThrowRuntime(std::string("libsoxr flush failed: ") + error);
    }

    out_pos += output_done;
    if (output_done == 0) {
      break;
    }
  }

  if (out_pos == 0) {
    ThrowRuntime("resampler produced an empty waveform");
  }

  output.resize(out_pos);
  return output;
}

}  // namespace beat_this::preprocess::internal
