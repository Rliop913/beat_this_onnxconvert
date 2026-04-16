#include "internal/decode.hpp"

#include "internal/common.hpp"

#include "miniaudio.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace beat_this::preprocess::internal {

namespace {

class DecoderHandle {
 public:
  DecoderHandle() = default;

  DecoderHandle(const DecoderHandle&) = delete;
  DecoderHandle& operator=(const DecoderHandle&) = delete;

  ~DecoderHandle() {
    if (initialized_) {
      ma_decoder_uninit(&decoder_);
    }
  }

  [[nodiscard]] ma_decoder* get() noexcept { return &decoder_; }
  void MarkInitialized() noexcept { initialized_ = true; }

 private:
  ma_decoder decoder_{};
  bool initialized_ = false;
};

}  // namespace

DecodedAudio DecodeFileToInterleavedF32(const std::filesystem::path& path) {
  DecoderHandle decoder;
  ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);

#if defined(_WIN32)
  const std::wstring wide_path = path.wstring();
  const ma_result init_result = ma_decoder_init_file_w(wide_path.c_str(), &config, decoder.get());
#else
  const std::string narrow_path = path.string();
  const ma_result init_result = ma_decoder_init_file(narrow_path.c_str(), &config, decoder.get());
#endif

  if (init_result != MA_SUCCESS) {
    ThrowRuntime("failed to decode audio file with miniaudio");
  }
  decoder.MarkInitialized();

  ma_format format = ma_format_unknown;
  ma_uint32 channels = 0;
  ma_uint32 sample_rate = 0;
  const ma_result format_result =
      ma_decoder_get_data_format(decoder.get(), &format, &channels, &sample_rate, nullptr, 0);
  if (format_result != MA_SUCCESS || format != ma_format_f32 || channels == 0 || sample_rate == 0) {
    ThrowRuntime("failed to query decoded audio format");
  }

  constexpr ma_uint64 kChunkFrames = 4096;
  std::vector<float> chunk(static_cast<std::size_t>(kChunkFrames * channels), 0.0F);
  std::vector<float> output;
  ma_uint64 total_frames = 0;
  if (ma_decoder_get_length_in_pcm_frames(decoder.get(), &total_frames) == MA_SUCCESS && total_frames > 0) {
    const ma_uint64 max_samples =
        static_cast<ma_uint64>(std::numeric_limits<std::size_t>::max() / channels);
    if (total_frames > max_samples) {
      ThrowRuntime("decoded audio file exceeds supported size");
    }
    output.reserve(static_cast<std::size_t>(total_frames * channels));
  }

  while (true) {
    ma_uint64 frames_read = 0;
    const ma_result read_result =
        ma_decoder_read_pcm_frames(decoder.get(), chunk.data(), kChunkFrames, &frames_read);
    if (read_result != MA_SUCCESS) {
      ThrowRuntime("failed while decoding PCM frames");
    }
    if (frames_read == 0) {
      break;
    }

    const std::size_t sample_count = static_cast<std::size_t>(frames_read * channels);
    output.insert(output.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(sample_count));
    if (frames_read < kChunkFrames) {
      break;
    }
  }

  if (output.empty()) {
    ThrowRuntime("decoded audio file is empty");
  }

  return {
      .sample_rate = static_cast<int>(sample_rate),
      .num_channels = static_cast<int>(channels),
      .interleaved_samples = std::move(output),
  };
}

}  // namespace beat_this::preprocess::internal
