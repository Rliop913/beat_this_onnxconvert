#pragma once

#include <cstddef>
#include <string_view>

namespace beat_this::preprocess::internal {

inline constexpr int kTargetSampleRate = 22050;
inline constexpr int kNfft = 1024;
inline constexpr int kHopLength = 441;
inline constexpr int kNumMels = 128;
inline constexpr int kChunkSize = 1500;
inline constexpr int kChunkBorderSize = 6;
inline constexpr int kModelOutputChannels = 2;
inline constexpr float kFMinHz = 30.0F;
inline constexpr float kFMaxHz = 11000.0F;
inline constexpr float kLogMultiplier = 1000.0F;
inline constexpr float kPaddingLogit = -1000.0F;
inline constexpr bool kCenterFrames = true;
inline constexpr int kPad = 512;
inline constexpr std::size_t kFftBins = static_cast<std::size_t>((kNfft / 2) + 1);

[[noreturn]] void ThrowInvalid(std::string_view message);
[[noreturn]] void ThrowRuntime(std::string_view message);

}  // namespace beat_this::preprocess::internal
