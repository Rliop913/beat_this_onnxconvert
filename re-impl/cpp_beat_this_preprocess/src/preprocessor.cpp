#include "beat_this_preprocess/preprocessor.hpp"

#include "internal/common.hpp"
#include "internal/decode.hpp"
#include "internal/frontend.hpp"
#include "internal/waveform.hpp"

namespace beat_this::preprocess {

Spectrogram BeatThisPreprocessor::ProcessWaveform(const AudioBufferView audio) const {
  const std::vector<float> mono = internal::PrepareMonoWaveform(audio);
  return internal::ComputeLogMelSpectrogram(mono);
}

Spectrogram BeatThisPreprocessor::ProcessFile(const std::filesystem::path& path) const {
  const internal::DecodedAudio decoded = internal::DecodeFileToInterleavedF32(path);
  const std::vector<float> mono = internal::PrepareMonoWaveform(
      AudioBufferView{
          .samples = std::span<const float>(decoded.interleaved_samples),
           .sample_rate = decoded.sample_rate,
           .num_channels = decoded.num_channels,
           .layout = ChannelLayout::kInterleavedFrames,
       });
  return internal::ComputeLogMelSpectrogram(mono);
}

}  // namespace beat_this::preprocess
