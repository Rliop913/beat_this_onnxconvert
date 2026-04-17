#include "beat_this_inference/inference.hpp"
#include "beat_this_preprocess/preprocessor.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using beat_this::inference::BeatThisOnnxRunner;
using beat_this::inference::BeatTimestamps;
using beat_this::inference::WriteBeatTsv;
using beat_this::preprocess::AudioBufferView;
using beat_this::preprocess::ChannelLayout;

struct Options {
  std::filesystem::path model_path;
  std::filesystem::path output_path;
  std::filesystem::path audio_path;
  std::filesystem::path waveform_path;
  int sample_rate = 0;
  int num_channels = 0;
};

[[noreturn]] void ThrowUsage(const std::string_view message) {
  throw std::invalid_argument(std::string(message));
}

std::string_view NextValue(int& index, const int argc, char** argv) {
  if (index + 1 >= argc) {
    ThrowUsage("missing value for command line option");
  }
  ++index;
  return argv[index];
}

Options ParseOptions(const int argc, char** argv) {
  Options options;

  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--model") {
      options.model_path = std::filesystem::path(NextValue(index, argc, argv));
    } else if (arg == "--output") {
      options.output_path = std::filesystem::path(NextValue(index, argc, argv));
    } else if (arg == "--audio") {
      options.audio_path = std::filesystem::path(NextValue(index, argc, argv));
    } else if (arg == "--waveform-bin") {
      options.waveform_path = std::filesystem::path(NextValue(index, argc, argv));
    } else if (arg == "--sample-rate") {
      options.sample_rate = std::stoi(std::string(NextValue(index, argc, argv)));
    } else if (arg == "--num-channels") {
      options.num_channels = std::stoi(std::string(NextValue(index, argc, argv)));
    } else {
      ThrowUsage("unknown argument");
    }
  }

  if (options.model_path.empty()) {
    ThrowUsage("--model is required");
  }
  if (options.output_path.empty()) {
    ThrowUsage("--output is required");
  }

  const bool using_audio_path = !options.audio_path.empty();
  const bool using_waveform_path = !options.waveform_path.empty();
  if (using_audio_path == using_waveform_path) {
    ThrowUsage("exactly one of --audio or --waveform-bin must be provided");
  }

  if (using_waveform_path) {
    if (options.sample_rate <= 0) {
      ThrowUsage("--sample-rate must be positive when using --waveform-bin");
    }
    if (options.num_channels <= 0) {
      ThrowUsage("--num-channels must be positive when using --waveform-bin");
    }
  }

  return options;
}

std::vector<float> ReadWaveformBinary(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("could not open waveform binary: " + path.string());
  }

  const std::streamsize byte_size = input.tellg();
  if (byte_size < 0 || (byte_size % static_cast<std::streamsize>(sizeof(float))) != 0) {
    throw std::runtime_error("waveform binary size is not a multiple of float32");
  }

  input.seekg(0, std::ios::beg);
  std::vector<float> values(static_cast<std::size_t>(byte_size / sizeof(float)), 0.0F);
  if (!input.read(reinterpret_cast<char*>(values.data()), byte_size)) {
    throw std::runtime_error("failed to read waveform binary: " + path.string());
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    BeatThisOnnxRunner runner(options.model_path);

    BeatTimestamps timestamps;
    if (!options.audio_path.empty()) {
      timestamps = runner.ProcessFileToBeats(options.audio_path);
    } else {
      const std::vector<float> waveform = ReadWaveformBinary(options.waveform_path);
      timestamps = runner.ProcessWaveformToBeats(AudioBufferView{
          .samples = std::span<const float>(waveform),
          .sample_rate = options.sample_rate,
          .num_channels = options.num_channels,
          .layout = ChannelLayout::kInterleavedFrames,
      });
    }

    WriteBeatTsv(timestamps, options.output_path);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
