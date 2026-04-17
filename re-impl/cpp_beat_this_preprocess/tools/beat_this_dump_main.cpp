#include "beat_this_inference/export.hpp"
#include "beat_this_inference/inference.hpp"
#include "beat_this_inference/postprocessor.hpp"
#include "beat_this_preprocess/preprocessor.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using beat_this::inference::BeatThisOnnxRunner;
using beat_this::inference::BeatTimestamps;
using beat_this::inference::FrameLogits;
using beat_this::inference::MinimalBeatPostprocessor;
using beat_this::inference::WriteBeatTsv;
using beat_this::preprocess::AudioBufferView;
using beat_this::preprocess::BeatThisPreprocessor;
using beat_this::preprocess::ChannelLayout;
using beat_this::preprocess::Spectrogram;

enum class OutputFormat {
  kSpectrogram,
  kLogits,
  kBeatsJson,
  kBeatsTsv,
};

struct Options {
  std::filesystem::path model_path;
  std::filesystem::path output_path;
  std::filesystem::path audio_path;
  std::filesystem::path waveform_path;
  OutputFormat format = OutputFormat::kLogits;
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

OutputFormat ParseFormat(const std::string_view value) {
  if (value == "spectrogram") {
    return OutputFormat::kSpectrogram;
  }
  if (value == "logits") {
    return OutputFormat::kLogits;
  }
  if (value == "beats-json") {
    return OutputFormat::kBeatsJson;
  }
  if (value == "beats-tsv") {
    return OutputFormat::kBeatsTsv;
  }
  ThrowUsage("unknown value for --format");
}

Options ParseOptions(const int argc, char** argv) {
  Options options;
  bool saw_format = false;

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
    } else if (arg == "--format") {
      options.format = ParseFormat(NextValue(index, argc, argv));
      saw_format = true;
    } else {
      ThrowUsage("unknown argument");
    }
  }

  if (options.format != OutputFormat::kSpectrogram && options.model_path.empty()) {
    ThrowUsage("--model is required");
  }
  if (options.output_path.empty()) {
    ThrowUsage("--output is required");
  }
  if (!saw_format) {
    ThrowUsage("--format is required");
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

FrameLogits LoadLogits(const BeatThisOnnxRunner& runner, const Options& options) {
  if (!options.audio_path.empty()) {
    return runner.ProcessFile(options.audio_path);
  }

  const std::vector<float> waveform = ReadWaveformBinary(options.waveform_path);
  return runner.ProcessWaveform(AudioBufferView{
      .samples = std::span<const float>(waveform),
      .sample_rate = options.sample_rate,
      .num_channels = options.num_channels,
      .layout = ChannelLayout::kInterleavedFrames,
  });
}

Spectrogram LoadSpectrogram(const BeatThisPreprocessor& preprocessor, const Options& options) {
  if (!options.audio_path.empty()) {
    return preprocessor.ProcessFile(options.audio_path);
  }

  const std::vector<float> waveform = ReadWaveformBinary(options.waveform_path);
  return preprocessor.ProcessWaveform(AudioBufferView{
      .samples = std::span<const float>(waveform),
      .sample_rate = options.sample_rate,
      .num_channels = options.num_channels,
      .layout = ChannelLayout::kInterleavedFrames,
  });
}

void EnsureOutputDirectory(const std::filesystem::path& output_path) {
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
}

void WriteLogitsTsv(const std::filesystem::path& output_path, const FrameLogits& logits) {
  EnsureOutputDirectory(output_path);

  std::ofstream output(output_path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("could not open logits output path: " + output_path.string());
  }

  output << "frame\tbeat\tdownbeat\n";
  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (int frame = 0; frame < logits.num_frames; ++frame) {
    output << frame << '\t' << logits.beat[static_cast<std::size_t>(frame)] << '\t'
           << logits.downbeat[static_cast<std::size_t>(frame)] << '\n';
  }
}

void WriteNumberArray(std::ofstream& output, const std::span<const double> values) {
  output << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) {
      output << ", ";
    }
    output << values[index];
  }
  output << "]";
}

void WriteBeatsJson(const std::filesystem::path& output_path, const BeatTimestamps& timestamps) {
  EnsureOutputDirectory(output_path);

  std::ofstream output(output_path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("could not open beats output path: " + output_path.string());
  }

  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "{\n  \"beats\": ";
  WriteNumberArray(output, timestamps.beats);
  output << ",\n  \"downbeats\": ";
  WriteNumberArray(output, timestamps.downbeats);
  output << "\n}\n";
}

void WriteSpectrogramTsv(const std::filesystem::path& output_path, const Spectrogram& spectrogram) {
  EnsureOutputDirectory(output_path);

  std::ofstream output(output_path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("could not open spectrogram output path: " + output_path.string());
  }

  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (int frame = 0; frame < spectrogram.num_frames; ++frame) {
    for (int bin = 0; bin < spectrogram.num_bins; ++bin) {
      if (bin != 0) {
        output << '\t';
      }
      output << spectrogram.values[static_cast<std::size_t>(frame * spectrogram.num_bins + bin)];
    }
    output << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);

    if (options.format == OutputFormat::kSpectrogram) {
      const BeatThisPreprocessor preprocessor;
      const Spectrogram spectrogram = LoadSpectrogram(preprocessor, options);
      WriteSpectrogramTsv(options.output_path, spectrogram);
      return 0;
    }

    const BeatThisOnnxRunner runner(options.model_path);
    const FrameLogits logits = LoadLogits(runner, options);

    if (options.format == OutputFormat::kLogits) {
      WriteLogitsTsv(options.output_path, logits);
      return 0;
    }

    const BeatTimestamps timestamps = MinimalBeatPostprocessor().Process(logits);
    if (options.format == OutputFormat::kBeatsJson) {
      WriteBeatsJson(options.output_path, timestamps);
      return 0;
    }

    WriteBeatTsv(timestamps, options.output_path);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
