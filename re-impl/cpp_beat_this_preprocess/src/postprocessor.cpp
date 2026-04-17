#include "beat_this_inference/inference.hpp"

#include "internal/common.hpp"

#include <charconv>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace beat_this::inference {

namespace {

using beat_this::preprocess::internal::ThrowInvalid;

constexpr int kPeakWindowSize = 7;
constexpr int kPeakRadius = kPeakWindowSize / 2;
constexpr int kDeduplicateWidth = 1;

void ValidateFrameLogits(const FrameLogits& logits) {
  if (logits.num_frames < 0) {
    ThrowInvalid("frame logits num_frames must not be negative");
  }

  const std::size_t expected_size = static_cast<std::size_t>(logits.num_frames);
  if (logits.beat.size() != expected_size) {
    ThrowInvalid("frame logits beat size does not match num_frames");
  }
  if (logits.downbeat.size() != expected_size) {
    ThrowInvalid("frame logits downbeat size does not match num_frames");
  }
}

bool IsLocalMaximum(const std::span<const float> values, const int index) {
  const int start = std::max(0, index - kPeakRadius);
  const int end = std::min(static_cast<int>(values.size()) - 1, index + kPeakRadius);

  float local_max = -std::numeric_limits<float>::infinity();
  for (int cursor = start; cursor <= end; ++cursor) {
    local_max = std::max(local_max, values[static_cast<std::size_t>(cursor)]);
  }
  return values[static_cast<std::size_t>(index)] == local_max;
}

std::vector<int> CollectPeakFrames(const std::span<const float> logits) {
  std::vector<int> peaks;
  peaks.reserve(logits.size());

  for (int frame = 0; frame < static_cast<int>(logits.size()); ++frame) {
    const float value = logits[static_cast<std::size_t>(frame)];
    if (value <= 0.0F) {
      continue;
    }
    if (IsLocalMaximum(logits, frame)) {
      peaks.push_back(frame);
    }
  }
  return peaks;
}

std::vector<double> DeduplicatePeaks(const std::span<const int> peaks) {
  std::vector<double> deduplicated;
  if (peaks.empty()) {
    return deduplicated;
  }

  deduplicated.reserve(peaks.size());
  double mean = static_cast<double>(peaks.front());
  int count = 1;

  for (std::size_t index = 1; index < peaks.size(); ++index) {
    const int current = peaks[index];
    if ((static_cast<double>(current) - mean) <= static_cast<double>(kDeduplicateWidth)) {
      ++count;
      mean += (static_cast<double>(current) - mean) / static_cast<double>(count);
      continue;
    }

    deduplicated.push_back(mean);
    mean = static_cast<double>(current);
    count = 1;
  }

  deduplicated.push_back(mean);
  return deduplicated;
}

std::vector<double> ConvertFramesToSeconds(const std::span<const double> frame_positions,
                                           const double fps) {
  std::vector<double> times;
  times.reserve(frame_positions.size());
  for (const double frame : frame_positions) {
    times.push_back(frame / fps);
  }
  return times;
}

std::vector<double> SnapDownbeatsToNearestBeats(const std::span<const double> beat_times,
                                                const std::span<const double> downbeat_times) {
  std::vector<double> snapped(downbeat_times.begin(), downbeat_times.end());
  if (beat_times.empty()) {
    std::sort(snapped.begin(), snapped.end());
    snapped.erase(std::unique(snapped.begin(), snapped.end()), snapped.end());
    return snapped;
  }

  for (double& downbeat : snapped) {
    const auto upper = std::lower_bound(beat_times.begin(), beat_times.end(), downbeat);
    if (upper == beat_times.begin()) {
      downbeat = *upper;
      continue;
    }
    if (upper == beat_times.end()) {
      downbeat = beat_times.back();
      continue;
    }

    const double previous = *(upper - 1);
    const double next = *upper;
    downbeat = (std::abs(downbeat - previous) <= std::abs(next - downbeat)) ? previous : next;
  }

  std::sort(snapped.begin(), snapped.end());
  snapped.erase(std::unique(snapped.begin(), snapped.end()), snapped.end());
  return snapped;
}

std::size_t SearchSorted(const std::span<const double> values, const double target) {
  return static_cast<std::size_t>(
      std::lower_bound(values.begin(), values.end(), target) - values.begin());
}

std::string FormatBeatTime(const double value) {
  std::array<char, 64> buffer{};
  const auto result =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general);
  if (result.ec != std::errc()) {
    ThrowInvalid("failed to format beat timestamp");
  }

  std::string formatted(buffer.data(), result.ptr);
  if (formatted.find_first_of(".eE") == std::string::npos) {
    formatted += ".0";
  }
  return formatted;
}

}  // namespace

MinimalBeatPostprocessor::MinimalBeatPostprocessor(const double fps) : fps_(fps) {
  if (!std::isfinite(fps_) || fps_ <= 0.0) {
    ThrowInvalid("minimal beat postprocessor fps must be positive and finite");
  }
}

BeatTimestamps MinimalBeatPostprocessor::Process(const FrameLogits& logits) const {
  ValidateFrameLogits(logits);

  const std::vector<int> beat_peaks = CollectPeakFrames(logits.beat);
  const std::vector<int> downbeat_peaks = CollectPeakFrames(logits.downbeat);

  const std::vector<double> beat_frames = DeduplicatePeaks(beat_peaks);
  const std::vector<double> downbeat_frames = DeduplicatePeaks(downbeat_peaks);

  const std::vector<double> beat_times = ConvertFramesToSeconds(beat_frames, fps_);
  const std::vector<double> downbeat_times = ConvertFramesToSeconds(downbeat_frames, fps_);

  return BeatTimestamps{
      .beats = beat_times,
      .downbeats = SnapDownbeatsToNearestBeats(beat_times, downbeat_times),
  };
}

std::vector<int> InferBeatNumbers(const BeatTimestamps& timestamps) {
  for (const double downbeat : timestamps.downbeats) {
    if (!std::binary_search(timestamps.beats.begin(), timestamps.beats.end(), downbeat)) {
      ThrowInvalid("not all downbeats are beats");
    }
  }

  int start_counter = 1;
  if (timestamps.downbeats.size() >= 2U) {
    const std::size_t first_downbeat = SearchSorted(timestamps.beats, timestamps.downbeats[0]);
    const std::size_t second_downbeat = SearchSorted(timestamps.beats, timestamps.downbeats[1]);
    const int beats_in_first_measure = static_cast<int>(second_downbeat - first_downbeat);
    const int pickup_beats = static_cast<int>(first_downbeat);
    if (pickup_beats < beats_in_first_measure) {
      start_counter = beats_in_first_measure - pickup_beats;
    } else {
      std::cerr
          << "WARNING: There are more beats in the pickup measure than in the first measure. "
             "The beat count will start from 2 without trying to estimate the length of the "
             "pickup measure.\n";
      start_counter = 1;
    }
  } else {
    std::cerr << "WARNING: There are less than two downbeats in the predictions. Something may "
                 "be wrong. The beat count will start from 2 without trying to estimate the "
                 "length of the pickup measure.\n";
  }

  std::vector<int> numbers;
  numbers.reserve(timestamps.beats.size());

  int counter = start_counter;
  std::size_t next_downbeat_index = 0;
  double next_downbeat = timestamps.downbeats.empty() ? -1.0 : timestamps.downbeats.front();

  for (const double beat : timestamps.beats) {
    if (beat == next_downbeat) {
      counter = 1;
      ++next_downbeat_index;
      next_downbeat = (next_downbeat_index < timestamps.downbeats.size())
                          ? timestamps.downbeats[next_downbeat_index]
                          : -1.0;
    } else {
      ++counter;
    }
    numbers.push_back(counter);
  }

  return numbers;
}

void WriteBeatTsv(const BeatTimestamps& timestamps, const std::filesystem::path& output_path) {
  const std::vector<int> numbers = InferBeatNumbers(timestamps);

  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream output(output_path, std::ios::binary);
  if (!output) {
    ThrowInvalid("could not open beat tsv output path");
  }

  for (std::size_t index = 0; index < timestamps.beats.size(); ++index) {
    output << FormatBeatTime(timestamps.beats[index]) << '\t' << numbers[index] << '\n';
  }
}

}  // namespace beat_this::inference
