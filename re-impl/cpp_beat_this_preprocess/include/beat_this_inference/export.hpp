#pragma once

#include "beat_this_inference/postprocessor.hpp"

#include <filesystem>

namespace beat_this::inference {

void WriteBeatTsv(const BeatTimestamps& timestamps, const std::filesystem::path& output_path);

}  // namespace beat_this::inference
