#include "internal/common.hpp"

#include <stdexcept>
#include <string>

namespace beat_this::preprocess::internal {

[[noreturn]] void ThrowInvalid(const std::string_view message) {
  throw std::invalid_argument(std::string(message));
}

[[noreturn]] void ThrowRuntime(const std::string_view message) {
  throw std::runtime_error(std::string(message));
}

}  // namespace beat_this::preprocess::internal
