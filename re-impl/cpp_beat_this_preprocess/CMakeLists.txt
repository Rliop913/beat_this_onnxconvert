cmake_minimum_required(VERSION 3.20)

set(_DEFAULT_CONAN_TOOLCHAIN "${CMAKE_CURRENT_BINARY_DIR}/generators/conan_toolchain.cmake")
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${_DEFAULT_CONAN_TOOLCHAIN}")
  set(CMAKE_TOOLCHAIN_FILE "${_DEFAULT_CONAN_TOOLCHAIN}" CACHE FILEPATH "Conan toolchain file")
endif()

project(beat_this_preprocess_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

enable_testing()

include(FetchContent)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SOXR_FORK_COMMIT "a66f3eeeeb62a32403ff143b756eed92b1ec6b62")
list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/generators")

find_package(onnxruntime CONFIG QUIET)
if(NOT onnxruntime_FOUND)
  message(FATAL_ERROR
    "onnxruntime was not found. Run Conan first:\n"
    "  conan install \"${CMAKE_CURRENT_SOURCE_DIR}\" --output-folder=\"${CMAKE_CURRENT_BINARY_DIR}\" -s compiler.cppstd=20 --build=missing\n"
    "Then re-run CMake configure."
  )
endif()

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WITH_LSR_BINDINGS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)
set(WITH_DEV_TRACE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  soxr_fork
  GIT_REPOSITORY https://github.com/dofuuz/soxr.git
  GIT_TAG ${SOXR_FORK_COMMIT}
  GIT_SHALLOW TRUE
  UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(soxr_fork)

add_library(
  beat_this_preprocess_cpp
  "${REPO_ROOT}/cpp_root/include/stft_wrapper/ConfigurableSerialStft.cpp"
  src/internal/common.cpp
  src/internal/decode.cpp
  src/internal/frontend.cpp
  src/internal/soxr_resampler.cpp
  src/internal/waveform.cpp
  src/preprocessor.cpp
  "${REPO_ROOT}/cpp_root/include/miniaudio.c"
)

target_include_directories(
  beat_this_preprocess_cpp
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${REPO_ROOT}/cpp_root/include"
    "${soxr_fork_SOURCE_DIR}/src"
    "${soxr_fork_BINARY_DIR}"
)

target_compile_definitions(
  beat_this_preprocess_cpp
  PRIVATE
    MA_NO_DEVICE_IO=1
)

target_link_libraries(
  beat_this_preprocess_cpp
  PRIVATE
    soxr
)

if(NOT WIN32)
  target_link_libraries(beat_this_preprocess_cpp PRIVATE m)
endif()

add_executable(
  beat_this_preprocess_tests
  tests/test_main.cpp
)

target_link_libraries(beat_this_preprocess_tests PRIVATE beat_this_preprocess_cpp)
target_include_directories(beat_this_preprocess_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(
  beat_this_inference_cpp
  src/internal/aggregate.cpp
  src/internal/onnx_session.cpp
  src/internal/split.cpp
  src/inference.cpp
  src/postprocessor.cpp
)

target_include_directories(
  beat_this_inference_cpp
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(
  beat_this_inference_cpp
  PUBLIC
    beat_this_preprocess_cpp
  PRIVATE
    onnxruntime::onnxruntime
)

add_executable(
  beat_this_inference_tests
  tests/inference_test_main.cpp
)

target_link_libraries(beat_this_inference_tests PRIVATE beat_this_inference_cpp)
target_include_directories(beat_this_inference_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(
  beat_this_dump
  tools/beat_this_dump_main.cpp
)

target_link_libraries(beat_this_dump PRIVATE beat_this_inference_cpp)

set(TEST_AUDIO "${REPO_ROOT}/tests/It Don't Mean A Thing - Kings of Swing.mp3")
set(TEST_MODEL "${REPO_ROOT}/re-impl/models/beat_this_model_final0.onnx")

add_test(
  NAME beat_this_preprocess_native
  COMMAND beat_this_preprocess_tests "${TEST_AUDIO}"
)

if(EXISTS "${TEST_MODEL}")
  add_test(
    NAME beat_this_inference_native
    COMMAND beat_this_inference_tests "${TEST_MODEL}" "${TEST_AUDIO}"
  )
else()
  message(STATUS "Skipping beat_this_inference_native registration because ${TEST_MODEL} was not found.")
endif()

set(TEST_CHECKPOINT "")
if(EXISTS "${REPO_ROOT}/checkpoints/beat_this-final0.ckpt")
  set(TEST_CHECKPOINT "${REPO_ROOT}/checkpoints/beat_this-final0.ckpt")
elseif(EXISTS "${REPO_ROOT}/beat_this-final0.ckpt")
  set(TEST_CHECKPOINT "${REPO_ROOT}/beat_this-final0.ckpt")
endif()

set(_PYTHON_HINTS)
if(EXISTS "${REPO_ROOT}/.venv")
  list(APPEND _PYTHON_HINTS
    "${REPO_ROOT}/.venv"
    "${REPO_ROOT}/.venv/bin"
    "${REPO_ROOT}/.venv/Scripts"
  )
endif()
if(EXISTS "${REPO_ROOT}/.venv/bin/python")
  set(Python3_EXECUTABLE "${REPO_ROOT}/.venv/bin/python" CACHE FILEPATH
      "Python interpreter for parity tests" FORCE)
elseif(EXISTS "${REPO_ROOT}/.venv/Scripts/python.exe")
  set(Python3_EXECUTABLE "${REPO_ROOT}/.venv/Scripts/python.exe" CACHE FILEPATH
      "Python interpreter for parity tests" FORCE)
endif()
find_package(Python3 COMPONENTS Interpreter QUIET HINTS ${_PYTHON_HINTS})

if(Python3_EXECUTABLE)
  add_test(
    NAME beat_this_python_spectrogram_parity
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_cpp_python_spectrogram.py"
      --cpp-exe "$<TARGET_FILE:beat_this_dump>"
      --artifact-dir "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts"
  )
  set_tests_properties(
    beat_this_python_spectrogram_parity
    PROPERTIES
      WORKING_DIRECTORY "${REPO_ROOT}"
      SKIP_RETURN_CODE 125
  )

else()
  message(STATUS "Skipping beat_this_python_spectrogram_parity registration because Python3 is unavailable.")
endif()

if(Python3_EXECUTABLE AND EXISTS "${TEST_MODEL}" AND TEST_CHECKPOINT)

  add_test(
    NAME beat_this_python_logits_parity
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_cpp_python_logits.py"
      --checkpoint "${TEST_CHECKPOINT}"
      --model "${TEST_MODEL}"
      --cpp-exe "$<TARGET_FILE:beat_this_dump>"
      --artifact-dir "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts"
  )
  set_tests_properties(
    beat_this_python_logits_parity
    PROPERTIES
      WORKING_DIRECTORY "${REPO_ROOT}"
      SKIP_RETURN_CODE 125
  )

  add_test(
    NAME beat_this_python_minimal_beats_parity
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_cpp_python_minimal_beats.py"
      --checkpoint "${TEST_CHECKPOINT}"
      --model "${TEST_MODEL}"
      --cpp-exe "$<TARGET_FILE:beat_this_dump>"
      --artifact-dir "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts"
  )
  set_tests_properties(
    beat_this_python_minimal_beats_parity
    PROPERTIES
      WORKING_DIRECTORY "${REPO_ROOT}"
      SKIP_RETURN_CODE 125
  )

  add_test(
    NAME beat_this_python_beats_tsv_parity
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_cpp_python_beats_tsv.py"
      --checkpoint "${TEST_CHECKPOINT}"
      --model "${TEST_MODEL}"
      --cpp-exe "$<TARGET_FILE:beat_this_dump>"
      --artifact-dir "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts"
  )
  set_tests_properties(
    beat_this_python_beats_tsv_parity
    PROPERTIES
      WORKING_DIRECTORY "${REPO_ROOT}"
      SKIP_RETURN_CODE 125
  )
else()
  message(STATUS "Skipping beat_this_python_logits_parity registration because Python3, the ONNX model, or the checkpoint is unavailable.")
endif()
