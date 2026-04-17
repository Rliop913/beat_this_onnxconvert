#pragma once

#include "stft_wrapper/MelFilterBank.hpp"

#include <span>
#include <vector>

namespace PDJE_PARALLEL {

enum class SerialWindowMode {
    Hanning = 0,
    HanningPeriodic,
};

enum class SerialOutputMode {
    Magnitude = 0,
    Mel,
};

struct SerialStftConfig {
    int              sample_rate                = 48000;
    unsigned int     window_size_exp            = 10;
    unsigned int     hop_length                 = 512;
    unsigned int     mel_bins                   = 80;
    float            f_min_hz                   = 0.0f;
    float            f_max_hz                   = -1.0f;
    MelFormula       mel_formula                = MelFormula::HTK;
    MelNorm          mel_norm                   = MelNorm::None;
    SerialWindowMode window_mode                = SerialWindowMode::Hanning;
    bool             remove_dc                  = true;
    bool             normalize_by_sqrt_window   = false;
    SerialOutputMode output_mode                = SerialOutputMode::Magnitude;
};

struct SerialStftResult {
    int               num_frames = 0;
    int               num_bins   = 0;
    std::vector<float> values;
};

class ConfigurableSerialStft final {
  public:
    ConfigurableSerialStft() = default;

    SerialStftResult
    Compute(std::span<const float> samples, const SerialStftConfig &config);

  private:
    unsigned int prev_overlap_fullsize = 0;
    unsigned int prev_bin_fullsize     = 0;
    unsigned int prev_mel_fullsize     = 0;
    unsigned int prev_window_size      = 0;
    int          prev_sample_rate      = 0;
    unsigned int prev_mel_bins         = 0;
    float        prev_f_min_hz         = 0.0f;
    float        prev_f_max_hz         = 0.0f;
    MelFormula   prev_mel_formula      = MelFormula::HTK;
    MelNorm      prev_mel_norm         = MelNorm::None;

    std::vector<float> real;
    std::vector<float> imag;
    std::vector<float> subreal;
    std::vector<float> subimag;
    std::vector<float> magnitude;
    std::vector<float> mel;
    std::vector<float> periodic_window;
    std::vector<float> mel_filter_bank;

    void
    EnsureBuffers(unsigned int overlap_fullsize,
                  unsigned int bin_fullsize,
                  unsigned int mel_fullsize,
                  bool         need_subbuffer);

    void
    EnsurePeriodicWindow(unsigned int window_size);

    void
    EnsureMelFilterBank(unsigned int window_size, const SerialStftConfig &config);

    void
    ApplyWindow(const SerialStftConfig &config,
                unsigned int            num_frames,
                unsigned int            window_size);

    void
    RunFft(unsigned int window_size_exp,
           unsigned int window_size,
           unsigned int overlap_halfsize,
           bool         need_subbuffer);
};

} // namespace PDJE_PARALLEL
