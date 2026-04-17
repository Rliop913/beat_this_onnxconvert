#include "stft_wrapper/ConfigurableSerialStft.hpp"

#include "STFT/GenOut/SERIAL/STFT_MAIN_SERIAL.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace PDJE_PARALLEL {

namespace {

void
ThrowInvalidConfig(const char *message)
{
    throw std::invalid_argument(message);
}

} // namespace

void
ConfigurableSerialStft::EnsureBuffers(const unsigned int overlap_fullsize,
                                      const unsigned int bin_fullsize,
                                      const unsigned int mel_fullsize,
                                      const bool         need_subbuffer)
{
    if (prev_overlap_fullsize != overlap_fullsize) {
        real.resize(overlap_fullsize);
        imag.resize(overlap_fullsize);
        prev_overlap_fullsize = overlap_fullsize;
    }

    std::fill(imag.begin(), imag.end(), 0.0f);

    if (need_subbuffer) {
        subreal.resize(overlap_fullsize);
        subimag.resize(overlap_fullsize);
        std::fill(subreal.begin(), subreal.end(), 0.0f);
        std::fill(subimag.begin(), subimag.end(), 0.0f);
    }

    if (prev_bin_fullsize != bin_fullsize) {
        magnitude.resize(bin_fullsize);
        prev_bin_fullsize = bin_fullsize;
    }

    if (prev_mel_fullsize != mel_fullsize) {
        mel.resize(mel_fullsize);
        prev_mel_fullsize = mel_fullsize;
    }
}

void
ConfigurableSerialStft::EnsurePeriodicWindow(const unsigned int window_size)
{
    if (prev_window_size == window_size && periodic_window.size() == window_size) {
        return;
    }

    periodic_window.resize(window_size);
    const float denom = static_cast<float>(window_size);
    for (unsigned int index = 0; index < window_size; ++index) {
        const float phase =
            (2.0f * std::numbers::pi_v<float> * static_cast<float>(index)) / denom;
        periodic_window[index] = 0.5f - (0.5f * std::cos(phase));
    }
    prev_window_size = window_size;
}

void
ConfigurableSerialStft::EnsureMelFilterBank(const unsigned int       window_size,
                                            const SerialStftConfig &config)
{
    const float f_max =
        config.f_max_hz < 0.0f
            ? static_cast<float>(config.sample_rate) * 0.5f
            : config.f_max_hz;
    const bool same_config =
        prev_window_size == window_size &&
        prev_sample_rate == config.sample_rate &&
        prev_mel_bins == config.mel_bins &&
        prev_f_min_hz == config.f_min_hz &&
        prev_f_max_hz == f_max &&
        prev_mel_formula == config.mel_formula &&
        prev_mel_norm == config.mel_norm;
    if (same_config && !mel_filter_bank.empty()) {
        return;
    }

    if (!CheckMelVals(config.sample_rate,
                      static_cast<int>(window_size),
                      static_cast<int>(config.mel_bins),
                      config.f_min_hz,
                      f_max,
                      config.mel_formula,
                      config.mel_norm)) {
        ThrowInvalidConfig("invalid mel filter bank configuration");
    }

    const unsigned int fft_bins = (window_size / 2U) + 1U;
    mel_filter_bank.assign(config.mel_bins * fft_bins, 0.0f);

    const float nyquist = static_cast<float>(config.sample_rate) * 0.5f;
    const float mel_min = detail::HzToMel(config.f_min_hz, config.mel_formula);
    const float mel_max = detail::HzToMel(f_max, config.mel_formula);
    const float mel_step =
        (mel_max - mel_min) / static_cast<float>(config.mel_bins + 1U);

    std::vector<float> hz_points(static_cast<std::size_t>(config.mel_bins + 2U), 0.0f);
    for (unsigned int point_index = 0; point_index < config.mel_bins + 2U; ++point_index) {
        const float mel_value =
            mel_min + (static_cast<float>(point_index) * mel_step);
        hz_points[point_index] = detail::MelToHz(mel_value, config.mel_formula);
    }

    std::vector<float> hz_widths(static_cast<std::size_t>(config.mel_bins + 1U), 0.0f);
    for (unsigned int mel_index = 0; mel_index < config.mel_bins + 1U; ++mel_index) {
        hz_widths[mel_index] = hz_points[mel_index + 1U] - hz_points[mel_index];
    }

    const float freq_step = nyquist / static_cast<float>(fft_bins - 1U);
    for (unsigned int fft_bin = 0; fft_bin < fft_bins; ++fft_bin) {
        const float bin_hz = static_cast<float>(fft_bin) * freq_step;
        for (unsigned int mel_index = 0; mel_index < config.mel_bins; ++mel_index) {
            const float down_slope =
                (bin_hz - hz_points[mel_index]) / hz_widths[mel_index];
            const float up_slope =
                (hz_points[mel_index + 2U] - bin_hz) / hz_widths[mel_index + 1U];
            const float weight = std::max(0.0f, std::min(down_slope, up_slope));
            mel_filter_bank[(mel_index * fft_bins) + fft_bin] = weight;
        }
    }

    if (config.mel_norm != MelNorm::None) {
        for (unsigned int mel_index = 0; mel_index < config.mel_bins; ++mel_index) {
            detail::ApplyNorm(mel_filter_bank.data() + (mel_index * fft_bins),
                              static_cast<int>(fft_bins),
                              hz_points[mel_index],
                              hz_points[mel_index + 2U],
                              config.mel_norm);
        }
    }

    prev_window_size = window_size;
    prev_sample_rate = config.sample_rate;
    prev_mel_bins = config.mel_bins;
    prev_f_min_hz = config.f_min_hz;
    prev_f_max_hz = f_max;
    prev_mel_formula = config.mel_formula;
    prev_mel_norm = config.mel_norm;
}

void
ConfigurableSerialStft::ApplyWindow(const SerialStftConfig &config,
                                    const unsigned int      num_frames,
                                    const unsigned int      window_size)
{
    if (config.window_mode == SerialWindowMode::HanningPeriodic) {
        EnsurePeriodicWindow(window_size);
        for (unsigned int frame_index = 0; frame_index < num_frames; ++frame_index) {
            const unsigned int frame_base = frame_index * window_size;
            for (unsigned int sample_index = 0; sample_index < window_size; ++sample_index) {
                real[frame_base + sample_index] *= periodic_window[sample_index];
            }
        }
        return;
    }

    Window_Hanning(real.data(), prev_overlap_fullsize, window_size);
}

void
ConfigurableSerialStft::RunFft(const unsigned int window_size_exp,
                               const unsigned int window_size,
                               const unsigned int overlap_halfsize,
                               const bool         need_subbuffer)
{
    switch (window_size_exp) {
    case 6:
        Stockhoptimized6(real.data(), imag.data(), overlap_halfsize);
        return;
    case 7:
        Stockhoptimized7(real.data(), imag.data(), overlap_halfsize);
        return;
    case 8:
        Stockhoptimized8(real.data(), imag.data(), overlap_halfsize);
        return;
    case 9:
        Stockhoptimized9(real.data(), imag.data(), overlap_halfsize);
        return;
    case 10:
        Stockhoptimized10(real.data(), imag.data(), overlap_halfsize);
        return;
    case 11:
        Stockhoptimized11(real.data(), imag.data(), overlap_halfsize);
        return;
    default:
        break;
    }

    if (!need_subbuffer) {
        ThrowInvalidConfig("window_size_exp requires subbuffer scratch");
    }

    const unsigned int half_window_size = window_size >> 1;
    for (unsigned int stage = 0; stage < window_size_exp; ++stage) {
        if ((stage % 2U) == 0U) {
            StockHamCommon(real.data(),
                           imag.data(),
                           subreal.data(),
                           subimag.data(),
                           half_window_size,
                           stage,
                           overlap_halfsize,
                           window_size_exp);
        } else {
            StockHamCommon(subreal.data(),
                           subimag.data(),
                           real.data(),
                           imag.data(),
                           half_window_size,
                           stage,
                           overlap_halfsize,
                           window_size_exp);
        }
    }
}

SerialStftResult
ConfigurableSerialStft::Compute(const std::span<const float> samples,
                                const SerialStftConfig      &config)
{
    if (samples.empty()) {
        ThrowInvalidConfig("samples must be non-empty");
    }
    if (config.sample_rate <= 0) {
        ThrowInvalidConfig("sample_rate must be positive");
    }
    if (config.window_size_exp == 0 || config.window_size_exp >= 31) {
        ThrowInvalidConfig("window_size_exp must be in the range [1, 30]");
    }
    if (config.hop_length == 0) {
        ThrowInvalidConfig("hop_length must be positive");
    }

    const unsigned int window_size = 1U << config.window_size_exp;
    if (samples.size() < window_size) {
        ThrowInvalidConfig("samples shorter than window size");
    }
    if (config.output_mode == SerialOutputMode::Mel && config.mel_bins == 0) {
        ThrowInvalidConfig("mel_bins must be positive when output_mode is Mel");
    }

    const unsigned int num_frames = 1U + static_cast<unsigned int>(
        (samples.size() - window_size) / config.hop_length);
    const unsigned int overlap_fullsize = num_frames * window_size;
    const unsigned int overlap_halfsize = overlap_fullsize / 2U;
    const unsigned int fft_bins = (window_size / 2U) + 1U;
    const unsigned int bin_fullsize = fft_bins * num_frames;
    const unsigned int mel_fullsize = config.mel_bins * num_frames;
    const bool need_subbuffer = config.window_size_exp > 11U;

    EnsureBuffers(overlap_fullsize, bin_fullsize, mel_fullsize, need_subbuffer);

    Overlap_Common(const_cast<float *>(samples.data()),
                   overlap_fullsize,
                   static_cast<unsigned int>(samples.size()),
                   static_cast<int>(config.window_size_exp),
                   config.hop_length,
                   real.data());

    if (config.remove_dc) {
        DCRemove_Common(real.data(), overlap_fullsize, window_size);
    }
    ApplyWindow(config, num_frames, window_size);
    RunFft(config.window_size_exp, window_size, overlap_halfsize, need_subbuffer);

    std::vector<float> *active_real = &real;
    std::vector<float> *active_imag = &imag;
    if (need_subbuffer && (config.window_size_exp % 2U) != 0U) {
        active_real = &subreal;
        active_imag = &subimag;
    }

    const float normalization =
        config.normalize_by_sqrt_window ? std::sqrt(static_cast<float>(window_size)) : 1.0f;
    for (unsigned int frame_index = 0; frame_index < num_frames; ++frame_index) {
        const unsigned int frame_base = frame_index * window_size;
        const unsigned int bin_base = frame_index * fft_bins;
        for (unsigned int bin_index = 0; bin_index < fft_bins; ++bin_index) {
            const float re = (*active_real)[frame_base + bin_index];
            const float im = (*active_imag)[frame_base + bin_index];
            magnitude[bin_base + bin_index] =
                std::sqrt((re * re) + (im * im)) / normalization;
        }
    }

    if (config.output_mode == SerialOutputMode::Magnitude) {
        return {
            .num_frames = static_cast<int>(num_frames),
            .num_bins   = static_cast<int>(fft_bins),
            .values     = std::vector<float>(magnitude.begin(), magnitude.begin() + bin_fullsize),
        };
    }

    EnsureMelFilterBank(window_size, config);
    MelScale(mel.data(),
             magnitude.data(),
             mel_filter_bank.data(),
             mel_fullsize,
             fft_bins,
             config.mel_bins);
    return {
        .num_frames = static_cast<int>(num_frames),
        .num_bins   = static_cast<int>(config.mel_bins),
        .values     = std::vector<float>(mel.begin(), mel.begin() + mel_fullsize),
    };
}

} // namespace PDJE_PARALLEL
