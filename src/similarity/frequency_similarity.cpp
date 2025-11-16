// File: src/similarity/frequency_similarity.cpp
#include "frequency_similarity.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace dpan {

// ============================================================================
// FrequencyAnalysis Implementation
// ============================================================================

std::vector<FrequencyAnalysis::Complex> FrequencyAnalysis::DFT(const std::vector<float>& signal) {
    size_t N = signal.size();
    if (N == 0) {
        return {};
    }

    std::vector<Complex> result(N);
    const float pi = 3.14159265358979323846f;

    for (size_t k = 0; k < N; ++k) {
        Complex sum(0.0f, 0.0f);
        for (size_t n = 0; n < N; ++n) {
            float angle = -2.0f * pi * k * n / N;
            Complex exponential(std::cos(angle), std::sin(angle));
            sum += signal[n] * exponential;
        }
        result[k] = sum;
    }

    return result;
}

std::vector<float> FrequencyAnalysis::PowerSpectrum(const std::vector<float>& signal) {
    auto dft = DFT(signal);
    std::vector<float> power(dft.size());

    for (size_t i = 0; i < dft.size(); ++i) {
        power[i] = std::norm(dft[i]);  // magnitude squared
    }

    return power;
}

std::vector<float> FrequencyAnalysis::Autocorrelation(const std::vector<float>& signal, size_t max_lag) {
    size_t N = signal.size();
    if (N == 0) {
        return {};
    }

    // Default max_lag to N-1 if not specified
    if (max_lag == 0 || max_lag >= N) {
        max_lag = N - 1;
    }

    std::vector<float> ac(max_lag + 1);

    // Compute mean
    float mean = Mean(signal);

    // Compute variance
    float variance = 0.0f;
    for (float val : signal) {
        float diff = val - mean;
        variance += diff * diff;
    }

    if (variance < 1e-10f) {
        // Constant signal
        ac[0] = 1.0f;
        return ac;
    }

    // Compute autocorrelation
    for (size_t lag = 0; lag <= max_lag; ++lag) {
        float sum = 0.0f;
        for (size_t i = 0; i < N - lag; ++i) {
            sum += (signal[i] - mean) * (signal[i + lag] - mean);
        }
        ac[lag] = sum / variance;
    }

    return ac;
}

std::vector<float> FrequencyAnalysis::Normalize(const std::vector<float>& signal) {
    if (signal.empty()) {
        return {};
    }

    float mean = Mean(signal);
    float std = StdDev(signal, mean);

    std::vector<float> normalized(signal.size());

    if (std < 1e-10f) {
        // Constant signal, return zeros
        return normalized;
    }

    for (size_t i = 0; i < signal.size(); ++i) {
        normalized[i] = (signal[i] - mean) / std;
    }

    return normalized;
}

std::vector<float> FrequencyAnalysis::ExtractSignal(const FeatureVector& features) {
    const auto& data = features.Data();
    return std::vector<float>(data.begin(), data.end());
}

float FrequencyAnalysis::Mean(const std::vector<float>& signal) {
    if (signal.empty()) {
        return 0.0f;
    }
    return std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
}

float FrequencyAnalysis::StdDev(const std::vector<float>& signal, float mean) {
    if (signal.empty()) {
        return 0.0f;
    }

    float variance = 0.0f;
    for (float val : signal) {
        float diff = val - mean;
        variance += diff * diff;
    }

    return std::sqrt(variance / signal.size());
}

// ============================================================================
// SpectralSimilarity Implementation
// ============================================================================

float SpectralSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float SpectralSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    auto signal_a = FrequencyAnalysis::ExtractSignal(a);
    auto signal_b = FrequencyAnalysis::ExtractSignal(b);

    if (normalize_) {
        signal_a = FrequencyAnalysis::Normalize(signal_a);
        signal_b = FrequencyAnalysis::Normalize(signal_b);
    }

    auto spectrum_a = FrequencyAnalysis::PowerSpectrum(signal_a);
    auto spectrum_b = FrequencyAnalysis::PowerSpectrum(signal_b);

    return SpectralCorrelation(spectrum_a, spectrum_b);
}

float SpectralSimilarity::SpectralCorrelation(const std::vector<float>& spectrum_a,
                                              const std::vector<float>& spectrum_b) {
    if (spectrum_a.empty() || spectrum_b.empty()) {
        return 0.0f;
    }

    // Use minimum length for comparison
    size_t N = std::min(spectrum_a.size(), spectrum_b.size());

    // Compute means
    float mean_a = 0.0f, mean_b = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        mean_a += spectrum_a[i];
        mean_b += spectrum_b[i];
    }
    mean_a /= N;
    mean_b /= N;

    // Compute correlation
    float numerator = 0.0f;
    float denom_a = 0.0f;
    float denom_b = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        float diff_a = spectrum_a[i] - mean_a;
        float diff_b = spectrum_b[i] - mean_b;

        numerator += diff_a * diff_b;
        denom_a += diff_a * diff_a;
        denom_b += diff_b * diff_b;
    }

    if (denom_a < 1e-10f || denom_b < 1e-10f) {
        return 0.0f;
    }

    float correlation = numerator / std::sqrt(denom_a * denom_b);

    // Convert correlation [-1, 1] to similarity [0, 1]
    return (correlation + 1.0f) / 2.0f;
}

// ============================================================================
// AutocorrelationSimilarity Implementation
// ============================================================================

float AutocorrelationSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float AutocorrelationSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    auto signal_a = FrequencyAnalysis::ExtractSignal(a);
    auto signal_b = FrequencyAnalysis::ExtractSignal(b);

    if (normalize_) {
        signal_a = FrequencyAnalysis::Normalize(signal_a);
        signal_b = FrequencyAnalysis::Normalize(signal_b);
    }

    auto ac_a = FrequencyAnalysis::Autocorrelation(signal_a, max_lag_);
    auto ac_b = FrequencyAnalysis::Autocorrelation(signal_b, max_lag_);

    return AutocorrelationCorrelation(ac_a, ac_b);
}

float AutocorrelationSimilarity::AutocorrelationCorrelation(const std::vector<float>& ac_a,
                                                           const std::vector<float>& ac_b) {
    if (ac_a.empty() || ac_b.empty()) {
        return 0.0f;
    }

    // Use minimum length
    size_t N = std::min(ac_a.size(), ac_b.size());

    // Compute dot product and norms
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        dot += ac_a[i] * ac_b[i];
        norm_a += ac_a[i] * ac_a[i];
        norm_b += ac_b[i] * ac_b[i];
    }

    if (norm_a < 1e-10f || norm_b < 1e-10f) {
        return 0.0f;
    }

    // Cosine similarity
    return dot / std::sqrt(norm_a * norm_b);
}

// ============================================================================
// FrequencyBandSimilarity Implementation
// ============================================================================

float FrequencyBandSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float FrequencyBandSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    auto signal_a = FrequencyAnalysis::ExtractSignal(a);
    auto signal_b = FrequencyAnalysis::ExtractSignal(b);

    auto spectrum_a = FrequencyAnalysis::PowerSpectrum(signal_a);
    auto spectrum_b = FrequencyAnalysis::PowerSpectrum(signal_b);

    auto bands_a = ExtractBandEnergy(spectrum_a, num_bands_);
    auto bands_b = ExtractBandEnergy(spectrum_b, num_bands_);

    if (normalize_) {
        // Normalize to sum to 1
        float sum_a = std::accumulate(bands_a.begin(), bands_a.end(), 0.0f);
        float sum_b = std::accumulate(bands_b.begin(), bands_b.end(), 0.0f);

        if (sum_a > 1e-10f) {
            for (auto& val : bands_a) val /= sum_a;
        }
        if (sum_b > 1e-10f) {
            for (auto& val : bands_b) val /= sum_b;
        }
    }

    return BandEnergySimilarity(bands_a, bands_b);
}

std::vector<float> FrequencyBandSimilarity::ExtractBandEnergy(const std::vector<float>& power_spectrum,
                                                             size_t num_bands) {
    if (power_spectrum.empty() || num_bands == 0) {
        return {};
    }

    std::vector<float> band_energy(num_bands, 0.0f);
    size_t spectrum_size = power_spectrum.size();

    // Divide spectrum into bands (logarithmic spacing would be better, but linear is simpler)
    for (size_t i = 0; i < spectrum_size; ++i) {
        size_t band = (i * num_bands) / spectrum_size;
        if (band >= num_bands) {
            band = num_bands - 1;
        }
        band_energy[band] += power_spectrum[i];
    }

    return band_energy;
}

float FrequencyBandSimilarity::BandEnergySimilarity(const std::vector<float>& bands_a,
                                                   const std::vector<float>& bands_b) {
    if (bands_a.empty() || bands_b.empty() || bands_a.size() != bands_b.size()) {
        return 0.0f;
    }

    // Cosine similarity
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < bands_a.size(); ++i) {
        dot += bands_a[i] * bands_b[i];
        norm_a += bands_a[i] * bands_a[i];
        norm_b += bands_b[i] * bands_b[i];
    }

    if (norm_a < 1e-10f || norm_b < 1e-10f) {
        return 0.0f;
    }

    return dot / std::sqrt(norm_a * norm_b);
}

// ============================================================================
// PhaseSimilarity Implementation
// ============================================================================

float PhaseSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float PhaseSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    auto signal_a = FrequencyAnalysis::ExtractSignal(a);
    auto signal_b = FrequencyAnalysis::ExtractSignal(b);

    auto spectrum_a = FrequencyAnalysis::DFT(signal_a);
    auto spectrum_b = FrequencyAnalysis::DFT(signal_b);

    return PhaseCoherence(spectrum_a, spectrum_b);
}

float PhaseSimilarity::PhaseCoherence(const std::vector<FrequencyAnalysis::Complex>& spectrum_a,
                                     const std::vector<FrequencyAnalysis::Complex>& spectrum_b) {
    if (spectrum_a.empty() || spectrum_b.empty()) {
        return 0.0f;
    }

    size_t N = std::min(spectrum_a.size(), spectrum_b.size());

    // Compute phase difference coherence
    float coherence = 0.0f;
    size_t count = 0;

    for (size_t i = 0; i < N; ++i) {
        float mag_a = std::abs(spectrum_a[i]);
        float mag_b = std::abs(spectrum_b[i]);

        // Only consider frequencies with significant magnitude
        if (mag_a > 1e-6f && mag_b > 1e-6f) {
            // Normalized complex multiplication gives phase difference
            auto normalized_a = spectrum_a[i] / mag_a;
            auto normalized_b = spectrum_b[i] / mag_b;

            // Phase coherence is real part of conjugate product
            float phase_coherence = std::real(normalized_a * std::conj(normalized_b));

            coherence += phase_coherence;
            count++;
        }
    }

    if (count == 0) {
        return 0.0f;
    }

    // Average coherence, convert from [-1, 1] to [0, 1]
    float avg_coherence = coherence / count;
    return (avg_coherence + 1.0f) / 2.0f;
}

} // namespace dpan
