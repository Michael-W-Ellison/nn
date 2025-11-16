// File: src/similarity/frequency_similarity.hpp
#pragma once

#include "similarity_metric.hpp"
#include <vector>
#include <complex>
#include <cmath>

namespace dpan {

/// Frequency domain utilities for signal analysis
class FrequencyAnalysis {
public:
    /// Complex number type
    using Complex = std::complex<float>;

    /// Compute Discrete Fourier Transform (DFT)
    /// @param signal Input signal
    /// @return Frequency domain representation
    static std::vector<Complex> DFT(const std::vector<float>& signal);

    /// Compute power spectral density from signal
    /// @param signal Input signal
    /// @return Power spectrum (magnitude squared of DFT)
    static std::vector<float> PowerSpectrum(const std::vector<float>& signal);

    /// Compute autocorrelation of signal
    /// @param signal Input signal
    /// @param max_lag Maximum lag to compute (0 = full autocorrelation)
    /// @return Autocorrelation values
    static std::vector<float> Autocorrelation(const std::vector<float>& signal, size_t max_lag = 0);

    /// Normalize signal to zero mean and unit variance
    /// @param signal Input signal
    /// @return Normalized signal
    static std::vector<float> Normalize(const std::vector<float>& signal);

    /// Extract signal from feature vector
    /// @param features Feature vector
    /// @return Signal data
    static std::vector<float> ExtractSignal(const FeatureVector& features);

private:
    /// Compute mean of signal
    static float Mean(const std::vector<float>& signal);

    /// Compute standard deviation of signal
    static float StdDev(const std::vector<float>& signal, float mean);
};

/// Spectral Similarity
///
/// Compares patterns based on their frequency domain representations.
/// Computes similarity between power spectra using correlation.
///
/// Use cases:
/// - Audio/speech recognition
/// - Vibration analysis
/// - Periodic pattern detection
class SpectralSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param normalize Whether to normalize signals before comparison
    explicit SpectralSimilarity(bool normalize = true)
        : normalize_(normalize) {}

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "Spectral"; }
    bool IsSymmetric() const override { return true; }

private:
    bool normalize_;

    /// Compute spectral correlation between two power spectra
    static float SpectralCorrelation(const std::vector<float>& spectrum_a,
                                     const std::vector<float>& spectrum_b);
};

/// Autocorrelation Similarity
///
/// Compares patterns based on their autocorrelation functions.
/// Captures periodic and self-similar structures.
///
/// Use cases:
/// - Detecting repeating patterns
/// - Rhythm analysis
/// - Texture similarity
class AutocorrelationSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param max_lag Maximum lag for autocorrelation (0 = auto)
    /// @param normalize Whether to normalize signals
    explicit AutocorrelationSimilarity(size_t max_lag = 0, bool normalize = true)
        : max_lag_(max_lag), normalize_(normalize) {}

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "Autocorrelation"; }
    bool IsSymmetric() const override { return true; }

private:
    size_t max_lag_;
    bool normalize_;

    /// Compute correlation between autocorrelation functions
    static float AutocorrelationCorrelation(const std::vector<float>& ac_a,
                                           const std::vector<float>& ac_b);
};

/// Frequency Band Energy Similarity
///
/// Divides frequency spectrum into bands and compares energy distribution.
/// Similar to how humans perceive sound (mel-scale, bark scale).
///
/// Use cases:
/// - Audio fingerprinting
/// - Music genre classification
/// - Environmental sound recognition
class FrequencyBandSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param num_bands Number of frequency bands
    /// @param normalize Whether to normalize energy across bands
    explicit FrequencyBandSimilarity(size_t num_bands = 8, bool normalize = true)
        : num_bands_(num_bands), normalize_(normalize) {
        if (num_bands_ == 0) {
            num_bands_ = 1;
        }
    }

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "FrequencyBand"; }
    bool IsSymmetric() const override { return true; }

private:
    size_t num_bands_;
    bool normalize_;

    /// Extract energy from frequency bands
    static std::vector<float> ExtractBandEnergy(const std::vector<float>& power_spectrum,
                                               size_t num_bands);

    /// Compute cosine similarity between band energies
    static float BandEnergySimilarity(const std::vector<float>& bands_a,
                                     const std::vector<float>& bands_b);
};

/// Phase Similarity
///
/// Compares phase information from Fourier transform.
/// Useful for signals where phase coherence is important.
///
/// Use cases:
/// - Coherent signal detection
/// - Synchronization analysis
/// - Wave interference patterns
class PhaseSimilarity : public SimilarityMetric {
public:
    PhaseSimilarity() = default;

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "Phase"; }
    bool IsSymmetric() const override { return true; }

private:
    /// Compute phase coherence between two complex spectra
    static float PhaseCoherence(const std::vector<FrequencyAnalysis::Complex>& spectrum_a,
                               const std::vector<FrequencyAnalysis::Complex>& spectrum_b);
};

} // namespace dpan
