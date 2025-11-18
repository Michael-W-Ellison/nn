// File: include/learning/pattern_importance.hpp
//
// PatternImportanceCalculator: Scoring patterns by importance
//
// This class provides multiple scoring methods to evaluate pattern importance:
// - Frequency: How often the pattern is accessed (log-scaled)
// - Confidence: Pattern's intrinsic confidence score
// - Association Richness: Number and quality of associations
// - Success Rate: Historical prediction accuracy
//
// These scores are combined into a single importance score using configurable weights.
//
// Usage:
// @code
//   ImportanceWeights weights;
//   weights.frequency = 0.3f;
//   weights.confidence = 0.3f;
//   weights.association = 0.2f;
//   weights.success_rate = 0.2f;
//
//   PatternImportanceCalculator calculator(pattern_db, weights);
//   float importance = calculator.ComputeImportance(pattern_id);
// @endcode

#ifndef DPAN_LEARNING_PATTERN_IMPORTANCE_HPP
#define DPAN_LEARNING_PATTERN_IMPORTANCE_HPP

#include "core/types.hpp"
#include "storage/pattern_database.hpp"
#include <map>
#include <mutex>
#include <cmath>

namespace dpan {
namespace attention {

/// Configuration for importance score weights
///
/// All weights should sum to 1.0 for proper normalization.
/// Default weights: frequency=0.3, confidence=0.3, association=0.2, success_rate=0.2
struct ImportanceWeights {
    float frequency = 0.3f;      ///< Weight for access frequency score
    float confidence = 0.3f;     ///< Weight for confidence score
    float association = 0.2f;    ///< Weight for association richness score
    float success_rate = 0.2f;   ///< Weight for prediction success rate

    /// Validate that weights are non-negative and sum to approximately 1.0
    ///
    /// @return true if weights are valid, false otherwise
    bool Validate() const {
        if (frequency < 0.0f || confidence < 0.0f ||
            association < 0.0f || success_rate < 0.0f) {
            return false;
        }

        float sum = frequency + confidence + association + success_rate;
        return std::abs(sum - 1.0f) < 0.01f;  // Allow small floating point error
    }

    /// Normalize weights to sum to 1.0
    void Normalize() {
        float sum = frequency + confidence + association + success_rate;
        if (sum > 0.0f) {
            frequency /= sum;
            confidence /= sum;
            association /= sum;
            success_rate /= sum;
        }
    }
};

/// Success rate tracking data for a pattern
struct SuccessRateData {
    float total_predictions = 0.0f;        ///< Total number of predictions (with decay)
    float successful_predictions = 0.0f;   ///< Number of successful predictions (with decay)
    float decay_factor = 0.95f;            ///< Decay factor for recency bias

    /// Get success rate as a value in [0, 1]
    ///
    /// @return Success rate, or 0.5 if no predictions yet
    float GetRate() const {
        if (total_predictions < 0.001f) {  // Essentially zero
            return 0.5f;  // Neutral score for new patterns
        }
        float rate = successful_predictions / total_predictions;
        // Clamp to [0, 1]
        return std::min(std::max(rate, 0.0f), 1.0f);
    }

    /// Record a prediction result
    ///
    /// @param success true if prediction was successful, false otherwise
    void RecordPrediction(bool success) {
        // Apply decay to give more weight to recent predictions
        total_predictions = total_predictions * decay_factor + 1.0f;
        successful_predictions = successful_predictions * decay_factor +
                                (success ? 1.0f : 0.0f);
    }
};

/// PatternImportanceCalculator: Evaluates pattern importance using multiple factors
///
/// This class provides a comprehensive importance scoring system that considers:
/// 1. **Frequency**: Patterns accessed more often are more important
/// 2. **Confidence**: Patterns with higher confidence are more important
/// 3. **Association Richness**: Patterns with many strong associations are more important
/// 4. **Success Rate**: Patterns that lead to accurate predictions are more important
///
/// Each factor is scored in [0, 1] and combined using configurable weights.
///
/// **Thread Safety**: All methods are thread-safe through mutex protection.
///
/// **Persistence**: Success rate data is maintained in memory. For production use,
/// consider persisting this data to disk.
class PatternImportanceCalculator {
public:
    /// Constructor with pattern database and optional weights
    ///
    /// @param db Pattern database for accessing pattern information (not owned)
    /// @param weights Importance weights (default: balanced weights)
    explicit PatternImportanceCalculator(
        PatternDatabase* db,
        const ImportanceWeights& weights = {}
    );

    /// Destructor
    ~PatternImportanceCalculator() = default;

    // ========================================================================
    // Individual Scoring Methods
    // ========================================================================

    /// Compute frequency-based importance score
    ///
    /// Uses log-scaled access count to prevent outlier dominance.
    /// Formula: score = log(1 + access_count) / log(1 + max_access_count)
    ///
    /// @param pattern_id Pattern to score
    /// @return Frequency score in [0, 1], or 0.0 if pattern not found
    float ComputeFrequencyScore(PatternID pattern_id) const;

    /// Compute confidence-based importance score
    ///
    /// Returns the pattern's intrinsic confidence score.
    ///
    /// @param pattern_id Pattern to score
    /// @return Confidence score in [0, 1], or 0.5 if pattern not found
    float ComputeConfidenceScore(PatternID pattern_id) const;

    /// Compute association richness score
    ///
    /// Scores based on number of strong associations (strength > 0.5).
    /// Formula: score = log(1 + strong_assoc_count) / log(1 + max_assoc_count)
    ///
    /// @param pattern_id Pattern to score
    /// @return Association score in [0, 1], or 0.0 if pattern not found
    ///
    /// @note Currently returns 0.5 as placeholder until AssociationSystem is integrated
    float ComputeAssociationScore(PatternID pattern_id) const;

    /// Compute prediction success rate score
    ///
    /// Returns the historical accuracy of predictions involving this pattern.
    ///
    /// @param pattern_id Pattern to score
    /// @return Success rate in [0, 1], or 0.5 if no history
    float ComputeSuccessRateScore(PatternID pattern_id) const;

    // ========================================================================
    // Combined Importance Scoring
    // ========================================================================

    /// Compute overall importance score
    ///
    /// Combines all individual scores using configured weights:
    /// importance = w_freq * freq_score + w_conf * conf_score +
    ///              w_assoc * assoc_score + w_success * success_score
    ///
    /// @param pattern_id Pattern to score
    /// @return Overall importance score in [0, 1]
    float ComputeImportance(PatternID pattern_id) const;

    /// Compute importance for multiple patterns
    ///
    /// Batch version for efficiency.
    ///
    /// @param pattern_ids Patterns to score
    /// @return Map from pattern ID to importance score
    std::map<PatternID, float> ComputeImportanceBatch(
        const std::vector<PatternID>& pattern_ids
    ) const;

    // ========================================================================
    // Success Rate Tracking
    // ========================================================================

    /// Record a prediction result for success rate tracking
    ///
    /// Call this when a prediction is made to update success statistics.
    ///
    /// @param pattern_id Pattern that was used for prediction
    /// @param success true if prediction was successful, false otherwise
    void RecordPrediction(PatternID pattern_id, bool success);

    /// Get success rate data for a pattern
    ///
    /// @param pattern_id Pattern to query
    /// @return Success rate data, or default data if not found
    SuccessRateData GetSuccessRateData(PatternID pattern_id) const;

    /// Clear all success rate tracking data
    void ClearSuccessRateData();

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set importance weights
    ///
    /// @param weights New weights (will be normalized if needed)
    void SetWeights(const ImportanceWeights& weights);

    /// Get current importance weights
    ///
    /// @return Current weights
    const ImportanceWeights& GetWeights() const;

    /// Set pattern database
    ///
    /// @param db New pattern database (not owned)
    void SetPatternDatabase(PatternDatabase* db);

    /// Set maximum access count for normalization
    ///
    /// This should be set to the maximum observed access count across all patterns
    /// for proper normalization of frequency scores.
    ///
    /// @param max_count Maximum access count
    void SetMaxAccessCount(uint32_t max_count);

    /// Get statistics about importance calculations
    ///
    /// @return Map of statistic names to values
    std::map<std::string, float> GetStatistics() const;

private:
    /// Pattern database for accessing pattern information (not owned)
    PatternDatabase* pattern_db_;

    /// Importance weights
    ImportanceWeights weights_;

    /// Success rate tracking data (pattern ID -> success data)
    mutable std::map<PatternID, SuccessRateData> success_rates_;

    /// Mutex for thread-safe access to success rates
    mutable std::mutex success_mutex_;

    /// Maximum observed access count (for normalization)
    uint32_t max_access_count_;

    /// Statistics counters
    mutable size_t importance_calculations_;
    mutable size_t success_recordings_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_PATTERN_IMPORTANCE_HPP
