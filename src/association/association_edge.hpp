// File: src/association/association_edge.hpp
#pragma once

#include "core/types.hpp"
#include <atomic>
#include <memory>
#include <mutex>

namespace dpan {

/// AssociationEdge: Directed relationship between two patterns
///
/// Represents a learned association between two patterns with:
/// - Thread-safe strength management
/// - Temporal correlation tracking
/// - Context-sensitive strength modulation
/// - Time-based decay
/// - Co-occurrence statistics
class AssociationEdge {
public:
    /// Default constructor
    AssociationEdge() = default;

    /// Construct association edge
    /// @param source Source pattern ID
    /// @param target Target pattern ID
    /// @param type Association type
    /// @param initial_strength Initial strength [0,1], default 0.5
    AssociationEdge(
        PatternID source,
        PatternID target,
        AssociationType type,
        float initial_strength = 0.5f
    );

    // ========================================================================
    // Core Identity
    // ========================================================================

    /// Get source pattern ID
    PatternID GetSource() const { return source_; }

    /// Get target pattern ID
    PatternID GetTarget() const { return target_; }

    /// Get association type
    AssociationType GetType() const { return type_; }

    // ========================================================================
    // Strength Management (Thread-Safe)
    // ========================================================================

    /// Get current association strength [0,1]
    float GetStrength() const {
        return strength_.load(std::memory_order_relaxed);
    }

    /// Set association strength (clamped to [0,1])
    /// @param strength New strength value
    void SetStrength(float strength);

    /// Adjust strength by delta (bounded to [0,1])
    /// @param delta Change in strength (can be negative)
    void AdjustStrength(float delta);

    // ========================================================================
    // Co-occurrence Tracking
    // ========================================================================

    /// Get number of times patterns co-occurred
    uint32_t GetCoOccurrenceCount() const {
        return co_occurrence_count_.load(std::memory_order_relaxed);
    }

    /// Increment co-occurrence counter
    /// @param count Number to add (default 1)
    void IncrementCoOccurrence(uint32_t count = 1);

    // ========================================================================
    // Temporal Correlation
    // ========================================================================

    /// Get temporal correlation coefficient [-1,1]
    /// Positive: target typically follows source
    /// Negative: target typically precedes source
    float GetTemporalCorrelation() const {
        return temporal_correlation_.load(std::memory_order_relaxed);
    }

    /// Set temporal correlation (clamped to [-1,1])
    /// @param correlation Temporal correlation value
    void SetTemporalCorrelation(float correlation);

    /// Update temporal correlation with new observation
    /// Uses exponential moving average
    /// @param new_observation New correlation measurement
    /// @param learning_rate Learning rate [0,1], default 0.1
    void UpdateTemporalCorrelation(float new_observation, float learning_rate = 0.1f);

    // ========================================================================
    // Decay Management
    // ========================================================================

    /// Get decay rate (strength loss per second)
    float GetDecayRate() const { return decay_rate_; }

    /// Set decay rate
    /// @param rate Decay rate (clamped to >= 0)
    void SetDecayRate(float rate);

    /// Get timestamp of last reinforcement
    Timestamp GetLastReinforcement() const;

    /// Record reinforcement (updates timestamp)
    void RecordReinforcement();

    /// Apply time-based decay to strength
    /// Uses exponential decay: s(t) = s(0) * exp(-decay_rate * t)
    /// @param elapsed_time Time elapsed since last update
    void ApplyDecay(Timestamp::Duration elapsed_time);

    // ========================================================================
    // Context Profile
    // ========================================================================

    /// Get context profile for this association
    const ContextVector& GetContextProfile() const;

    /// Set context profile
    /// @param context New context profile
    void SetContextProfile(const ContextVector& context);

    /// Update context profile with observed context
    /// Uses exponential moving average per dimension
    /// @param observed_context Observed context during activation
    /// @param learning_rate Learning rate [0,1], default 0.1
    void UpdateContextProfile(const ContextVector& observed_context, float learning_rate = 0.1f);

    /// Compute context-modulated strength
    /// @param current_context Current context vector
    /// @return Strength modulated by context similarity
    float GetContextualStrength(const ContextVector& current_context) const;

    // ========================================================================
    // Age and Statistics
    // ========================================================================

    /// Get age of this association
    Timestamp::Duration GetAge() const;

    /// Check if association is active (recently reinforced)
    /// @param max_idle_time Maximum allowed idle time
    /// @return True if last reinforcement within max_idle_time
    bool IsActive(Timestamp::Duration max_idle_time) const;

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Serialize to output stream
    void Serialize(std::ostream& out) const;

    /// Deserialize from input stream
    /// @return Unique pointer to deserialized edge
    static std::unique_ptr<AssociationEdge> Deserialize(std::istream& in);

    // ========================================================================
    // Utility
    // ========================================================================

    /// Get string representation
    std::string ToString() const;

    /// Estimate memory usage in bytes
    size_t EstimateMemoryUsage() const;

    // ========================================================================
    // Comparison Operators
    // ========================================================================

    /// Equality comparison (compares source, target, type)
    bool operator==(const AssociationEdge& other) const;

    /// Less-than comparison (sorts by strength, descending)
    bool operator<(const AssociationEdge& other) const;

private:
    // Core identification
    PatternID source_;
    PatternID target_;
    AssociationType type_{AssociationType::CATEGORICAL};

    // Strength (atomic for thread-safety)
    std::atomic<float> strength_{0.5f};

    // Statistics (atomic)
    std::atomic<uint32_t> co_occurrence_count_{0};
    std::atomic<float> temporal_correlation_{0.0f};

    // Decay parameters
    float decay_rate_{0.01f};  // Strength loss per second
    mutable std::atomic<uint64_t> last_reinforcement_{0};  // Timestamp in microseconds

    // Context (protected by mutex)
    mutable std::mutex context_mutex_;
    ContextVector context_profile_;

    // Creation timestamp
    Timestamp creation_time_;
};

} // namespace dpan
