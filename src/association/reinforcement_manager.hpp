// File: src/association/reinforcement_manager.hpp
#pragma once

#include "association/association_edge.hpp"
#include "association/association_matrix.hpp"
#include "core/types.hpp"
#include <vector>
#include <utility>

namespace dpan {

/// ReinforcementManager: Manages association strength through reinforcement learning
///
/// Implements Hebbian learning principles to strengthen associations that successfully
/// predict pattern activations, and weaken those that fail. Applies time-based decay
/// to unused associations.
///
/// Thread-safety: Methods are thread-safe when operating on different edges.
/// External synchronization required for batch operations on shared matrices.
class ReinforcementManager {
public:
    /// Configuration for reinforcement learning
    struct Config {
        Config() = default;

        /// Learning rate for reinforcement (η)
        float learning_rate{0.1f};

        /// Exponential decay rate for time-based weakening
        float decay_rate{0.01f};

        /// Minimum allowed strength (prevents complete deletion)
        float min_strength{0.1f};

        /// Maximum allowed strength (prevents unbounded growth)
        float max_strength{1.0f};

        /// Strength threshold for pruning weak associations
        float prune_threshold{0.05f};
    };

    // ========================================================================
    // Construction
    // ========================================================================

    ReinforcementManager();
    explicit ReinforcementManager(const Config& config);
    ~ReinforcementManager() = default;

    // ========================================================================
    // Single Edge Reinforcement
    // ========================================================================

    /// Reinforce (strengthen) an association
    /// Uses Hebbian learning: Δs = η × (1 - s) × reward
    /// @param edge Association edge to strengthen
    /// @param reward Reward value [0,1] indicating prediction accuracy
    void Reinforce(AssociationEdge& edge, float reward = 1.0f);

    /// Weaken (punish) an association
    /// Reduces strength: Δs = -η × s × penalty
    /// @param edge Association edge to weaken
    /// @param penalty Penalty value [0,1] indicating prediction error
    void Weaken(AssociationEdge& edge, float penalty = 1.0f);

    /// Apply time-based exponential decay
    /// s(t) = s(0) × exp(-d × t)
    /// @param edge Association edge to decay
    /// @param elapsed Time since last reinforcement
    void ApplyDecay(AssociationEdge& edge, Timestamp::Duration elapsed);

    /// Update strength directly (clamped to [min, max])
    /// @param edge Association edge to update
    /// @param new_strength New strength value
    void SetStrength(AssociationEdge& edge, float new_strength);

    /// Clamp strength to configured bounds
    /// @param strength Strength value to clamp
    /// @return Clamped strength in [min_strength, max_strength]
    float ClampStrength(float strength) const;

    // ========================================================================
    // Batch Reinforcement
    // ========================================================================

    /// Reinforce multiple associations in a matrix
    /// @param matrix Association matrix containing edges
    /// @param pairs Vector of (source, target) pattern pairs to reinforce
    /// @param reward Reward value for all pairs
    void ReinforceBatch(
        AssociationMatrix& matrix,
        const std::vector<std::pair<PatternID, PatternID>>& pairs,
        float reward = 1.0f
    );

    /// Weaken multiple associations in a matrix
    /// @param matrix Association matrix containing edges
    /// @param pairs Vector of (source, target) pattern pairs to weaken
    /// @param penalty Penalty value for all pairs
    void WeakenBatch(
        AssociationMatrix& matrix,
        const std::vector<std::pair<PatternID, PatternID>>& pairs,
        float penalty = 1.0f
    );

    /// Apply decay to all associations in a matrix
    /// @param matrix Association matrix
    /// @param elapsed Time since last decay application
    /// @param auto_prune Automatically remove edges below prune_threshold
    void ApplyDecayAll(
        AssociationMatrix& matrix,
        Timestamp::Duration elapsed,
        bool auto_prune = false
    );

    // ========================================================================
    // Prediction-Based Reinforcement
    // ========================================================================

    /// Reinforce based on prediction result
    /// @param edge Association edge that made the prediction
    /// @param predicted True if prediction was made
    /// @param actual_occurred True if predicted event actually occurred
    void ReinforcePrediction(
        AssociationEdge& edge,
        bool predicted,
        bool actual_occurred
    );

    /// Batch reinforcement based on prediction results
    /// @param matrix Association matrix
    /// @param predictions Vector of (source, target, predicted, occurred) tuples
    void ReinforcePredictionsBatch(
        AssociationMatrix& matrix,
        const std::vector<std::tuple<PatternID, PatternID, bool, bool>>& predictions
    );

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Check if an edge should be pruned
    /// @param edge Association edge to check
    /// @return True if strength < prune_threshold
    bool ShouldPrune(const AssociationEdge& edge) const;

    /// Get number of edges that would be pruned in a matrix
    /// @param matrix Association matrix to analyze
    /// @return Count of edges below prune_threshold
    size_t CountPrunableEdges(const AssociationMatrix& matrix) const;

    /// Prune weak associations from matrix
    /// @param matrix Association matrix
    /// @return Number of edges removed
    size_t PruneWeakAssociations(AssociationMatrix& matrix);

    // ========================================================================
    // Statistics
    // ========================================================================

    struct ReinforcementStats {
        uint64_t reinforcements{0};          // Total reinforcement operations
        uint64_t weakenings{0};              // Total weakening operations
        uint64_t decays{0};                  // Total decay operations
        uint64_t pruned{0};                  // Total edges pruned
        float average_strength_delta{0.0f};  // Average strength change per operation
    };

    /// Get reinforcement statistics
    const ReinforcementStats& GetStats() const { return stats_; }

    /// Reset statistics
    void ResetStats();

    /// Get configuration
    const Config& GetConfig() const { return config_; }

    /// Set configuration (does not affect existing edges)
    void SetConfig(const Config& config) { config_ = config; }

private:
    Config config_;
    ReinforcementStats stats_;

    // Helper methods

    /// Compute reinforcement delta using Hebbian learning
    /// Δs = η × (1 - s) × reward
    float ComputeReinforcementDelta(float current_strength, float reward) const;

    /// Compute weakening delta
    /// Δs = -η × s × penalty
    float ComputeWeakeningDelta(float current_strength, float penalty) const;

    /// Compute decay factor for time elapsed
    /// factor = exp(-d × t)
    float ComputeDecayFactor(Timestamp::Duration elapsed) const;

    /// Update statistics
    void RecordReinforcement(float delta);
    void RecordWeakening(float delta);
    void RecordDecay();
    void RecordPruned(size_t count);
};

} // namespace dpan
