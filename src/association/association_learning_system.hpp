// File: src/association/association_learning_system.hpp
#pragma once

#include "association/association_matrix.hpp"
#include "association/co_occurrence_tracker.hpp"
#include "association/formation_rules.hpp"
#include "association/reinforcement_manager.hpp"
#include "association/competitive_learner.hpp"
#include "association/strength_normalizer.hpp"
#include "storage/pattern_database.hpp"
#include "core/types.hpp"
#include <deque>
#include <memory>
#include <vector>
#include <mutex>

namespace dpan {

/// AssociationLearningSystem: Unified system for learning and managing associations
///
/// This system integrates all association learning components into a cohesive whole:
/// - Co-occurrence tracking: Detects patterns that occur together
/// - Association formation: Creates new associations based on statistics
/// - Reinforcement learning: Strengthens/weakens based on prediction accuracy
/// - Competitive learning: Winner-take-all competition among associations
/// - Strength normalization: Prevents strength inflation
/// - Decay: Time-based weakening of unused associations
/// - Activation propagation: Spreads activation through the network
///
/// Thread-safety: All public methods are thread-safe
class AssociationLearningSystem {
public:
    // ========================================================================
    // Configuration
    // ========================================================================

    struct Config {
        /// Co-occurrence tracking configuration
        CoOccurrenceTracker::Config co_occurrence;

        /// Association formation rules configuration
        AssociationFormationRules::Config formation;

        /// Reinforcement learning configuration
        ReinforcementManager::Config reinforcement;

        /// Competitive learning configuration
        CompetitiveLearner::Config competition;

        /// Strength normalization configuration
        StrengthNormalizer::Config normalization;

        /// Maximum number of associations to maintain
        size_t association_capacity{1000000};

        /// Activation history window (for temporal learning)
        Timestamp::Duration activation_window{std::chrono::seconds(10)};

        /// Maximum activation history size
        size_t max_activation_history{10000};

        /// Auto-apply decay interval (0 = disabled)
        Timestamp::Duration auto_decay_interval{std::chrono::hours(1)};

        /// Auto-apply competition interval (0 = disabled)
        Timestamp::Duration auto_competition_interval{std::chrono::minutes(30)};

        /// Auto-apply normalization interval (0 = disabled)
        Timestamp::Duration auto_normalization_interval{std::chrono::minutes(30)};

        /// Minimum strength for pruning weak associations
        float prune_threshold{0.05f};

        /// Enable automatic maintenance (decay, competition, normalization, pruning)
        bool enable_auto_maintenance{true};
    };

    // ========================================================================
    // Construction & Initialization
    // ========================================================================

    /// Construct with default configuration
    AssociationLearningSystem();

    /// Construct with custom configuration
    explicit AssociationLearningSystem(const Config& config);

    /// Destructor
    ~AssociationLearningSystem();

    // Delete copy operations (use shared_ptr if sharing needed)
    AssociationLearningSystem(const AssociationLearningSystem&) = delete;
    AssociationLearningSystem& operator=(const AssociationLearningSystem&) = delete;

    // ========================================================================
    // Pattern Activation Recording
    // ========================================================================

    /// Record single pattern activation
    /// Updates co-occurrence statistics and activation history
    /// @param pattern Pattern that was activated
    /// @param context Optional context vector for context-sensitive learning
    void RecordPatternActivation(
        PatternID pattern,
        const ContextVector& context = ContextVector()
    );

    /// Record multiple pattern activations (batch)
    /// More efficient than individual calls
    /// @param patterns Patterns that were activated
    /// @param context Optional context vector
    void RecordPatternActivations(
        const std::vector<PatternID>& patterns,
        const ContextVector& context = ContextVector()
    );

    // ========================================================================
    // Association Formation
    // ========================================================================

    /// Trigger association formation based on co-occurrence statistics
    /// Analyzes tracked co-occurrences and creates new associations
    /// @param pattern_db Pattern database for accessing pattern details
    /// @return Number of new associations formed
    size_t FormNewAssociations(const PatternDatabase& pattern_db);

    /// Form associations for specific pattern
    /// @param pattern Source pattern
    /// @param pattern_db Pattern database
    /// @return Number of new associations formed
    size_t FormAssociationsForPattern(
        PatternID pattern,
        const PatternDatabase& pattern_db
    );

    // ========================================================================
    // Reinforcement Learning
    // ========================================================================

    /// Apply reinforcement based on prediction accuracy
    /// @param predicted Pattern that was predicted
    /// @param actual Pattern that actually occurred
    /// @param correct Whether the prediction was correct
    void Reinforce(PatternID predicted, PatternID actual, bool correct);

    /// Batch reinforcement
    /// @param outcomes Vector of (predicted, actual, correct) tuples
    void ReinforceBatch(
        const std::vector<std::tuple<PatternID, PatternID, bool>>& outcomes
    );

    // ========================================================================
    // Maintenance Operations
    // ========================================================================

    /// Apply time-based decay to all associations
    /// @param elapsed Time elapsed since last decay application
    void ApplyDecay(Timestamp::Duration elapsed);

    /// Apply competitive learning to all patterns
    /// Winner-take-all: strongest associations boosted, weaker suppressed
    /// @return Number of patterns that had competition applied
    size_t ApplyCompetition();

    /// Apply strength normalization to all patterns
    /// Ensures outgoing strengths sum to 1.0
    /// @return Number of patterns normalized
    size_t ApplyNormalization();

    /// Prune weak associations below threshold
    /// @param min_strength Minimum strength threshold (default from config)
    /// @return Number of associations pruned
    size_t PruneWeakAssociations(float min_strength = 0.0f);

    /// Compact internal data structures
    /// Reclaims memory from deleted associations
    void Compact();

    /// Perform all maintenance operations
    /// Applies decay, competition, normalization, and pruning
    /// @return Statistics about maintenance performed
    struct MaintenanceStats {
        size_t competitions_applied;
        size_t normalizations_applied;
        size_t associations_pruned;
        Timestamp::Duration decay_applied;
    };
    MaintenanceStats PerformMaintenance();

    // ========================================================================
    // Query & Prediction
    // ========================================================================

    /// Get association matrix (read-only access)
    const AssociationMatrix& GetAssociationMatrix() const;

    /// Get all associations for a pattern
    /// @param pattern Source pattern
    /// @param outgoing If true, get outgoing; if false, get incoming
    /// @return Vector of association edges
    std::vector<const AssociationEdge*> GetAssociations(
        PatternID pattern,
        bool outgoing = true
    ) const;

    /// Predict next patterns based on current pattern
    /// Uses association strengths and activation propagation
    /// @param pattern Current pattern
    /// @param k Number of predictions to return
    /// @param context Optional context for context-sensitive prediction
    /// @return Vector of predicted patterns (sorted by strength/activation)
    std::vector<PatternID> Predict(
        PatternID pattern,
        size_t k = 5,
        const ContextVector* context = nullptr
    ) const;

    /// Predict with confidence scores
    /// @param pattern Current pattern
    /// @param k Number of predictions
    /// @param context Optional context
    /// @return Vector of (pattern, confidence) pairs
    std::vector<std::pair<PatternID, float>> PredictWithConfidence(
        PatternID pattern,
        size_t k = 5,
        const ContextVector* context = nullptr
    ) const;

    /// Propagate activation through association network
    /// @param source Starting pattern
    /// @param initial_activation Initial activation level
    /// @param max_hops Maximum propagation distance
    /// @param min_activation Minimum activation to continue
    /// @param context Optional context
    /// @return Activation results for all reached patterns
    std::vector<AssociationMatrix::ActivationResult> PropagateActivation(
        PatternID source,
        float initial_activation = 1.0f,
        size_t max_hops = 3,
        float min_activation = 0.01f,
        const ContextVector* context = nullptr
    ) const;

    // ========================================================================
    // Statistics & Monitoring
    // ========================================================================

    struct Statistics {
        // Association counts
        size_t total_associations;
        size_t active_associations;  // Non-zero strength

        // Strength statistics
        float average_strength;
        float min_strength;
        float max_strength;

        // Pattern statistics
        size_t patterns_with_associations;
        float average_associations_per_pattern;

        // Co-occurrence statistics
        size_t total_co_occurrences;
        size_t activation_history_size;

        // Maintenance statistics
        Timestamp last_decay;
        Timestamp last_competition;
        Timestamp last_normalization;
        Timestamp last_pruning;

        // Performance metrics
        size_t formations_count;
        size_t reinforcements_count;
        size_t predictions_count;
    };

    /// Get comprehensive system statistics
    Statistics GetStatistics() const;

    /// Get association count
    size_t GetAssociationCount() const;

    /// Get average association strength
    float GetAverageStrength() const;

    /// Print statistics to stream
    void PrintStatistics(std::ostream& out) const;

    // ========================================================================
    // Configuration Management
    // ========================================================================

    /// Update configuration (thread-safe)
    void SetConfig(const Config& config);

    /// Get current configuration (copy)
    Config GetConfig() const;

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Save system state to file
    /// @param filepath Path to save file
    /// @return True if successful
    bool Save(const std::string& filepath) const;

    /// Load system state from file
    /// @param filepath Path to load file
    /// @return True if successful
    bool Load(const std::string& filepath);

private:
    // Configuration
    Config config_;
    mutable std::mutex config_mutex_;

    // Core components
    AssociationMatrix matrix_;
    CoOccurrenceTracker tracker_;
    AssociationFormationRules formation_rules_;
    ReinforcementManager reinforcement_mgr_;

    // Activation history for temporal learning
    std::deque<std::pair<Timestamp, PatternID>> activation_history_;
    mutable std::mutex history_mutex_;

    // Statistics tracking
    mutable Statistics stats_;
    mutable std::mutex stats_mutex_;

    // Maintenance timestamps
    Timestamp last_decay_;
    Timestamp last_competition_;
    Timestamp last_normalization_;
    Timestamp last_pruning_;

    // Helper methods
    void UpdateActivationHistory(PatternID pattern, Timestamp timestamp);
    void TrimActivationHistory();
    void UpdateStatistics() const;
    void CheckAutoMaintenance();
};

} // namespace dpan
