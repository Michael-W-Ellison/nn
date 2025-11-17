// File: src/memory/memory_manager.hpp
#pragma once

#include "memory/utility_calculator.hpp"
#include "memory/adaptive_thresholds.hpp"
#include "memory/utility_tracker.hpp"
#include "memory/tier_manager.hpp"
#include "memory/pattern_pruner.hpp"
#include "memory/association_pruner.hpp"
#include "memory/consolidator.hpp"
#include "memory/decay_functions.hpp"
#include "memory/interference.hpp"
#include "memory/sleep_consolidator.hpp"
#include "storage/pattern_database.hpp"
#include "association/association_matrix.hpp"
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>

namespace dpan {

/**
 * @brief Unified facade for all memory management operations
 *
 * MemoryManager integrates:
 * - Utility calculation and tracking
 * - Memory hierarchy (tier management)
 * - Pruning system (patterns and associations)
 * - Forgetting mechanisms (decay, interference, consolidation)
 * - Adaptive thresholding
 *
 * Provides a clean, high-level interface for memory management operations.
 */
class MemoryManager {
public:
    /**
     * @brief Configuration for all memory management subsystems
     */
    struct Config {
        // Subsystem configurations
        UtilityCalculator::Config utility_config;
        AdaptiveThresholdManager::Config threshold_config;
        TierManager::Config tier_config;
        PatternPruner::Config pattern_pruner_config;
        AssociationPruner::Config association_pruner_config;
        MemoryConsolidator::Config consolidator_config;
        SleepConsolidator::Config sleep_config;

        // Global settings
        bool enable_automatic_pruning{true};
        bool enable_tier_transitions{true};
        bool enable_consolidation{true};
        bool enable_forgetting{true};
        bool enable_sleep_consolidation{true};

        // Timing intervals
        std::chrono::seconds maintenance_interval{300};    // 5 minutes
        std::chrono::seconds pruning_interval{3600};       // 1 hour
        std::chrono::seconds transition_interval{60};      // 1 minute
        std::chrono::seconds consolidation_interval{86400}; // 24 hours

        // Decay function (default: exponential)
        std::string decay_function_type{"exponential"};
        float decay_constant{0.01f};

        /**
         * @brief Validate configuration
         */
        bool IsValid() const {
            return utility_config.IsValid() &&
                   threshold_config.IsValid() &&
                   tier_config.IsValid() &&
                   pattern_pruner_config.IsValid() &&
                   sleep_config.IsValid() &&
                   maintenance_interval.count() > 0 &&
                   pruning_interval.count() > 0 &&
                   transition_interval.count() > 0 &&
                   consolidation_interval.count() > 0;
        }
    };

    /**
     * @brief Comprehensive memory statistics
     */
    struct MemoryStats {
        // Pattern counts by tier
        size_t total_patterns{0};
        size_t active_patterns{0};
        size_t warm_patterns{0};
        size_t cold_patterns{0};
        size_t archive_patterns{0};

        // Association counts
        size_t total_associations{0};
        size_t strong_associations{0};
        size_t weak_associations{0};

        // Pruning statistics
        size_t patterns_pruned_total{0};
        size_t associations_pruned_total{0};
        size_t patterns_pruned_last_cycle{0};
        size_t associations_pruned_last_cycle{0};

        // Memory usage
        size_t total_memory_bytes{0};
        size_t active_tier_bytes{0};
        size_t warm_tier_bytes{0};
        size_t cold_tier_bytes{0};
        size_t archive_tier_bytes{0};

        // Thresholds and pressure
        float memory_pressure{0.0f};
        float current_utility_threshold{0.0f};
        float current_association_threshold{0.0f};

        // Forgetting statistics
        size_t patterns_with_decay{0};
        size_t patterns_with_interference{0};
        float average_interference{0.0f};

        // Sleep consolidation
        SleepConsolidator::ActivityState sleep_state{SleepConsolidator::ActivityState::ACTIVE};
        size_t consolidation_cycles{0};
        size_t patterns_strengthened{0};

        // Timestamps
        Timestamp last_maintenance_time;
        Timestamp last_pruning_time;
        Timestamp last_transition_time;
        Timestamp last_consolidation_time;
    };

    /**
     * @brief Construct with default configuration
     */
    MemoryManager();

    /**
     * @brief Construct with custom configuration
     */
    explicit MemoryManager(const Config& config);

    /**
     * @brief Initialize with pattern database and association matrix
     *
     * Must be called before using the MemoryManager.
     *
     * @param pattern_db Pattern database (not owned)
     * @param assoc_matrix Association matrix (not owned)
     * @param similarity_metric Similarity metric for interference calculation
     */
    void Initialize(
        PatternDatabase* pattern_db,
        AssociationMatrix* assoc_matrix,
        std::shared_ptr<SimilarityMetric> similarity_metric = nullptr
    );

    /**
     * @brief Check if initialized
     */
    bool IsInitialized() const { return is_initialized_; }

    /**
     * @brief Perform complete maintenance cycle
     *
     * Includes:
     * - Utility recalculation
     * - Threshold adjustment
     * - Tier transitions
     * - Pruning (if needed)
     * - Consolidation (if needed)
     * - Forgetting (decay and interference)
     */
    void PerformMaintenance();

    /**
     * @brief Perform pruning operations
     *
     * Removes low-utility patterns and weak associations.
     */
    void PerformPruning();

    /**
     * @brief Perform tier transitions
     *
     * Moves patterns between tiers based on utility and access patterns.
     */
    void PerformTierTransitions();

    /**
     * @brief Perform memory consolidation
     *
     * Merges similar patterns and compresses associations.
     */
    void PerformConsolidation();

    /**
     * @brief Apply forgetting mechanisms
     *
     * Applies decay and interference to reduce strength of unused patterns.
     */
    void ApplyForgetting();

    /**
     * @brief Record an operation for activity monitoring
     *
     * Call this for significant operations (pattern access, etc.)
     * to enable sleep consolidation.
     */
    void RecordOperation();

    /**
     * @brief Update sleep state and trigger consolidation if needed
     */
    void UpdateSleepState();

    /**
     * @brief Get memory statistics
     */
    MemoryStats GetStatistics() const;

    /**
     * @brief Get configuration
     */
    const Config& GetConfig() const { return config_; }

    /**
     * @brief Set configuration
     *
     * Note: Does not affect already initialized subsystems.
     * Call Initialize() again to apply new config.
     */
    void SetConfig(const Config& config);

    /**
     * @brief Get utility calculator
     */
    UtilityCalculator& GetUtilityCalculator() { return utility_calculator_; }
    const UtilityCalculator& GetUtilityCalculator() const { return utility_calculator_; }

    /**
     * @brief Get tier manager
     */
    TierManager& GetTierManager() { return *tier_manager_; }
    const TierManager& GetTierManager() const { return *tier_manager_; }

    /**
     * @brief Get sleep consolidator
     */
    SleepConsolidator* GetSleepConsolidator() { return sleep_consolidator_.get(); }
    const SleepConsolidator* GetSleepConsolidator() const { return sleep_consolidator_.get(); }

private:
    Config config_;

    // Initialization state
    bool is_initialized_{false};

    // External dependencies (not owned)
    PatternDatabase* pattern_db_{nullptr};
    AssociationMatrix* assoc_matrix_{nullptr};

    // Core subsystems
    UtilityCalculator utility_calculator_;
    std::unique_ptr<UtilityTracker> utility_tracker_;
    AdaptiveThresholdManager threshold_manager_;
    std::unique_ptr<TierManager> tier_manager_;
    std::unique_ptr<PatternPruner> pattern_pruner_;
    AssociationPruner association_pruner_;
    MemoryConsolidator memory_consolidator_;
    std::unique_ptr<SleepConsolidator> sleep_consolidator_;

    // Forgetting mechanisms
    std::unique_ptr<IDecayFunction> decay_function_;
    InterferenceCalculator interference_calculator_;

    // Statistics
    mutable std::mutex stats_mutex_;
    MemoryStats cached_stats_;
    Timestamp last_stats_update_;

    // Timing
    Timestamp last_maintenance_;
    Timestamp last_pruning_;
    Timestamp last_transition_;
    Timestamp last_consolidation_;

    // Helper methods
    void UpdateCachedStatistics();
    void InitializeDecayFunction();
    void ApplyDecayToPatterns();
    void ApplyInterferenceToPatterns();
};

} // namespace dpan
