// File: src/memory/tier_manager.hpp
//
// Tier Manager for Automatic Pattern Promotion/Demotion
//
// This module manages automatic transitions of patterns between memory tiers
// based on utility scores and capacity constraints. It implements intelligent
// promotion/demotion policies to keep frequently accessed patterns in fast
// storage while moving cold patterns to slower, cheaper storage.
//
// Tier Transition Rules:
//   Promotion (Archive → Cold → Warm → Active):
//     IF utility(p) > threshold_promote[current_tier] THEN
//       Move p to higher tier
//
//   Demotion (Active → Warm → Cold → Archive):
//     IF utility(p) < threshold_demote[current_tier] THEN
//       Move p to lower tier

#pragma once

#include "memory/memory_tier.hpp"
#include "memory/utility_calculator.hpp"
#include "core/types.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <optional>

namespace dpan {

/// Tier Manager for automatic pattern migration
class TierManager {
public:
    /// Configuration for tier management
    struct Config {
        // Tier capacities (number of patterns)
        size_t active_capacity{100000};      ///< Max patterns in active tier
        size_t warm_capacity{1000000};       ///< Max patterns in warm tier
        size_t cold_capacity{10000000};      ///< Max patterns in cold tier
        // Archive has unlimited capacity

        // Promotion thresholds (utility score required to move up)
        float warm_to_active_threshold{0.8f};
        float cold_to_warm_threshold{0.6f};
        float archive_to_cold_threshold{0.4f};

        // Demotion thresholds (below this utility, move down)
        float active_to_warm_threshold{0.7f};
        float warm_to_cold_threshold{0.4f};
        float cold_to_archive_threshold{0.2f};

        // Transition settings
        size_t transition_batch_size{1000};           ///< Patterns per batch transition
        float transition_interval_seconds{300.0f};    ///< 5 minutes between transitions

        /// Validate configuration
        bool IsValid() const;
    };

    /// Statistics for tier manager
    struct TierStats {
        size_t active_count{0};
        size_t warm_count{0};
        size_t cold_count{0};
        size_t archive_count{0};

        size_t promotions_count{0};
        size_t demotions_count{0};

        Timestamp last_transition;
    };

    /// Construct with default configuration
    TierManager();

    /// Construct with custom configuration
    explicit TierManager(const Config& config);

    /// Destructor - stops background thread
    ~TierManager();

    // Disable copy/move (manages resources)
    TierManager(const TierManager&) = delete;
    TierManager& operator=(const TierManager&) = delete;

    // ========================================================================
    // Initialization
    // ========================================================================

    /// Initialize with tier instances
    ///
    /// @param active Active tier (RAM-based)
    /// @param warm Warm tier (SSD-based)
    /// @param cold Cold tier (HDD-based)
    /// @param archive Archive tier (compressed)
    void Initialize(
        std::unique_ptr<IMemoryTier> active,
        std::unique_ptr<IMemoryTier> warm,
        std::unique_ptr<IMemoryTier> cold,
        std::unique_ptr<IMemoryTier> archive
    );

    /// Check if manager is initialized
    bool IsInitialized() const;

    // ========================================================================
    // Tier Transitions
    // ========================================================================

    /// Perform tier transitions based on utility scores
    ///
    /// @param utilities Map of pattern ID to utility score
    /// @return Number of patterns transitioned
    size_t PerformTierTransitions(const std::unordered_map<PatternID, float>& utilities);

    /// Manually promote a pattern to a higher tier
    ///
    /// @param id Pattern to promote
    /// @param target_tier Target tier (must be higher than current)
    /// @return true if promoted successfully
    bool PromotePattern(PatternID id, MemoryTier target_tier);

    /// Manually demote a pattern to a lower tier
    ///
    /// @param id Pattern to demote
    /// @param target_tier Target tier (must be lower than current)
    /// @return true if demoted successfully
    bool DemotePattern(PatternID id, MemoryTier target_tier);

    // ========================================================================
    // Pattern Location
    // ========================================================================

    /// Get current tier of a pattern
    ///
    /// @param id Pattern ID
    /// @return Current tier, or nullopt if not tracked
    std::optional<MemoryTier> GetPatternTier(PatternID id) const;

    /// Store pattern in specified tier
    ///
    /// @param pattern Pattern to store
    /// @param tier Tier to store in
    /// @return true if stored successfully
    bool StorePattern(const PatternNode& pattern, MemoryTier tier);

    /// Load pattern from its current tier
    ///
    /// @param id Pattern ID
    /// @return Pattern if found
    std::optional<PatternNode> LoadPattern(PatternID id);

    /// Remove pattern from all tiers
    ///
    /// @param id Pattern ID
    /// @return true if removed
    bool RemovePattern(PatternID id);

    // ========================================================================
    // Statistics and Monitoring
    // ========================================================================

    /// Get current tier statistics
    TierStats GetStats() const;

    /// Get promotion threshold for a tier
    float GetPromotionThreshold(MemoryTier tier) const;

    /// Get demotion threshold for a tier
    float GetDemotionThreshold(MemoryTier tier) const;

    // ========================================================================
    // Background Operations
    // ========================================================================

    /// Start background transition thread
    ///
    /// @param utility_calc Utility calculator for scoring patterns
    /// @param access_tracker Access tracker for statistics
    void StartBackgroundTransitions(
        const UtilityCalculator* utility_calc,
        const AccessTracker* access_tracker
    );

    /// Stop background transition thread
    void StopBackgroundTransitions();

    /// Check if background thread is running
    bool IsBackgroundRunning() const;

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

    /// Update configuration
    void SetConfig(const Config& config);

private:
    Config config_;

    // Tier instances
    std::unique_ptr<IMemoryTier> active_tier_;
    std::unique_ptr<IMemoryTier> warm_tier_;
    std::unique_ptr<IMemoryTier> cold_tier_;
    std::unique_ptr<IMemoryTier> archive_tier_;

    // Pattern location tracking
    std::unordered_map<PatternID, MemoryTier> pattern_locations_;
    mutable std::shared_mutex location_mutex_;

    // Transition statistics
    std::atomic<size_t> promotions_count_{0};
    std::atomic<size_t> demotions_count_{0};
    Timestamp last_transition_;

    // Background thread
    std::unique_ptr<std::thread> background_thread_;
    std::atomic<bool> running_{false};

    // Helper methods
    IMemoryTier* GetTier(MemoryTier tier);
    const IMemoryTier* GetTier(MemoryTier tier) const;

    bool MovePattern(PatternID id, MemoryTier from, MemoryTier to);

    std::vector<PatternID> SelectPatternsForPromotion(
        MemoryTier tier,
        const std::unordered_map<PatternID, float>& utilities
    );

    std::vector<PatternID> SelectPatternsForDemotion(
        MemoryTier tier,
        const std::unordered_map<PatternID, float>& utilities
    );

    void BackgroundTransitionLoop(
        const UtilityCalculator* utility_calc,
        const AccessTracker* access_tracker
    );

    void EnforceCapacityLimits(const std::unordered_map<PatternID, float>& utilities);

    size_t PromotePatternsFromTier(
        MemoryTier tier,
        const std::unordered_map<PatternID, float>& utilities
    );

    size_t DemotePatternsFromTier(
        MemoryTier tier,
        const std::unordered_map<PatternID, float>& utilities
    );
};

} // namespace dpan
