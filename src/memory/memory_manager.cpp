// File: src/memory/memory_manager.cpp
#include "memory/memory_manager.hpp"
#include <algorithm>
#include <stdexcept>

namespace dpan {

MemoryManager::MemoryManager()
    : config_(), last_maintenance_(Timestamp::Now()),
      last_pruning_(Timestamp::Now()), last_transition_(Timestamp::Now()),
      last_consolidation_(Timestamp::Now()), last_stats_update_(Timestamp::Now()) {
}

MemoryManager::MemoryManager(const Config& config)
    : config_(config), last_maintenance_(Timestamp::Now()),
      last_pruning_(Timestamp::Now()), last_transition_(Timestamp::Now()),
      last_consolidation_(Timestamp::Now()), last_stats_update_(Timestamp::Now()) {
    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid MemoryManager configuration");
    }
}

void MemoryManager::Initialize(
    PatternDatabase* pattern_db,
    AssociationMatrix* assoc_matrix,
    std::shared_ptr<SimilarityMetric> similarity_metric
) {
    if (!pattern_db || !assoc_matrix) {
        throw std::invalid_argument("PatternDatabase and AssociationMatrix cannot be null");
    }

    pattern_db_ = pattern_db;
    assoc_matrix_ = assoc_matrix;

    // Initialize utility calculator with config
    utility_calculator_ = UtilityCalculator(config_.utility_config);

    // Initialize threshold manager
    threshold_manager_ = AdaptiveThresholdManager(config_.threshold_config);

    // Initialize tier manager
    tier_manager_ = std::make_unique<TierManager>(config_.tier_config);

    // Initialize pattern pruner
    pattern_pruner_ = std::make_unique<PatternPruner>(config_.pattern_pruner_config);

    // Initialize association pruner
    association_pruner_ = AssociationPruner(config_.association_pruner_config);

    // Initialize memory consolidator
    memory_consolidator_ = MemoryConsolidator(config_.consolidator_config);

    // Initialize sleep consolidator
    sleep_consolidator_ = std::make_unique<SleepConsolidator>(config_.sleep_config);

    // Initialize decay function
    InitializeDecayFunction();

    // Initialize interference calculator with similarity metric
    if (similarity_metric) {
        interference_calculator_ = InterferenceCalculator(
            InterferenceCalculator::Config{},
            similarity_metric
        );
    }

    is_initialized_ = true;
}

void MemoryManager::SetConfig(const Config& config) {
    if (!config.IsValid()) {
        throw std::invalid_argument("Invalid MemoryManager configuration");
    }
    config_ = config;
}

void MemoryManager::PerformMaintenance() {
    if (!is_initialized_) {
        throw std::runtime_error("MemoryManager not initialized");
    }

    Timestamp now = Timestamp::Now();

    // Update sleep state
    if (config_.enable_sleep_consolidation) {
        UpdateSleepState();
    }

    // Check if pruning needed
    auto time_since_pruning = now - last_pruning_;
    if (config_.enable_automatic_pruning &&
        time_since_pruning >= config_.pruning_interval) {
        PerformPruning();
        last_pruning_ = now;
    }

    // Check if tier transitions needed
    auto time_since_transition = now - last_transition_;
    if (config_.enable_tier_transitions &&
        time_since_transition >= config_.transition_interval) {
        PerformTierTransitions();
        last_transition_ = now;
    }

    // Check if consolidation needed
    auto time_since_consolidation = now - last_consolidation_;
    if (config_.enable_consolidation &&
        time_since_consolidation >= config_.consolidation_interval) {
        PerformConsolidation();
        last_consolidation_ = now;
    }

    // Apply forgetting mechanisms
    if (config_.enable_forgetting) {
        ApplyForgetting();
    }

    last_maintenance_ = now;

    // Update cached statistics
    UpdateCachedStatistics();
}

void MemoryManager::PerformPruning() {
    if (!is_initialized_) {
        throw std::runtime_error("MemoryManager not initialized");
    }

    // TODO: Get utilities from UtilityTracker once AccessTracker is integrated
    std::unordered_map<PatternID, float> utilities;

    // Get current threshold
    auto threshold_stats = threshold_manager_.GetStats();
    float threshold = threshold_stats.current_threshold;

    // Perform pattern pruning
    if (config_.enable_automatic_pruning && pattern_pruner_) {
        auto prune_result = pattern_pruner_->PrunePatterns(
            *pattern_db_,
            *assoc_matrix_,
            utilities
        );

        // Update statistics
        std::lock_guard<std::mutex> lock(stats_mutex_);
        cached_stats_.patterns_pruned_last_cycle = prune_result.pruned_patterns.size();
        cached_stats_.patterns_pruned_total += prune_result.pruned_patterns.size();
    }

    // Perform association pruning
    if (config_.enable_automatic_pruning) {
        auto assoc_pruned_count = association_pruner_.PruneWeakAssociations(*assoc_matrix_);

        // Update statistics
        std::lock_guard<std::mutex> lock(stats_mutex_);
        cached_stats_.associations_pruned_last_cycle = assoc_pruned_count;
        cached_stats_.associations_pruned_total += assoc_pruned_count;
    }
}

void MemoryManager::PerformTierTransitions() {
    if (!is_initialized_ || !tier_manager_) {
        return;
    }

    // TODO: Get utilities from UtilityTracker once AccessTracker is integrated
    std::unordered_map<PatternID, float> utilities;

    // Tier manager would handle transitions based on utilities
    // (This is a simplified version - actual implementation would be more complex)

    // Update threshold based on current memory usage
    threshold_manager_.UpdateThreshold(
        pattern_db_->Count() * 1024,  // Estimate memory usage
        pattern_db_->Count()
    );
}

void MemoryManager::PerformConsolidation() {
    if (!is_initialized_) {
        return;
    }

    // Perform memory consolidation
    // Note: This requires pattern database and similarity metric
    // Actual consolidation is complex and would integrate with tier management

    std::lock_guard<std::mutex> lock(stats_mutex_);
    cached_stats_.last_consolidation_time = Timestamp::Now();
}

void MemoryManager::ApplyForgetting() {
    if (!is_initialized_ || !decay_function_) {
        return;
    }

    // Apply decay to patterns
    ApplyDecayToPatterns();

    // Apply interference to patterns
    ApplyInterferenceToPatterns();
}

void MemoryManager::RecordOperation() {
    if (config_.enable_sleep_consolidation && sleep_consolidator_) {
        sleep_consolidator_->RecordOperation();
    }
}

void MemoryManager::UpdateSleepState() {
    if (!is_initialized_ || !sleep_consolidator_) {
        return;
    }

    // Update sleep consolidator state
    bool state_changed = sleep_consolidator_->UpdateActivityState();

    // Check if consolidation should be triggered
    if (sleep_consolidator_->ShouldTriggerConsolidation()) {
        auto result = sleep_consolidator_->TriggerConsolidation();

        // Update statistics
        std::lock_guard<std::mutex> lock(stats_mutex_);
        cached_stats_.consolidation_cycles++;
        cached_stats_.patterns_strengthened += result.patterns_strengthened;
    }
}

MemoryManager::MemoryStats MemoryManager::GetStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    MemoryStats stats = cached_stats_;

    // Update real-time statistics
    if (is_initialized_) {
        stats.total_patterns = pattern_db_ ? pattern_db_->Count() : 0;
        stats.total_associations = assoc_matrix_ ? assoc_matrix_->GetAssociationCount() : 0;
        auto threshold_stats = threshold_manager_.GetStats();
        stats.current_utility_threshold = threshold_stats.current_threshold;
        stats.memory_pressure = threshold_stats.memory_pressure;
        if (sleep_consolidator_) {
            stats.sleep_state = sleep_consolidator_->GetActivityState();
        }
    }

    stats.last_maintenance_time = last_maintenance_;
    stats.last_pruning_time = last_pruning_;
    stats.last_transition_time = last_transition_;
    stats.last_consolidation_time = last_consolidation_;

    return stats;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void MemoryManager::UpdateCachedStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (!is_initialized_) {
        return;
    }

    // Update pattern counts
    cached_stats_.total_patterns = pattern_db_ ? pattern_db_->Count() : 0;
    cached_stats_.total_associations = assoc_matrix_ ? assoc_matrix_->GetAssociationCount() : 0;

    // Update memory usage (simplified - actual would query tiers)
    cached_stats_.total_memory_bytes = cached_stats_.total_patterns * 1024; // Estimate

    // Update thresholds
    auto threshold_stats = threshold_manager_.GetStats();
    cached_stats_.current_utility_threshold = threshold_stats.current_threshold;
    cached_stats_.memory_pressure = threshold_stats.memory_pressure;

    // Update sleep state
    if (sleep_consolidator_) {
        cached_stats_.sleep_state = sleep_consolidator_->GetActivityState();
        auto sleep_stats = sleep_consolidator_->GetStatistics();
        cached_stats_.consolidation_cycles = sleep_stats.total_consolidation_cycles;
        cached_stats_.patterns_strengthened = sleep_stats.total_patterns_strengthened;
    }

    last_stats_update_ = Timestamp::Now();
}

void MemoryManager::InitializeDecayFunction() {
    // Create decay function based on config
    if (config_.decay_function_type == "exponential") {
        decay_function_ = std::make_unique<ExponentialDecay>(config_.decay_constant);
    } else if (config_.decay_function_type == "powerlaw") {
        decay_function_ = std::make_unique<PowerLawDecay>();
    } else if (config_.decay_function_type == "step") {
        decay_function_ = std::make_unique<StepDecay>();
    } else {
        // Default to exponential
        decay_function_ = std::make_unique<ExponentialDecay>(config_.decay_constant);
    }
}

void MemoryManager::ApplyDecayToPatterns() {
    if (!pattern_db_ || !decay_function_) {
        return;
    }

    // Note: In a full implementation, this would:
    // 1. Iterate through patterns
    // 2. Calculate time since last access
    // 3. Apply decay to confidence/utility
    // 4. Update pattern database

    // Simplified version for integration
    Timestamp now = Timestamp::Now();
    size_t patterns_with_decay = 0;

    // This would be implemented with actual pattern iteration
    // For now, just track that decay is being applied

    std::lock_guard<std::mutex> lock(stats_mutex_);
    cached_stats_.patterns_with_decay = patterns_with_decay;
}

void MemoryManager::ApplyInterferenceToPatterns() {
    if (!pattern_db_ || !assoc_matrix_) {
        return;
    }

    // Note: In a full implementation, this would:
    // 1. Identify similar patterns
    // 2. Calculate interference between them
    // 3. Reduce strength of interfered patterns
    // 4. Update pattern database

    // Simplified version for integration
    size_t patterns_with_interference = 0;
    float total_interference = 0.0f;

    // This would be implemented with actual interference calculation

    std::lock_guard<std::mutex> lock(stats_mutex_);
    cached_stats_.patterns_with_interference = patterns_with_interference;
    if (patterns_with_interference > 0) {
        cached_stats_.average_interference = total_interference / patterns_with_interference;
    }
}

} // namespace dpan
