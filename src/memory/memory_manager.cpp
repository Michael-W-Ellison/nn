// File: src/memory/memory_manager.cpp
#include "memory/memory_manager.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace dpan {

MemoryManager::MemoryManager()
    : config_(), last_stats_update_(Timestamp::Now()), last_maintenance_(Timestamp::Now()),
      last_pruning_(Timestamp::Now()), last_transition_(Timestamp::Now()),
      last_consolidation_(Timestamp::Now()), last_forgetting_(Timestamp::Now()) {
}

MemoryManager::MemoryManager(const Config& config)
    : config_(config), last_stats_update_(Timestamp::Now()), last_maintenance_(Timestamp::Now()),
      last_pruning_(Timestamp::Now()), last_transition_(Timestamp::Now()),
      last_consolidation_(Timestamp::Now()), last_forgetting_(Timestamp::Now()) {
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
    similarity_metric_ = similarity_metric;

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
    sleep_consolidator_->SetConsolidationWork(
        [this](SleepConsolidator::ConsolidationCycleResult& result) {
            RunSleepConsolidation(result);
        });

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

void MemoryManager::RecordPatternAccess(PatternID pattern, Timestamp timestamp) {
    access_tracker_.RecordPatternAccess(pattern, timestamp);
    RecordOperation();
}

void MemoryManager::RecordAssociationAccess(PatternID source, PatternID target, Timestamp timestamp) {
    access_tracker_.RecordAssociationAccess(source, target, timestamp);
    RecordOperation();
}

std::unordered_map<PatternID, float> MemoryManager::ComputePatternUtilities() const {
    std::unordered_map<PatternID, float> utilities;
    if (!is_initialized_) {
        return utilities;
    }

    QueryOptions all = QueryOptions::Unlimited();
    auto ids = pattern_db_->FindAll(all);
    utilities.reserve(ids.size());

    for (const auto& id : ids) {
        auto node = pattern_db_->Retrieve(id);
        if (!node) {
            continue;  // Removed concurrently
        }

        // Associations involving the pattern (copies: edges are not copyable)
        std::vector<AssociationEdge> associations;
        for (const auto* edge : assoc_matrix_->GetOutgoingAssociations(id)) {
            associations.push_back(std::move(*edge->Clone()));
        }
        for (const auto* edge : assoc_matrix_->GetIncomingAssociations(id)) {
            associations.push_back(std::move(*edge->Clone()));
        }

        auto tracked = access_tracker_.GetPatternStatsSnapshot(id);
        AccessStats stats = tracked ? *tracked : AccessStatsFromNode(*node);
        utilities[id] = utility_calculator_.CalculatePatternUtility(*node, stats, associations);
    }

    return utilities;
}

void MemoryManager::PerformPruning() {
    if (!is_initialized_) {
        throw std::runtime_error("MemoryManager not initialized");
    }

    auto utilities = ComputePatternUtilities();

    // Prune below the adaptive threshold (it tracks memory pressure)
    auto threshold_stats = threshold_manager_.GetStats();
    float threshold = threshold_stats.current_threshold;

    // Perform pattern pruning
    if (config_.enable_automatic_pruning && pattern_pruner_) {
        auto pruner_config = pattern_pruner_->GetConfig();
        pruner_config.utility_threshold = threshold;
        pattern_pruner_->SetConfig(pruner_config);

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

    // Update threshold based on current memory usage
    size_t pattern_count = pattern_db_->Count();
    size_t memory_bytes = pattern_db_->GetStats().memory_usage_bytes;
    if (memory_bytes == 0) {
        memory_bytes = pattern_count * 1024;  // Backend does not report memory: estimate
    }
    threshold_manager_.UpdateThreshold(memory_bytes, pattern_count);

    // Move patterns between tiers once the tiers have been set up
    if (tier_manager_->IsInitialized()) {
        tier_manager_->RunTransitionCycle(utility_calculator_, access_tracker_);

        auto tier_stats = tier_manager_->GetStats();
        std::lock_guard<std::mutex> lock(stats_mutex_);
        cached_stats_.active_patterns = tier_stats.active_count;
        cached_stats_.warm_patterns = tier_stats.warm_count;
        cached_stats_.cold_patterns = tier_stats.cold_count;
        cached_stats_.archive_patterns = tier_stats.archive_count;
    }
}

void MemoryManager::RunSleepConsolidation(SleepConsolidator::ConsolidationCycleResult& result) {
    // 1. Strengthen the most useful patterns (raise their confidence)
    auto utilities = ComputePatternUtilities();
    auto to_strengthen = sleep_consolidator_->IdentifyPatternsToStrengthen(utilities);

    float total_change = 0.0f;
    for (const auto& info : to_strengthen) {
        auto node = pattern_db_->Retrieve(info.pattern_id);
        if (!node) {
            continue;
        }
        float confidence = node->GetConfidenceScore();
        float boosted = std::min(1.0f, confidence + info.boost_amount);
        if (boosted > confidence) {
            node->SetConfidenceScore(boosted);
            if (pattern_db_->Update(*node)) {
                result.patterns_strengthened++;
                total_change += info.boost_amount;
            }
        }
    }
    if (result.patterns_strengthened > 0) {
        result.average_utility_change = total_change / result.patterns_strengthened;
    }

    // 2. Prune more aggressively than during waking maintenance
    if (pattern_pruner_) {
        auto original = pattern_pruner_->GetConfig();
        auto aggressive = original;
        aggressive.utility_threshold = std::min(
            1.0f, threshold_manager_.GetStats().current_threshold *
                  config_.sleep_config.sleep_pruning_multiplier);
        pattern_pruner_->SetConfig(aggressive);

        auto prune_result = pattern_pruner_->PrunePatterns(*pattern_db_, *assoc_matrix_,
                                                           ComputePatternUtilities());
        pattern_pruner_->SetConfig(original);

        result.patterns_pruned = prune_result.pruned_patterns.size() +
                                 prune_result.merged_patterns.size();
        result.memory_freed_bytes += prune_result.bytes_freed;
    }

    // 3. Merge similar patterns and compress association paths
    if (similarity_metric_) {
        size_t edges_before = assoc_matrix_->GetAssociationCount();
        auto consolidation = memory_consolidator_.Consolidate(
            *pattern_db_, *assoc_matrix_, *similarity_metric_,
            access_tracker_.GetAssociationAccessCounts());
        size_t edges_after = assoc_matrix_->GetAssociationCount();
        result.associations_reorganized = edges_after > edges_before
            ? edges_after - edges_before
            : edges_before - edges_after;
        result.memory_freed_bytes += consolidation.memory_freed_bytes;
    }
}

void MemoryManager::PerformConsolidation() {
    if (!is_initialized_) {
        return;
    }

    // Merging and clustering compare patterns, so they need a similarity metric
    if (similarity_metric_) {
        memory_consolidator_.Consolidate(*pattern_db_, *assoc_matrix_, *similarity_metric_,
                                         access_tracker_.GetAssociationAccessCounts());
    }

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

    last_forgetting_ = Timestamp::Now();
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

    // Decay each pattern's confidence for the time it went unused since the
    // previous forgetting pass (so repeated passes do not compound)
    Timestamp now = Timestamp::Now();
    size_t patterns_with_decay = 0;

    QueryOptions all = QueryOptions::Unlimited();
    for (const auto& id : pattern_db_->FindAll(all)) {
        auto node = pattern_db_->Retrieve(id);
        if (!node) {
            continue;
        }

        auto stats = access_tracker_.GetPatternStatsSnapshot(id);
        Timestamp last_use = stats ? stats->last_access : AccessStatsFromNode(*node).last_access;
        Timestamp decay_start = std::max(last_use, last_forgetting_);
        if (!(decay_start < now)) {
            continue;
        }

        float confidence = node->GetConfidenceScore();
        float decayed = decay_function_->ApplyDecay(confidence, now - decay_start);
        if (decayed < confidence) {
            node->SetConfidenceScore(decayed);
            if (pattern_db_->Update(*node)) {
                patterns_with_decay++;
            }
        }
    }

    std::lock_guard<std::mutex> lock(stats_mutex_);
    cached_stats_.patterns_with_decay = patterns_with_decay;
}

void MemoryManager::ApplyInterferenceToPatterns() {
    if (!pattern_db_ || !assoc_matrix_ || !similarity_metric_) {
        return;
    }

    // Similar patterns weaken each other's confidence:
    //   s'(p) = s(p) × (1 - α × min(1, Σ similarity(p, q) × s(q)))
    // scaled by the hours elapsed since the previous pass (capped at 1) so
    // the effect is per unit time rather than per maintenance call.
    Timestamp now = Timestamp::Now();
    float hours = std::chrono::duration<float, std::ratio<3600>>(now - last_forgetting_).count();
    float time_scale = std::clamp(hours, 0.0f, 1.0f);
    if (time_scale <= 0.0f) {
        return;
    }

    QueryOptions all = QueryOptions::Unlimited();
    std::vector<PatternNode> all_nodes;
    for (const auto& id : pattern_db_->FindAll(all)) {
        if (auto node = pattern_db_->Retrieve(id)) {
            all_nodes.push_back(std::move(*node));
        }
    }

    // Keep the most recently accessed patterns when over the limit
    // (PatternNode is not move-assignable, so sort indices)
    std::vector<size_t> order(all_nodes.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    if (order.size() > config_.max_interference_patterns) {
        std::partial_sort(order.begin(), order.begin() + config_.max_interference_patterns, order.end(),
            [&all_nodes](size_t a, size_t b) {
                return all_nodes[b].GetLastAccessed() < all_nodes[a].GetLastAccessed();
            });
        order.resize(config_.max_interference_patterns);
    }

    std::vector<PatternNode*> nodes;
    nodes.reserve(order.size());
    for (size_t index : order) {
        nodes.push_back(&all_nodes[index]);
    }

    std::vector<FeatureVector> features;
    features.reserve(nodes.size());
    for (const auto& node : nodes) {
        features.push_back(node->GetData().GetFeatures());
    }

    size_t patterns_with_interference = 0;
    float total_interference = 0.0f;

    for (size_t i = 0; i < nodes.size(); ++i) {
        float interference = 0.0f;
        for (size_t j = 0; j < nodes.size(); ++j) {
            if (i == j || features[i].Dimension() != features[j].Dimension()) {
                continue;
            }
            interference += interference_calculator_.CalculateInterference(
                features[i], features[j], nodes[j]->GetConfidenceScore());
        }
        interference = std::min(1.0f, interference) * time_scale;
        if (interference <= 0.0f) {
            continue;
        }

        float confidence = nodes[i]->GetConfidenceScore();
        float reduced = interference_calculator_.ApplyInterference(confidence, interference);
        if (reduced < confidence) {
            nodes[i]->SetConfidenceScore(reduced);
            if (pattern_db_->Update(*nodes[i])) {
                patterns_with_interference++;
                total_interference += interference;
            }
        }
    }

    std::lock_guard<std::mutex> lock(stats_mutex_);
    cached_stats_.patterns_with_interference = patterns_with_interference;
    cached_stats_.average_interference = patterns_with_interference > 0
        ? total_interference / patterns_with_interference
        : 0.0f;
}

} // namespace dpan
