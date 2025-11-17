// File: src/memory/tier_manager.cpp
//
// Implementation of Tier Manager

#include "memory/tier_manager.hpp"
#include <algorithm>
#include <stdexcept>
#include <chrono>

namespace dpan {

// ============================================================================
// Config
// ============================================================================

bool TierManager::Config::IsValid() const {
    // Check capacities
    if (active_capacity == 0 || warm_capacity == 0 || cold_capacity == 0) {
        return false;
    }

    if (active_capacity > warm_capacity || warm_capacity > cold_capacity) {
        return false;  // Tiers should increase in capacity
    }

    // Check thresholds are in valid range [0,1]
    if (warm_to_active_threshold < 0.0f || warm_to_active_threshold > 1.0f) return false;
    if (cold_to_warm_threshold < 0.0f || cold_to_warm_threshold > 1.0f) return false;
    if (archive_to_cold_threshold < 0.0f || archive_to_cold_threshold > 1.0f) return false;
    if (active_to_warm_threshold < 0.0f || active_to_warm_threshold > 1.0f) return false;
    if (warm_to_cold_threshold < 0.0f || warm_to_cold_threshold > 1.0f) return false;
    if (cold_to_archive_threshold < 0.0f || cold_to_archive_threshold > 1.0f) return false;

    // Check promotion thresholds are higher than demotion thresholds (hysteresis)
    if (warm_to_active_threshold <= active_to_warm_threshold) return false;
    if (cold_to_warm_threshold <= warm_to_cold_threshold) return false;
    if (archive_to_cold_threshold <= cold_to_archive_threshold) return false;

    // Check batch size
    if (transition_batch_size == 0 || transition_batch_size > 100000) {
        return false;
    }

    // Check interval
    if (transition_interval_seconds <= 0.0f) {
        return false;
    }

    return true;
}

// ============================================================================
// Constructor/Destructor
// ============================================================================

TierManager::TierManager()
    : config_(Config{}), last_transition_(Timestamp::Now()) {
    // Use default configuration
}

TierManager::TierManager(const Config& config)
    : config_(config), last_transition_(Timestamp::Now()) {

    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid TierManager configuration");
    }
}

TierManager::~TierManager() {
    StopBackgroundTransitions();
}

// ============================================================================
// Initialization
// ============================================================================

void TierManager::Initialize(
    std::unique_ptr<IMemoryTier> active,
    std::unique_ptr<IMemoryTier> warm,
    std::unique_ptr<IMemoryTier> cold,
    std::unique_ptr<IMemoryTier> archive) {

    if (!active || !warm || !cold || !archive) {
        throw std::invalid_argument("All tiers must be non-null");
    }

    active_tier_ = std::move(active);
    warm_tier_ = std::move(warm);
    cold_tier_ = std::move(cold);
    archive_tier_ = std::move(archive);
}

bool TierManager::IsInitialized() const {
    return active_tier_ != nullptr &&
           warm_tier_ != nullptr &&
           cold_tier_ != nullptr &&
           archive_tier_ != nullptr;
}

// ============================================================================
// Tier Access
// ============================================================================

IMemoryTier* TierManager::GetTier(MemoryTier tier) {
    switch (tier) {
        case MemoryTier::ACTIVE:
            return active_tier_.get();
        case MemoryTier::WARM:
            return warm_tier_.get();
        case MemoryTier::COLD:
            return cold_tier_.get();
        case MemoryTier::ARCHIVE:
            return archive_tier_.get();
        default:
            return nullptr;
    }
}

const IMemoryTier* TierManager::GetTier(MemoryTier tier) const {
    switch (tier) {
        case MemoryTier::ACTIVE:
            return active_tier_.get();
        case MemoryTier::WARM:
            return warm_tier_.get();
        case MemoryTier::COLD:
            return cold_tier_.get();
        case MemoryTier::ARCHIVE:
            return archive_tier_.get();
        default:
            return nullptr;
    }
}

// ============================================================================
// Pattern Operations
// ============================================================================

bool TierManager::StorePattern(const PatternNode& pattern, MemoryTier tier) {
    if (!IsInitialized()) {
        return false;
    }

    IMemoryTier* target_tier = GetTier(tier);
    if (!target_tier) {
        return false;
    }

    bool success = target_tier->StorePattern(pattern);
    if (success) {
        std::unique_lock<std::shared_mutex> lock(location_mutex_);
        pattern_locations_[pattern.GetID()] = tier;
    }

    return success;
}

std::optional<PatternNode> TierManager::LoadPattern(PatternID id) {
    if (!IsInitialized()) {
        return std::nullopt;
    }

    // Find tier containing the pattern
    std::optional<MemoryTier> tier;
    {
        std::shared_lock<std::shared_mutex> lock(location_mutex_);
        auto it = pattern_locations_.find(id);
        if (it != pattern_locations_.end()) {
            tier = it->second;
        }
    }

    if (!tier) {
        return std::nullopt;
    }

    IMemoryTier* source_tier = GetTier(*tier);
    if (!source_tier) {
        return std::nullopt;
    }

    return source_tier->LoadPattern(id);
}

bool TierManager::RemovePattern(PatternID id) {
    if (!IsInitialized()) {
        return false;
    }

    // Find and remove from current tier
    std::optional<MemoryTier> tier;
    {
        std::unique_lock<std::shared_mutex> lock(location_mutex_);
        auto it = pattern_locations_.find(id);
        if (it != pattern_locations_.end()) {
            tier = it->second;
            pattern_locations_.erase(it);
        }
    }

    if (!tier) {
        return false;
    }

    IMemoryTier* source_tier = GetTier(*tier);
    if (!source_tier) {
        return false;
    }

    return source_tier->RemovePattern(id);
}

std::optional<MemoryTier> TierManager::GetPatternTier(PatternID id) const {
    std::shared_lock<std::shared_mutex> lock(location_mutex_);
    auto it = pattern_locations_.find(id);
    if (it != pattern_locations_.end()) {
        return it->second;
    }
    return std::nullopt;
}

// ============================================================================
// Tier Transitions
// ============================================================================

bool TierManager::MovePattern(PatternID id, MemoryTier from, MemoryTier to) {
    if (!IsInitialized()) {
        return false;
    }

    IMemoryTier* source_tier = GetTier(from);
    IMemoryTier* target_tier = GetTier(to);

    if (!source_tier || !target_tier) {
        return false;
    }

    // Load from source
    auto pattern = source_tier->LoadPattern(id);
    if (!pattern) {
        return false;
    }

    // Store in target
    if (!target_tier->StorePattern(*pattern)) {
        return false;
    }

    // Remove from source
    if (!source_tier->RemovePattern(id)) {
        // Rollback: remove from target
        target_tier->RemovePattern(id);
        return false;
    }

    // Update location
    {
        std::unique_lock<std::shared_mutex> lock(location_mutex_);
        pattern_locations_[id] = to;
    }

    // Update statistics
    if (static_cast<int>(to) < static_cast<int>(from)) {
        promotions_count_.fetch_add(1);
    } else {
        demotions_count_.fetch_add(1);
    }

    return true;
}

bool TierManager::PromotePattern(PatternID id, MemoryTier target_tier) {
    auto current_tier = GetPatternTier(id);
    if (!current_tier) {
        return false;
    }

    // Verify target is higher tier (lower enum value)
    if (static_cast<int>(target_tier) >= static_cast<int>(*current_tier)) {
        return false;
    }

    return MovePattern(id, *current_tier, target_tier);
}

bool TierManager::DemotePattern(PatternID id, MemoryTier target_tier) {
    auto current_tier = GetPatternTier(id);
    if (!current_tier) {
        return false;
    }

    // Verify target is lower tier (higher enum value)
    if (static_cast<int>(target_tier) <= static_cast<int>(*current_tier)) {
        return false;
    }

    return MovePattern(id, *current_tier, target_tier);
}

// ============================================================================
// Pattern Selection for Transitions
// ============================================================================

std::vector<PatternID> TierManager::SelectPatternsForPromotion(
    MemoryTier tier,
    const std::unordered_map<PatternID, float>& utilities) {

    float threshold = GetPromotionThreshold(tier);
    if (threshold < 0.0f) {
        return {};  // No promotion from this tier
    }

    std::vector<PatternID> candidates;

    // Find patterns in this tier with utility above threshold
    {
        std::shared_lock<std::shared_mutex> lock(location_mutex_);
        for (const auto& [pattern_id, pattern_tier] : pattern_locations_) {
            if (pattern_tier == tier) {
                auto it = utilities.find(pattern_id);
                if (it != utilities.end() && it->second >= threshold) {
                    candidates.push_back(pattern_id);
                }
            }
        }
    }

    // Sort by utility (highest first)
    std::sort(candidates.begin(), candidates.end(),
              [&utilities](PatternID a, PatternID b) {
                  auto it_a = utilities.find(a);
                  auto it_b = utilities.find(b);
                  float util_a = (it_a != utilities.end()) ? it_a->second : 0.0f;
                  float util_b = (it_b != utilities.end()) ? it_b->second : 0.0f;
                  return util_a > util_b;
              });

    // Limit to batch size
    if (candidates.size() > config_.transition_batch_size) {
        candidates.resize(config_.transition_batch_size);
    }

    return candidates;
}

std::vector<PatternID> TierManager::SelectPatternsForDemotion(
    MemoryTier tier,
    const std::unordered_map<PatternID, float>& utilities) {

    float threshold = GetDemotionThreshold(tier);
    if (threshold < 0.0f) {
        return {};  // No demotion from this tier
    }

    std::vector<PatternID> candidates;

    // Find patterns in this tier with utility below threshold
    {
        std::shared_lock<std::shared_mutex> lock(location_mutex_);
        for (const auto& [pattern_id, pattern_tier] : pattern_locations_) {
            if (pattern_tier == tier) {
                auto it = utilities.find(pattern_id);
                if (it != utilities.end() && it->second < threshold) {
                    candidates.push_back(pattern_id);
                }
            }
        }
    }

    // Sort by utility (lowest first)
    std::sort(candidates.begin(), candidates.end(),
              [&utilities](PatternID a, PatternID b) {
                  auto it_a = utilities.find(a);
                  auto it_b = utilities.find(b);
                  float util_a = (it_a != utilities.end()) ? it_a->second : 0.0f;
                  float util_b = (it_b != utilities.end()) ? it_b->second : 0.0f;
                  return util_a < util_b;
              });

    // Limit to batch size
    if (candidates.size() > config_.transition_batch_size) {
        candidates.resize(config_.transition_batch_size);
    }

    return candidates;
}

// ============================================================================
// Capacity Enforcement
// ============================================================================

void TierManager::EnforceCapacityLimits(
    const std::unordered_map<PatternID, float>& utilities) {

    // Check each tier's capacity
    size_t active_count = active_tier_->GetPatternCount();
    size_t warm_count = warm_tier_->GetPatternCount();
    size_t cold_count = cold_tier_->GetPatternCount();

    // Active tier over capacity - demote lowest utility patterns
    if (active_count > config_.active_capacity) {
        size_t to_demote = active_count - config_.active_capacity;
        auto patterns = SelectPatternsForDemotion(MemoryTier::ACTIVE, utilities);
        for (size_t i = 0; i < std::min(to_demote, patterns.size()); ++i) {
            MovePattern(patterns[i], MemoryTier::ACTIVE, MemoryTier::WARM);
        }
    }

    // Warm tier over capacity
    if (warm_count > config_.warm_capacity) {
        size_t to_demote = warm_count - config_.warm_capacity;
        auto patterns = SelectPatternsForDemotion(MemoryTier::WARM, utilities);
        for (size_t i = 0; i < std::min(to_demote, patterns.size()); ++i) {
            MovePattern(patterns[i], MemoryTier::WARM, MemoryTier::COLD);
        }
    }

    // Cold tier over capacity
    if (cold_count > config_.cold_capacity) {
        size_t to_demote = cold_count - config_.cold_capacity;
        auto patterns = SelectPatternsForDemotion(MemoryTier::COLD, utilities);
        for (size_t i = 0; i < std::min(to_demote, patterns.size()); ++i) {
            MovePattern(patterns[i], MemoryTier::COLD, MemoryTier::ARCHIVE);
        }
    }
}

// ============================================================================
// Tier Transition Operations
// ============================================================================

size_t TierManager::PromotePatternsFromTier(
    MemoryTier tier,
    const std::unordered_map<PatternID, float>& utilities) {

    auto candidates = SelectPatternsForPromotion(tier, utilities);
    if (candidates.empty()) {
        return 0;
    }

    // Determine target tier (one level up)
    MemoryTier target_tier;
    switch (tier) {
        case MemoryTier::WARM:
            target_tier = MemoryTier::ACTIVE;
            break;
        case MemoryTier::COLD:
            target_tier = MemoryTier::WARM;
            break;
        case MemoryTier::ARCHIVE:
            target_tier = MemoryTier::COLD;
            break;
        default:
            return 0;  // Can't promote from active
    }

    size_t promoted = 0;
    for (const auto& pattern_id : candidates) {
        if (MovePattern(pattern_id, tier, target_tier)) {
            promoted++;
        }
    }

    return promoted;
}

size_t TierManager::DemotePatternsFromTier(
    MemoryTier tier,
    const std::unordered_map<PatternID, float>& utilities) {

    auto candidates = SelectPatternsForDemotion(tier, utilities);
    if (candidates.empty()) {
        return 0;
    }

    // Determine target tier (one level down)
    MemoryTier target_tier;
    switch (tier) {
        case MemoryTier::ACTIVE:
            target_tier = MemoryTier::WARM;
            break;
        case MemoryTier::WARM:
            target_tier = MemoryTier::COLD;
            break;
        case MemoryTier::COLD:
            target_tier = MemoryTier::ARCHIVE;
            break;
        default:
            return 0;  // Can't demote from archive
    }

    size_t demoted = 0;
    for (const auto& pattern_id : candidates) {
        if (MovePattern(pattern_id, tier, target_tier)) {
            demoted++;
        }
    }

    return demoted;
}

size_t TierManager::PerformTierTransitions(
    const std::unordered_map<PatternID, float>& utilities) {

    if (!IsInitialized()) {
        return 0;
    }

    size_t total_transitions = 0;

    // First enforce capacity limits
    EnforceCapacityLimits(utilities);

    // Perform promotions (Archive → Cold → Warm → Active)
    total_transitions += PromotePatternsFromTier(MemoryTier::ARCHIVE, utilities);
    total_transitions += PromotePatternsFromTier(MemoryTier::COLD, utilities);
    total_transitions += PromotePatternsFromTier(MemoryTier::WARM, utilities);

    // Perform demotions (Active → Warm → Cold → Archive)
    total_transitions += DemotePatternsFromTier(MemoryTier::ACTIVE, utilities);
    total_transitions += DemotePatternsFromTier(MemoryTier::WARM, utilities);
    total_transitions += DemotePatternsFromTier(MemoryTier::COLD, utilities);

    if (total_transitions > 0) {
        last_transition_ = Timestamp::Now();
    }

    return total_transitions;
}

// ============================================================================
// Thresholds
// ============================================================================

float TierManager::GetPromotionThreshold(MemoryTier tier) const {
    switch (tier) {
        case MemoryTier::WARM:
            return config_.warm_to_active_threshold;
        case MemoryTier::COLD:
            return config_.cold_to_warm_threshold;
        case MemoryTier::ARCHIVE:
            return config_.archive_to_cold_threshold;
        default:
            return -1.0f;  // No promotion from active
    }
}

float TierManager::GetDemotionThreshold(MemoryTier tier) const {
    switch (tier) {
        case MemoryTier::ACTIVE:
            return config_.active_to_warm_threshold;
        case MemoryTier::WARM:
            return config_.warm_to_cold_threshold;
        case MemoryTier::COLD:
            return config_.cold_to_archive_threshold;
        default:
            return -1.0f;  // No demotion from archive
    }
}

// ============================================================================
// Statistics
// ============================================================================

TierManager::TierStats TierManager::GetStats() const {
    TierStats stats;

    if (IsInitialized()) {
        stats.active_count = active_tier_->GetPatternCount();
        stats.warm_count = warm_tier_->GetPatternCount();
        stats.cold_count = cold_tier_->GetPatternCount();
        stats.archive_count = archive_tier_->GetPatternCount();
    }

    stats.promotions_count = promotions_count_.load();
    stats.demotions_count = demotions_count_.load();
    stats.last_transition = last_transition_;

    return stats;
}

// ============================================================================
// Background Thread
// ============================================================================

void TierManager::StartBackgroundTransitions(
    const UtilityCalculator* utility_calc,
    const AccessTracker* access_tracker) {

    if (!utility_calc || !access_tracker) {
        throw std::invalid_argument("Utility calculator and access tracker required");
    }

    if (running_.load()) {
        return;  // Already running
    }

    running_.store(true);
    background_thread_ = std::make_unique<std::thread>(
        &TierManager::BackgroundTransitionLoop,
        this,
        utility_calc,
        access_tracker
    );
}

void TierManager::StopBackgroundTransitions() {
    if (!running_.load()) {
        return;  // Not running
    }

    running_.store(false);

    if (background_thread_ && background_thread_->joinable()) {
        background_thread_->join();
    }

    background_thread_.reset();
}

bool TierManager::IsBackgroundRunning() const {
    return running_.load();
}

void TierManager::BackgroundTransitionLoop(
    const UtilityCalculator* utility_calc,
    const AccessTracker* access_tracker) {

    (void)utility_calc;      // TODO: Calculate utilities
    (void)access_tracker;    // TODO: Get access stats

    while (running_.load()) {
        // Sleep for interval
        auto interval = std::chrono::duration<float>(config_.transition_interval_seconds);
        auto interval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(interval);
        std::this_thread::sleep_for(interval_ms);

        if (!running_.load()) {
            break;
        }

        // TODO: Calculate utilities for all patterns
        // TODO: Perform tier transitions
        // For now, this is a placeholder
    }
}

// ============================================================================
// Configuration
// ============================================================================

void TierManager::SetConfig(const Config& config) {
    if (!config.IsValid()) {
        throw std::invalid_argument("Invalid TierManager configuration");
    }

    config_ = config;
}

} // namespace dpan
