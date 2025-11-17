// File: src/association/spatial_learner.cpp
#include "association/spatial_learner.hpp"
#include <algorithm>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

SpatialLearner::SpatialLearner()
    : config_()
{
}

SpatialLearner::SpatialLearner(const Config& config)
    : config_(config)
{
}

// ============================================================================
// Recording Spatial Context
// ============================================================================

void SpatialLearner::RecordSpatialContext(
    PatternID pattern,
    const ContextVector& context,
    Timestamp timestamp
) {
    RecordSpatialContext(pattern, context, {}, timestamp);
}

void SpatialLearner::RecordSpatialContext(
    PatternID pattern,
    const ContextVector& context,
    const std::vector<PatternID>& co_occurring,
    Timestamp timestamp
) {
    // Create context observation
    SpatialContext observation;
    observation.context = context;
    observation.timestamp = timestamp;
    observation.co_occurring_patterns = co_occurring;

    // Add to history
    auto& history = context_history_[pattern];
    history.push_back(observation);

    // Prune if exceeds max history
    if (history.size() > config_.max_history) {
        history.pop_front();
    }

    // Update aggregated statistics
    UpdateAverageContext(pattern, context, timestamp);
}

// ============================================================================
// Querying Spatial Relationships
// ============================================================================

bool SpatialLearner::AreSpatiallyRelated(
    PatternID p1,
    PatternID p2,
    float threshold
) const {
    // Use config threshold if not specified
    if (threshold < 0.0f) {
        threshold = config_.min_similarity_threshold;
    }

    // Check if both patterns have sufficient observations
    if (!HasSufficientObservations(p1) || !HasSufficientObservations(p2)) {
        return false;
    }

    // Compute similarity
    float similarity = GetSpatialSimilarity(p1, p2);

    return similarity >= threshold;
}

ContextVector SpatialLearner::GetAverageContext(PatternID pattern) const {
    auto it = spatial_stats_.find(pattern);
    if (it == spatial_stats_.end()) {
        return ContextVector();  // Empty context
    }

    return it->second.average_context;
}

std::optional<SpatialLearner::SpatialStats> SpatialLearner::GetSpatialStats(
    PatternID pattern
) const {
    auto it = spatial_stats_.find(pattern);
    if (it == spatial_stats_.end()) {
        return std::nullopt;
    }

    return it->second;
}

float SpatialLearner::GetSpatialSimilarity(PatternID p1, PatternID p2) const {
    // Get average contexts
    auto ctx1_it = spatial_stats_.find(p1);
    auto ctx2_it = spatial_stats_.find(p2);

    // Return 0 if either pattern has no observations
    if (ctx1_it == spatial_stats_.end() || ctx2_it == spatial_stats_.end()) {
        return 0.0f;
    }

    // Check minimum observations
    if (!HasSufficientObservations(p1) || !HasSufficientObservations(p2)) {
        return 0.0f;
    }

    const ContextVector& ctx1 = ctx1_it->second.average_context;
    const ContextVector& ctx2 = ctx2_it->second.average_context;

    // Compute cosine similarity
    return ctx1.CosineSimilarity(ctx2);
}

std::vector<std::pair<PatternID, float>> SpatialLearner::GetSpatiallySimilar(
    PatternID pattern,
    float min_similarity
) const {
    std::vector<std::pair<PatternID, float>> results;

    // Get average context for query pattern
    auto it = spatial_stats_.find(pattern);
    if (it == spatial_stats_.end()) {
        return results;  // No data for pattern
    }

    if (!HasSufficientObservations(pattern)) {
        return results;  // Insufficient observations
    }

    const ContextVector& query_context = it->second.average_context;

    // Compare with all other patterns
    for (const auto& [other_pattern, other_stats] : spatial_stats_) {
        // Skip self
        if (other_pattern == pattern) {
            continue;
        }

        // Skip patterns with insufficient observations
        if (!HasSufficientObservations(other_pattern)) {
            continue;
        }

        // Compute similarity
        float similarity = query_context.CosineSimilarity(other_stats.average_context);

        // Add if meets threshold
        if (similarity >= min_similarity) {
            results.push_back({other_pattern, similarity});
        }
    }

    // Sort by similarity (descending)
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return results;
}

std::vector<SpatialLearner::SpatialContext> SpatialLearner::GetContextHistory(
    PatternID pattern
) const {
    auto it = context_history_.find(pattern);
    if (it == context_history_.end()) {
        return {};
    }

    // Convert deque to vector
    return std::vector<SpatialContext>(it->second.begin(), it->second.end());
}

// ============================================================================
// Maintenance
// ============================================================================

void SpatialLearner::PruneHistory(PatternID pattern, size_t max_to_keep) {
    auto it = context_history_.find(pattern);
    if (it == context_history_.end()) {
        return;
    }

    auto& history = it->second;
    while (history.size() > max_to_keep) {
        history.pop_front();
    }
}

void SpatialLearner::Clear() {
    spatial_stats_.clear();
    context_history_.clear();
}

void SpatialLearner::ClearPattern(PatternID pattern) {
    spatial_stats_.erase(pattern);
    context_history_.erase(pattern);
}

// ============================================================================
// Statistics
// ============================================================================

size_t SpatialLearner::GetTotalObservations() const {
    size_t total = 0;
    for (const auto& [pattern, stats] : spatial_stats_) {
        total += stats.observation_count;
    }
    return total;
}

size_t SpatialLearner::GetObservationCount(PatternID pattern) const {
    auto it = spatial_stats_.find(pattern);
    if (it == spatial_stats_.end()) {
        return 0;
    }
    return it->second.observation_count;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void SpatialLearner::UpdateAverageContext(
    PatternID pattern,
    const ContextVector& observed_context,
    Timestamp timestamp
) {
    auto& stats = spatial_stats_[pattern];

    if (stats.observation_count == 0) {
        // First observation: initialize with observed context
        stats.average_context = observed_context;
        stats.observation_count = 1;
        stats.last_observed = timestamp;
    } else {
        // Update using exponential moving average
        // For each dimension in the observed context
        for (const auto& dim : observed_context.GetDimensions()) {
            float current = stats.average_context.Get(dim);
            float observed = observed_context.Get(dim);
            float updated = current + config_.learning_rate * (observed - current);
            stats.average_context.Set(dim, updated);
        }

        // Also update dimensions that exist in average but not in observed
        // (they should decay toward 0)
        for (const auto& dim : stats.average_context.GetDimensions()) {
            if (!observed_context.Has(dim)) {
                float current = stats.average_context.Get(dim);
                float updated = current * (1.0f - config_.learning_rate);
                stats.average_context.Set(dim, updated);
            }
        }

        stats.observation_count++;
        stats.last_observed = timestamp;
    }
}

bool SpatialLearner::HasSufficientObservations(PatternID pattern) const {
    auto it = spatial_stats_.find(pattern);
    if (it == spatial_stats_.end()) {
        return false;
    }
    return it->second.observation_count >= config_.min_observations;
}

} // namespace dpan
