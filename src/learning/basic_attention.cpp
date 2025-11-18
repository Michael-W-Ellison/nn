// File: src/learning/basic_attention.cpp
//
// Implementation of BasicAttentionMechanism
//
// Key implementation details:
// - Dot-product similarity with temperature scaling
// - LRU cache with configurable size
// - Optional debug logging with minimal overhead
// - Thread-safe cache operations

#include "learning/basic_attention.hpp"
#include "learning/attention_utils.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace dpan {
namespace attention {

// ============================================================================
// CacheKey Implementation
// ============================================================================

bool CacheKey::operator==(const CacheKey& other) const {
    if (query != other.query) return false;
    if (candidates.size() != other.candidates.size()) return false;

    // Compare candidates (order matters for caching)
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i] != other.candidates[i]) return false;
    }

    // For simplicity, we don't compare context in cache key
    // (context similarity is less critical for caching)
    return true;
}

bool CacheKey::operator<(const CacheKey& other) const {
    // Compare query first
    if (query < other.query) return true;
    if (query > other.query) return false;

    // Compare number of candidates
    if (candidates.size() < other.candidates.size()) return true;
    if (candidates.size() > other.candidates.size()) return false;

    // Compare candidates lexicographically
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i] < other.candidates[i]) return true;
        if (candidates[i] > other.candidates[i]) return false;
    }

    // All equal
    return false;
}

size_t CacheKey::Hash::operator()(const CacheKey& key) const {
    size_t hash = std::hash<uint64_t>()(key.query.value());

    // Combine candidate hashes
    for (const auto& candidate : key.candidates) {
        hash ^= std::hash<uint64_t>()(candidate.value()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    return hash;
}

// ============================================================================
// BasicAttentionMechanism Implementation
// ============================================================================

BasicAttentionMechanism::BasicAttentionMechanism(const AttentionConfig& config)
    : config_(config)
    , pattern_db_(nullptr)
    , debug_stream_(&std::cout)
    , cache_hits_(0)
    , cache_misses_(0)
    , total_computations_(0)
{
    // Validate configuration
    if (!config_.Validate()) {
        // Use default config if invalid
        config_ = AttentionConfig();
    }

    // Set default feature config
    feature_config_.include_confidence = true;
    feature_config_.include_access_count = true;
    feature_config_.include_age = false;
    feature_config_.include_type = false;
}

std::map<PatternID, float> BasicAttentionMechanism::ComputeAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    ++total_computations_;

    // Handle empty candidates
    if (candidates.empty()) {
        LogDebug("ComputeAttention: No candidates provided");
        return {};
    }

    // Handle single candidate
    if (candidates.size() == 1) {
        LogDebug("ComputeAttention: Single candidate, returning weight 1.0");
        return {{candidates[0], 1.0f}};
    }

    // Check cache if enabled
    if (config_.enable_caching) {
        CacheKey key{query, candidates, context};
        auto cached = GetCachedAttention(key);
        if (cached) {
            LogDebug("ComputeAttention: Cache hit");
            return *cached;
        }
    }

    LogDebug("ComputeAttention: Cache miss, computing from scratch");

    // Validate pattern database
    if (!pattern_db_) {
        LogDebug("ERROR: Pattern database not set");
        // Return uniform distribution
        float uniform = 1.0f / candidates.size();
        std::map<PatternID, float> weights;
        for (const auto& candidate : candidates) {
            weights[candidate] = uniform;
        }
        return weights;
    }

    // Extract features for query
    auto query_features = ExtractFeatures(query, pattern_db_, feature_config_);
    if (query_features.empty()) {
        LogDebug("ERROR: Failed to extract query features");
        // Return uniform distribution
        float uniform = 1.0f / candidates.size();
        std::map<PatternID, float> weights;
        for (const auto& candidate : candidates) {
            weights[candidate] = uniform;
        }
        return weights;
    }

    // Extract features for all candidates
    auto candidate_features = ExtractMultipleFeatures(candidates);

    // Compute raw similarity scores
    auto raw_scores = ComputeRawScores(query_features, candidate_features);

    // Apply temperature scaling and softmax normalization
    auto normalized_weights = Softmax(raw_scores, config_.temperature);

    // Build result map
    std::map<PatternID, float> weights;
    for (size_t i = 0; i < candidates.size(); ++i) {
        weights[candidates[i]] = normalized_weights[i];
    }

    // Log debug information
    if (config_.debug_logging) {
        LogAttentionDetails(query, raw_scores, weights);
    }

    // Cache result if enabled
    if (config_.enable_caching) {
        CacheKey key{query, candidates, context};
        CacheAttention(key, weights);
    }

    return weights;
}

std::vector<AttentionScore> BasicAttentionMechanism::ComputeDetailedAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    std::vector<AttentionScore> detailed_scores;

    // Compute basic attention weights
    auto weights = ComputeAttention(query, candidates, context);

    // Build detailed scores
    for (const auto& candidate : candidates) {
        AttentionScore score;
        score.pattern_id = candidate;
        score.weight = weights[candidate];

        // For basic attention, we don't have component breakdown yet
        // (This will be enhanced in later phases with importance and context)
        score.raw_score = score.weight;  // Placeholder
        score.components.semantic_similarity = score.weight;

        detailed_scores.push_back(score);
    }

    // Sort by weight (descending)
    std::sort(detailed_scores.begin(), detailed_scores.end(),
        [](const AttentionScore& a, const AttentionScore& b) {
            return a.weight > b.weight;
        });

    return detailed_scores;
}

std::vector<std::pair<PatternID, float>> BasicAttentionMechanism::ApplyAttention(
    PatternID query,
    const std::vector<PatternID>& predictions,
    const ContextVector& context) {

    // Compute attention weights
    auto attention_weights = ComputeAttention(query, predictions, context);

    // For now, we don't have association scores, so we just use attention weights
    // In future integration with AssociationSystem, we'll combine:
    // final_score = attention_weight * attention_score + association_weight * assoc_score

    std::vector<std::pair<PatternID, float>> results;
    results.reserve(predictions.size());

    for (const auto& pred_id : predictions) {
        float attention_score = attention_weights[pred_id];

        // Weighted combination (placeholder for association scores)
        // When integrated with AssociationSystem, this will be:
        // float final_score = config_.attention_weight * attention_score +
        //                     config_.association_weight * association_score;
        float final_score = attention_score;

        results.push_back({pred_id, final_score});
    }

    // Sort by final score (descending)
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    return results;
}

void BasicAttentionMechanism::SetPatternDatabase(PatternDatabase* db) {
    pattern_db_ = db;

    // Clear cache when database changes
    if (config_.enable_caching) {
        ClearCache();
    }

    LogDebug("Pattern database set");
}

const AttentionConfig& BasicAttentionMechanism::GetConfig() const {
    return config_;
}

void BasicAttentionMechanism::SetConfig(const AttentionConfig& config) {
    config_ = config;

    // Clear cache when configuration changes
    if (config_.enable_caching) {
        ClearCache();
    }

    LogDebug("Configuration updated");
}

void BasicAttentionMechanism::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    cache_.clear();
    cache_order_.clear();

    LogDebug("Cache cleared");
}

std::map<std::string, float> BasicAttentionMechanism::GetStatistics() const {
    std::map<std::string, float> stats;

    stats["total_computations"] = static_cast<float>(total_computations_);
    stats["cache_hits"] = static_cast<float>(cache_hits_);
    stats["cache_misses"] = static_cast<float>(cache_misses_);

    float total_requests = static_cast<float>(cache_hits_ + cache_misses_);
    if (total_requests > 0) {
        stats["cache_hit_rate"] = static_cast<float>(cache_hits_) / total_requests;
    } else {
        stats["cache_hit_rate"] = 0.0f;
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        stats["cache_size"] = static_cast<float>(cache_.size());
    }

    return stats;
}

void BasicAttentionMechanism::SetFeatureConfig(const FeatureExtractionConfig& config) {
    feature_config_ = config;

    // Clear cache when feature config changes
    if (config_.enable_caching) {
        ClearCache();
    }

    LogDebug("Feature configuration updated");
}

const FeatureExtractionConfig& BasicAttentionMechanism::GetFeatureConfig() const {
    return feature_config_;
}

void BasicAttentionMechanism::SetDebugStream(std::ostream* os) {
    if (os) {
        debug_stream_ = os;
    }
}

// ============================================================================
// Private Methods
// ============================================================================

std::vector<float> BasicAttentionMechanism::ComputeRawScores(
    const std::vector<float>& query_features,
    const std::vector<std::vector<float>>& candidate_features) const {

    std::vector<float> scores;
    scores.reserve(candidate_features.size());

    for (const auto& candidate_feat : candidate_features) {
        // Use scaled dot-product for similarity
        float score = ScaledDotProduct(query_features, candidate_feat, true);
        scores.push_back(score);
    }

    return scores;
}

std::vector<std::vector<float>> BasicAttentionMechanism::ExtractMultipleFeatures(
    const std::vector<PatternID>& pattern_ids) const {

    std::vector<std::vector<float>> features;
    features.reserve(pattern_ids.size());

    for (const auto& pattern_id : pattern_ids) {
        auto feat = ExtractFeatures(pattern_id, pattern_db_, feature_config_);

        // Handle missing patterns with zero vector
        if (feat.empty()) {
            LogDebug("WARNING: Failed to extract features for pattern " +
                    std::to_string(pattern_id.value()));
            // Use small vector as placeholder
            feat = {0.0f};
        }

        features.push_back(feat);
    }

    return features;
}

std::optional<std::map<PatternID, float>> BasicAttentionMechanism::GetCachedAttention(
    const CacheKey& key) {

    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        ++cache_hits_;

        // Update LRU order (move to back)
        cache_order_.remove(key);
        cache_order_.push_back(key);

        return it->second;
    }

    ++cache_misses_;
    return std::nullopt;
}

void BasicAttentionMechanism::CacheAttention(
    const CacheKey& key,
    const std::map<PatternID, float>& weights) {

    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Check if cache is full
    if (cache_.size() >= config_.cache_size) {
        // Remove least recently used (front of list)
        if (!cache_order_.empty()) {
            auto lru_key = cache_order_.front();
            cache_order_.pop_front();
            cache_.erase(lru_key);
        }
    }

    // Add to cache
    cache_[key] = weights;
    cache_order_.push_back(key);
}

void BasicAttentionMechanism::LogDebug(const std::string& message) const {
    if (config_.debug_logging && debug_stream_) {
        *debug_stream_ << "[BasicAttention] " << message << std::endl;
    }
}

void BasicAttentionMechanism::LogAttentionDetails(
    PatternID query,
    const std::vector<float>& raw_scores,
    const std::map<PatternID, float>& weights) const {

    if (!config_.debug_logging || !debug_stream_) {
        return;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    oss << "\n=== Attention Computation Details ===" << std::endl;
    oss << "Query: " << query.value() << std::endl;
    oss << "Temperature: " << config_.temperature << std::endl;
    oss << "Num candidates: " << raw_scores.size() << std::endl;

    oss << "\nRaw scores:" << std::endl;
    size_t idx = 0;
    for (const auto& [pattern_id, weight] : weights) {
        if (idx < raw_scores.size()) {
            oss << "  Pattern " << pattern_id.value()
                << ": raw=" << raw_scores[idx]
                << ", weight=" << weight
                << std::endl;
        }
        ++idx;
    }

    // Verify normalization
    float sum = 0.0f;
    for (const auto& [_, weight] : weights) {
        sum += weight;
    }
    oss << "\nWeight sum: " << sum << " (should be ~1.0)" << std::endl;
    oss << "====================================\n" << std::endl;

    *debug_stream_ << oss.str();
}

} // namespace attention
} // namespace dpan
