// File: src/discovery/pattern_matcher.cpp
#include "pattern_matcher.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace dpan {

// ============================================================================
// PatternMatcher Implementation
// ============================================================================

PatternMatcher::PatternMatcher(
    std::shared_ptr<PatternDatabase> database,
    std::shared_ptr<SimilarityMetric> metric,
    const Config& config)
    : database_(database), metric_(metric), config_(config) {

    if (!database_) {
        throw std::invalid_argument("PatternMatcher requires non-null database");
    }
    if (!metric_) {
        throw std::invalid_argument("PatternMatcher requires non-null metric");
    }
    if (config_.similarity_threshold < 0.0f || config_.similarity_threshold > 1.0f) {
        throw std::invalid_argument("similarity_threshold must be in range [0.0, 1.0]");
    }
    if (config_.strong_match_threshold < config_.weak_match_threshold) {
        throw std::invalid_argument("strong_match_threshold must be >= weak_match_threshold");
    }
}

PatternMatcher::PatternMatcher(
    std::shared_ptr<PatternDatabase> database,
    std::shared_ptr<SimilarityMetric> metric)
    : PatternMatcher(database, metric, Config{}) {
}

void PatternMatcher::SetConfig(const Config& config) {
    if (config.similarity_threshold < 0.0f || config.similarity_threshold > 1.0f) {
        throw std::invalid_argument("similarity_threshold must be in range [0.0, 1.0]");
    }
    if (config.strong_match_threshold < config.weak_match_threshold) {
        throw std::invalid_argument("strong_match_threshold must be >= weak_match_threshold");
    }
    {
        std::lock_guard<std::mutex> lock(sync_mutex_);
        const auto& old_index = config_.fast_search_index;
        const auto& new_index = config.fast_search_index;
        if (old_index.num_tables != new_index.num_tables ||
            old_index.bits_per_table != new_index.bits_per_table ||
            old_index.multi_probe != new_index.multi_probe || old_index.seed != new_index.seed) {
            index_.reset();  // Rebuilt with the new parameters on next use
            index_valid_ = false;
        }
    }
    config_ = config;
}

// ============================================================================
// Fast-search index
// ============================================================================

void PatternMatcher::SyncIndex() const {
    if (!index_) {
        index_ = std::make_unique<LshIndex>(config_.fast_search_index);
        index_valid_ = false;
    }

    auto ids = database_->FindAll(QueryOptions::Unlimited());

    if (!index_valid_) {
        index_->Clear();
        for (const auto& id : ids) {
            if (auto node = database_->Retrieve(id)) {
                index_->Insert(id, node->GetData().GetFeatures());
            }
        }
        index_valid_ = true;
        return;
    }

    // Incremental: index new IDs, drop IDs no longer in the database
    std::unordered_set<PatternID> current(ids.begin(), ids.end());
    for (const auto& id : ids) {
        if (!index_->Contains(id)) {
            if (auto node = database_->Retrieve(id)) {
                index_->Insert(id, node->GetData().GetFeatures());
            }
        }
    }
    for (const auto& id : index_->Ids()) {
        if (current.count(id) == 0) {
            index_->Remove(id);
        }
    }
}

bool PatternMatcher::IsUsingFastSearch() const {
    return config_.use_fast_search && database_->Count() >= config_.fast_search_min_patterns;
}

std::vector<PatternID> PatternMatcher::CandidateIds(const PatternData& candidate) const {
    if (!IsUsingFastSearch()) {
        return database_->FindAll(QueryOptions::Unlimited());
    }

    std::lock_guard<std::mutex> lock(sync_mutex_);
    if (!index_ || !index_valid_ || reconcile_pending_ || index_->Size() != database_->Count()) {
        SyncIndex();
        reconcile_pending_ = false;
    }
    return index_->Candidates(candidate.GetFeatures());
}

void PatternMatcher::NotifyPatternChanged(PatternID id) {
    std::lock_guard<std::mutex> lock(sync_mutex_);
    if (!index_ || !index_valid_) {
        return;  // Built from the database on first fast search
    }
    if (auto node = database_->Retrieve(id)) {
        index_->Insert(id, node->GetData().GetFeatures());
    } else {
        index_->Remove(id);
    }
}

void PatternMatcher::NotifyPatternRemoved(PatternID id) {
    std::lock_guard<std::mutex> lock(sync_mutex_);
    if (index_) {
        index_->Remove(id);
    }
}

void PatternMatcher::InvalidateIndex() {
    std::lock_guard<std::mutex> lock(sync_mutex_);
    reconcile_pending_ = true;
}

void PatternMatcher::RebuildIndex() {
    std::lock_guard<std::mutex> lock(sync_mutex_);
    index_valid_ = false;
}

void PatternMatcher::SetMetric(std::shared_ptr<SimilarityMetric> metric) {
    if (!metric) {
        throw std::invalid_argument("PatternMatcher requires non-null metric");
    }
    metric_ = metric;
}

std::vector<PatternMatcher::Match> PatternMatcher::FindMatches(const PatternData& candidate) const {
    std::vector<Match> matches;

    // Every pattern, or approximate-search candidates for large databases
    auto all_ids = CandidateIds(candidate);

    // Compute similarity for each pattern
    for (const auto& pattern_id : all_ids) {
        auto node_opt = database_->Retrieve(pattern_id);
        if (!node_opt) {
            continue;
        }

        // Compute similarity
        float similarity = metric_->Compute(candidate, node_opt->GetData());

        // Filter by threshold
        if (similarity < config_.similarity_threshold) {
            continue;
        }

        // Compute confidence
        float confidence = ComputeConfidence(similarity, *node_opt);

        matches.emplace_back(pattern_id, similarity, confidence);
    }

    // Sort by similarity (highest first)
    std::sort(matches.begin(), matches.end(),
        [](const Match& a, const Match& b) {
            return a.similarity > b.similarity;
        });

    // Limit to max_matches
    if (matches.size() > config_.max_matches) {
        matches.resize(config_.max_matches);
    }

    return matches;
}

PatternMatcher::MatchDecision PatternMatcher::MakeDecision(const PatternData& candidate) const {
    // Find matches
    auto matches = FindMatches(candidate);

    // No matches found - create new pattern
    if (matches.empty()) {
        return MatchDecision(
            Decision::CREATE_NEW,
            std::nullopt,
            1.0f,
            "No similar patterns found above threshold"
        );
    }

    // Get best match
    const Match& best_match = matches[0];

    // Strong match - update existing pattern
    if (best_match.similarity >= config_.strong_match_threshold) {
        std::ostringstream reason;
        reason << "Strong match found (similarity=" << best_match.similarity
               << ", confidence=" << best_match.confidence << ")";

        return MatchDecision(
            Decision::UPDATE_EXISTING,
            best_match.id,
            best_match.confidence,
            reason.str()
        );
    }

    // Weak match - merge patterns
    if (best_match.similarity >= config_.weak_match_threshold) {
        std::ostringstream reason;
        reason << "Weak match found (similarity=" << best_match.similarity
               << ", confidence=" << best_match.confidence << "), merge recommended";

        return MatchDecision(
            Decision::MERGE_SIMILAR,
            best_match.id,
            best_match.confidence * 0.8f,  // Lower confidence for merge
            reason.str()
        );
    }

    // Below weak threshold - create new
    std::ostringstream reason;
    reason << "Best match too weak (similarity=" << best_match.similarity << ")";

    return MatchDecision(
        Decision::CREATE_NEW,
        std::nullopt,
        0.9f,
        reason.str()
    );
}

float PatternMatcher::ComputeConfidence(float similarity, const PatternNode& node) const {
    // Confidence is based on:
    // 1. Similarity score (primary factor)
    // 2. Pattern's existing confidence score
    // 3. Pattern's access count (experience)

    float confidence = similarity;  // Start with similarity

    // Incorporate pattern's confidence
    float pattern_confidence = node.GetConfidenceScore();
    confidence = (confidence + pattern_confidence) / 2.0f;

    // Incorporate access count (more accesses = more reliable)
    uint32_t access_count = node.GetAccessCount();
    float experience_factor = std::min(1.0f, access_count / 100.0f);  // Cap at 100
    confidence = confidence * 0.7f + confidence * experience_factor * 0.3f;

    return std::max(0.0f, std::min(1.0f, confidence));
}

} // namespace dpan
