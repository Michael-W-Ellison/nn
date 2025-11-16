// File: src/discovery/pattern_matcher.cpp
#include "pattern_matcher.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>

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
    config_ = config;
}

void PatternMatcher::SetMetric(std::shared_ptr<SimilarityMetric> metric) {
    if (!metric) {
        throw std::invalid_argument("PatternMatcher requires non-null metric");
    }
    metric_ = metric;
}

std::vector<PatternMatcher::Match> PatternMatcher::FindMatches(const PatternData& candidate) const {
    std::vector<Match> matches;

    // Get all pattern IDs from database
    auto all_ids = database_->FindAll();

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
