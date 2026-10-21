// File: src/discovery/pattern_matcher.hpp
#pragma once

#include "core/pattern_node.hpp"
#include "similarity/similarity_metric.hpp"
#include "similarity/lsh_index.hpp"
#include "storage/pattern_database.hpp"
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace dpan {

/// PatternMatcher - Finds matching patterns in the database and makes decisions
/// about pattern creation, update, or merging.
class PatternMatcher {
public:
    /// Configuration for pattern matching
    struct Config {
        /// Similarity threshold for considering a match (0.0 to 1.0)
        float similarity_threshold{0.7f};

        /// Maximum number of matches to return
        size_t max_matches{10};

        /// Use approximate search (LSH candidates re-ranked with the metric)
        /// once the database holds at least fast_search_min_patterns
        /// patterns; below that, FindMatches() scans every pattern.
        /// Approximate search can miss matches whose features point in a
        /// different direction than the candidate's (recall < 100%).
        bool use_fast_search{true};

        /// Database size at which approximate search starts being used
        size_t fast_search_min_patterns{10000};

        /// Hash index parameters for approximate search
        LshIndex::Config fast_search_index;

        /// Threshold for "strong match" (update existing pattern)
        float strong_match_threshold{0.85f};

        /// Threshold for "weak match" (merge patterns)
        float weak_match_threshold{0.7f};

        /// Minimum confidence for decision making
        float min_confidence{0.5f};
    };

    /// Match result containing pattern ID, similarity, and confidence
    struct Match {
        PatternID id;           ///< Matched pattern ID
        float similarity;       ///< Similarity score [0, 1]
        float confidence;       ///< Confidence in this match [0, 1]

        /// Default constructor
        Match() : id(0), similarity(0.0f), confidence(0.0f) {}

        /// Constructor
        Match(PatternID id_, float similarity_, float confidence_)
            : id(id_), similarity(similarity_), confidence(confidence_) {}
    };

    /// Decision about what to do with a candidate pattern
    enum class Decision {
        CREATE_NEW,      ///< No good match found, create new pattern
        UPDATE_EXISTING, ///< Strong match found, update existing pattern
        MERGE_SIMILAR    ///< Weak match found, merge with existing pattern
    };

    /// Decision result with reasoning
    struct MatchDecision {
        Decision decision;                      ///< The decision made
        std::optional<PatternID> existing_id;   ///< ID of existing pattern (if applicable)
        float confidence;                       ///< Confidence in this decision [0, 1]
        std::string reasoning;                  ///< Human-readable explanation

        /// Constructor
        MatchDecision(Decision dec, std::optional<PatternID> id, float conf, std::string reason)
            : decision(dec), existing_id(id), confidence(conf), reasoning(std::move(reason)) {}
    };

    /// Constructor with configuration
    /// @param database Pattern database to search
    /// @param metric Similarity metric to use
    /// @param config Configuration options
    PatternMatcher(
        std::shared_ptr<PatternDatabase> database,
        std::shared_ptr<SimilarityMetric> metric,
        const Config& config
    );

    /// Constructor with default configuration
    /// @param database Pattern database to search
    /// @param metric Similarity metric to use
    PatternMatcher(
        std::shared_ptr<PatternDatabase> database,
        std::shared_ptr<SimilarityMetric> metric
    );

    /// Find matching patterns for a candidate
    /// @param candidate Pattern data to find matches for
    /// @return Vector of matches sorted by similarity (highest first)
    std::vector<Match> FindMatches(const PatternData& candidate) const;

    /// Make a decision about what to do with a candidate pattern
    /// @param candidate Pattern data to make decision for
    /// @return Decision with reasoning
    MatchDecision MakeDecision(const PatternData& candidate) const;

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

    /// Update configuration
    void SetConfig(const Config& config);

    /// Set similarity metric
    void SetMetric(std::shared_ptr<SimilarityMetric> metric);

    // ------------------------------------------------------------------
    // Fast-search index maintenance
    //
    // The index follows the database automatically when the pattern count
    // changes (patterns are added or removed). Call these for changes the
    // count does not reveal: a pattern's features were updated, or patterns
    // were replaced one-for-one outside this matcher. PatternEngine calls
    // them for its own operations.
    // ------------------------------------------------------------------

    /// Index (or re-index) a pattern after it was stored or updated
    void NotifyPatternChanged(PatternID id);

    /// Drop a deleted pattern from the index
    void NotifyPatternRemoved(PatternID id);

    /// Reconcile indexed IDs with the database on the next fast search
    /// (after patterns were replaced, e.g. merged or split)
    void InvalidateIndex();

    /// Re-index every pattern on the next fast search (after bulk changes
    /// such as restoring a snapshot)
    void RebuildIndex();

    /// True when FindMatches() currently uses approximate search
    bool IsUsingFastSearch() const;

private:
    std::shared_ptr<PatternDatabase> database_;
    std::shared_ptr<SimilarityMetric> metric_;
    Config config_;

    // Approximate search state (index_ is internally synchronized;
    // sync_mutex_ serializes reconciliation with the database)
    mutable std::unique_ptr<LshIndex> index_;
    mutable std::mutex sync_mutex_;
    mutable bool index_valid_{false};       // false: full rebuild needed
    mutable bool reconcile_pending_{false};  // true: incremental ID sync needed

    /// Bring index_ in line with the database's pattern IDs
    void SyncIndex() const;

    /// Candidate IDs for a query: all patterns, or LSH candidates
    std::vector<PatternID> CandidateIds(const PatternData& candidate) const;

    /// Compute confidence score for a match
    float ComputeConfidence(float similarity, const PatternNode& node) const;
};

} // namespace dpan
