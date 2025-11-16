// File: src/core/pattern_engine.hpp
#pragma once

#include "core/pattern_node.hpp"
#include "storage/pattern_database.hpp"
#include "similarity/similarity_metric.hpp"
#include "similarity/similarity_search.hpp"
#include "discovery/pattern_extractor.hpp"
#include "discovery/pattern_matcher.hpp"
#include "discovery/pattern_creator.hpp"
#include "discovery/pattern_refiner.hpp"
#include <memory>
#include <vector>
#include <string>
#include <mutex>

namespace dpan {

/// PatternEngine - Unified interface for all pattern operations
///
/// This facade class provides a high-level API that integrates all
/// pattern processing components: extraction, matching, creation,
/// refinement, and search.
class PatternEngine {
public:
    /// Configuration for the pattern engine
    struct Config {
        // Database configuration
        std::string database_path;
        std::string database_type{"memory"};  // "memory" or "persistent"

        // Component configurations
        PatternExtractor::Config extraction_config;
        PatternMatcher::Config matching_config;

        // Similarity metric selection
        std::string similarity_metric{"context"};  // "context", "hausdorff", "temporal", etc.

        // Engine options
        bool enable_auto_refinement{true};
        bool enable_indexing{true};
    };

    /// Result from processing input
    struct ProcessResult {
        std::vector<PatternID> activated_patterns;
        std::vector<PatternID> created_patterns;
        std::vector<PatternID> updated_patterns;
        float processing_time_ms;
    };

    /// Engine statistics
    struct Statistics {
        size_t total_patterns{0};
        size_t atomic_patterns{0};
        size_t composite_patterns{0};
        size_t meta_patterns{0};
        float avg_confidence{0.0f};
        float avg_pattern_size_bytes{0.0f};
        StorageStats storage_stats;
    };

    /// Constructor
    /// @param config Engine configuration
    explicit PatternEngine(const Config& config);

    /// Destructor
    ~PatternEngine();

    // Disable copy and move
    PatternEngine(const PatternEngine&) = delete;
    PatternEngine& operator=(const PatternEngine&) = delete;
    PatternEngine(PatternEngine&&) = delete;
    PatternEngine& operator=(PatternEngine&&) = delete;

    // ========================================================================
    // High-Level API
    // ========================================================================

    /// Process raw input end-to-end
    /// @param raw_input Raw input bytes
    /// @param modality Data modality
    /// @return Processing result with activated/created patterns
    ProcessResult ProcessInput(
        const std::vector<uint8_t>& raw_input,
        DataModality modality
    );

    /// Discover patterns from raw input
    /// @param raw_input Raw input bytes
    /// @param modality Data modality
    /// @return IDs of discovered patterns
    std::vector<PatternID> DiscoverPatterns(
        const std::vector<uint8_t>& raw_input,
        DataModality modality
    );

    // ========================================================================
    // Pattern Retrieval
    // ========================================================================

    /// Retrieve a single pattern
    /// @param id Pattern ID
    /// @return Pattern node if found
    std::optional<PatternNode> GetPattern(PatternID id) const;

    /// Retrieve multiple patterns
    /// @param ids Vector of pattern IDs
    /// @return Vector of pattern nodes (only found patterns)
    std::vector<PatternNode> GetPatternsBatch(const std::vector<PatternID>& ids) const;

    /// Get all pattern IDs
    /// @return Vector of all pattern IDs in the database
    std::vector<PatternID> GetAllPatternIDs() const;

    // ========================================================================
    // Pattern Search
    // ========================================================================

    /// Find similar patterns
    /// @param query Query pattern data
    /// @param k Number of results
    /// @param threshold Minimum similarity threshold
    /// @return Vector of search results
    std::vector<SearchResult> FindSimilarPatterns(
        const PatternData& query,
        size_t k = 10,
        float threshold = 0.0f
    ) const;

    /// Find similar patterns by ID
    /// @param query_id Query pattern ID
    /// @param k Number of results
    /// @param threshold Minimum similarity threshold
    /// @return Vector of search results
    std::vector<SearchResult> FindSimilarPatternsById(
        PatternID query_id,
        size_t k = 10,
        float threshold = 0.0f
    ) const;

    // ========================================================================
    // Pattern Management
    // ========================================================================

    /// Create a new atomic pattern
    /// @param data Pattern data
    /// @param confidence Initial confidence
    /// @return Created pattern ID
    PatternID CreatePattern(
        const PatternData& data,
        float confidence = 0.5f
    );

    /// Create a composite pattern
    /// @param sub_patterns Sub-pattern IDs
    /// @param data Composite pattern data
    /// @return Created pattern ID
    PatternID CreateCompositePattern(
        const std::vector<PatternID>& sub_patterns,
        const PatternData& data
    );

    /// Update an existing pattern
    /// @param id Pattern ID
    /// @param new_data New pattern data
    /// @return true if successful
    bool UpdatePattern(
        PatternID id,
        const PatternData& new_data
    );

    /// Delete a pattern
    /// @param id Pattern ID
    /// @return true if successful
    bool DeletePattern(PatternID id);

    // ========================================================================
    // Statistics & Information
    // ========================================================================

    /// Get engine statistics
    /// @return Current engine statistics
    Statistics GetStatistics() const;

    /// Get configuration
    /// @return Current configuration
    const Config& GetConfig() const { return config_; }

    // ========================================================================
    // Maintenance
    // ========================================================================

    /// Compact the database
    void Compact();

    /// Flush pending writes
    void Flush();

    /// Run maintenance tasks (auto-refinement, pruning, etc.)
    void RunMaintenance();

    // ========================================================================
    // Snapshot & Restore
    // ========================================================================

    /// Save engine state to snapshot
    /// @param path Snapshot file path
    /// @return true if successful
    bool SaveSnapshot(const std::string& path);

    /// Load engine state from snapshot
    /// @param path Snapshot file path
    /// @return true if successful
    bool LoadSnapshot(const std::string& path);

private:
    Config config_;

    // Core components
    std::shared_ptr<PatternDatabase> database_;
    std::shared_ptr<SimilarityMetric> similarity_metric_;
    std::unique_ptr<SimilaritySearch> similarity_search_;
    std::unique_ptr<PatternExtractor> extractor_;
    std::unique_ptr<PatternMatcher> matcher_;
    std::unique_ptr<PatternCreator> creator_;
    std::unique_ptr<PatternRefiner> refiner_;

    // Statistics tracking
    mutable std::mutex stats_mutex_;
    size_t total_inputs_processed_{0};
    size_t total_patterns_created_{0};
    size_t total_patterns_updated_{0};

    // Helper methods
    void InitializeComponents();
    std::shared_ptr<SimilarityMetric> CreateSimilarityMetric(const std::string& metric_name);
    void UpdateStatisticsAfterProcessing(const ProcessResult& result);
};

} // namespace dpan
