// File: src/core/pattern_engine.cpp
#include "pattern_engine.hpp"
#include "storage/memory_backend.hpp"
#include "storage/persistent_backend.hpp"
#include "similarity/contextual_similarity.hpp"
#include "similarity/geometric_similarity.hpp"
#include "similarity/statistical_similarity.hpp"
#include "similarity/frequency_similarity.hpp"
#include <chrono>
#include <stdexcept>

namespace dpan {

// ============================================================================
// Constructor & Initialization
// ============================================================================

PatternEngine::PatternEngine(const Config& config)
    : config_(config) {
    InitializeComponents();
}

PatternEngine::~PatternEngine() {
    // Ensure all components are properly cleaned up
    if (database_) {
        Flush();
    }
}

void PatternEngine::InitializeComponents() {
    // Create database
    if (config_.database_type == "memory") {
        MemoryBackend::Config db_config;
        database_ = std::make_shared<MemoryBackend>(db_config);
    } else if (config_.database_type == "persistent") {
        PersistentBackend::Config db_config;
        db_config.db_path = config_.database_path;
        database_ = std::make_shared<PersistentBackend>(db_config);
    } else {
        throw std::invalid_argument("Unknown database type: " + config_.database_type);
    }

    // Create similarity metric
    similarity_metric_ = CreateSimilarityMetric(config_.similarity_metric);

    // Create pattern extractor
    extractor_ = std::make_unique<PatternExtractor>(config_.extraction_config);

    // Create pattern creator
    creator_ = std::make_unique<PatternCreator>(database_);

    // Create pattern matcher
    matcher_ = std::make_unique<PatternMatcher>(
        database_,
        similarity_metric_,
        config_.matching_config
    );

    // Create pattern refiner
    refiner_ = std::make_unique<PatternRefiner>(database_);

    // Create similarity search
    if (config_.enable_indexing) {
        similarity_search_ = std::make_unique<SimilaritySearch>(
            database_,
            similarity_metric_
        );
    }
}

std::shared_ptr<SimilarityMetric> PatternEngine::CreateSimilarityMetric(
    const std::string& metric_name) {

    if (metric_name == "context" || metric_name == "contextvector") {
        return std::make_shared<ContextVectorSimilarity>();
    } else if (metric_name == "hausdorff") {
        return std::make_shared<HausdorffSimilarity>();
    } else if (metric_name == "chamfer") {
        return std::make_shared<ChamferSimilarity>();
    } else if (metric_name == "temporal") {
        return std::make_shared<TemporalSimilarity>();
    } else if (metric_name == "histogram") {
        return std::make_shared<HistogramSimilarity>();
    } else if (metric_name == "spectral") {
        return std::make_shared<SpectralSimilarity>();
    } else {
        // Default to context vector similarity
        return std::make_shared<ContextVectorSimilarity>();
    }
}

// ============================================================================
// High-Level API
// ============================================================================

PatternEngine::ProcessResult PatternEngine::ProcessInput(
    const std::vector<uint8_t>& raw_input,
    DataModality modality) {

    auto start_time = std::chrono::high_resolution_clock::now();

    ProcessResult result;
    result.activated_patterns.clear();
    result.created_patterns.clear();
    result.updated_patterns.clear();

    // Step 1: Extract patterns from raw input
    auto extracted_patterns = extractor_->Extract(raw_input);

    if (extracted_patterns.empty()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        result.processing_time_ms = std::chrono::duration<float, std::milli>(
            end_time - start_time).count();
        return result;
    }

    // Step 2: For each extracted pattern, find matches and make decisions
    for (const auto& pattern_data : extracted_patterns) {
        auto matches = matcher_->FindMatches(pattern_data);
        auto decision = matcher_->MakeDecision(pattern_data);

        switch (decision.decision) {
            case PatternMatcher::Decision::CREATE_NEW: {
                // Create new pattern
                PatternID new_id = creator_->CreatePattern(
                    pattern_data,
                    PatternType::ATOMIC,
                    decision.confidence
                );
                result.created_patterns.push_back(new_id);
                break;
            }

            case PatternMatcher::Decision::UPDATE_EXISTING: {
                // Update existing pattern (if decision specifies one)
                if (decision.existing_id.has_value()) {
                    PatternID match_id = decision.existing_id.value();
                    result.activated_patterns.push_back(match_id);

                    // Adjust confidence based on good match
                    if (config_.enable_auto_refinement) {
                        refiner_->AdjustConfidence(match_id, true);
                    }
                }
                break;
            }

            case PatternMatcher::Decision::MERGE_SIMILAR: {
                // Merge with similar patterns
                std::vector<PatternID> merge_candidates;
                for (const auto& match : matches) {
                    if (match.similarity >= config_.matching_config.weak_match_threshold) {
                        merge_candidates.push_back(match.id);
                    }
                }

                if (!merge_candidates.empty() && config_.enable_auto_refinement) {
                    auto merge_result = refiner_->MergePatterns(merge_candidates);
                    if (merge_result.success) {
                        result.created_patterns.push_back(merge_result.merged_id);
                        // Mark originals as updated (merged away)
                        for (const auto& id : merge_candidates) {
                            result.updated_patterns.push_back(id);
                        }
                    }
                }
                break;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<float, std::milli>(
        end_time - start_time).count();

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        total_inputs_processed_++;
        total_patterns_created_ += result.created_patterns.size();
        total_patterns_updated_ += result.updated_patterns.size();
    }

    return result;
}

std::vector<PatternID> PatternEngine::DiscoverPatterns(
    const std::vector<uint8_t>& raw_input,
    DataModality modality) {

    std::vector<PatternID> discovered;

    // Extract patterns from raw input
    auto extracted_patterns = extractor_->Extract(raw_input);

    // Create patterns for all extracted data
    for (const auto& pattern_data : extracted_patterns) {
        PatternID id = creator_->CreatePattern(pattern_data);
        discovered.push_back(id);
    }

    return discovered;
}

// ============================================================================
// Pattern Retrieval
// ============================================================================

std::optional<PatternNode> PatternEngine::GetPattern(PatternID id) const {
    return database_->Retrieve(id);
}

std::vector<PatternNode> PatternEngine::GetPatternsBatch(
    const std::vector<PatternID>& ids) const {

    std::vector<PatternNode> patterns;
    patterns.reserve(ids.size());

    for (const auto& id : ids) {
        auto pattern_opt = database_->Retrieve(id);
        if (pattern_opt.has_value()) {
            patterns.push_back(std::move(pattern_opt.value()));
        }
    }

    return patterns;
}

std::vector<PatternID> PatternEngine::GetAllPatternIDs() const {
    return database_->FindAll();
}

// ============================================================================
// Pattern Search
// ============================================================================

std::vector<SearchResult> PatternEngine::FindSimilarPatterns(
    const PatternData& query,
    size_t k,
    float threshold) const {

    if (!similarity_search_) {
        // Fallback to brute-force search if indexing disabled
        std::vector<SearchResult> results;
        auto all_ids = database_->FindAll();

        for (const auto& id : all_ids) {
            auto pattern_opt = database_->Retrieve(id);
            if (pattern_opt.has_value()) {
                float similarity = similarity_metric_->Compute(
                    query,
                    pattern_opt->GetData()
                );

                if (similarity >= threshold) {
                    results.emplace_back(id, similarity);
                }
            }
        }

        // Sort by similarity descending
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.similarity > b.similarity; });

        // Return top k
        if (results.size() > k) {
            results.erase(results.begin() + k, results.end());
        }

        return results;
    }

    SearchConfig config = SearchConfig::WithThreshold(threshold, k);
    return similarity_search_->Search(query, config);
}

std::vector<SearchResult> PatternEngine::FindSimilarPatternsById(
    PatternID query_id,
    size_t k,
    float threshold) const {

    auto query_opt = database_->Retrieve(query_id);
    if (!query_opt.has_value()) {
        return {};
    }

    return FindSimilarPatterns(query_opt->GetData(), k, threshold);
}

// ============================================================================
// Pattern Management
// ============================================================================

PatternID PatternEngine::CreatePattern(
    const PatternData& data,
    float confidence) {

    PatternID id = creator_->CreatePattern(data, PatternType::ATOMIC, confidence);

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        total_patterns_created_++;
    }

    return id;
}

PatternID PatternEngine::CreateCompositePattern(
    const std::vector<PatternID>& sub_patterns,
    const PatternData& data) {

    PatternID id = creator_->CreateCompositePattern(sub_patterns, data);

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        total_patterns_created_++;
    }

    return id;
}

bool PatternEngine::UpdatePattern(
    PatternID id,
    const PatternData& new_data) {

    bool success = refiner_->UpdatePattern(id, new_data);

    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        total_patterns_updated_++;
    }

    return success;
}

bool PatternEngine::DeletePattern(PatternID id) {
    return database_->Delete(id);
}

// ============================================================================
// Statistics & Information
// ============================================================================

PatternEngine::Statistics PatternEngine::GetStatistics() const {
    Statistics stats;

    auto all_ids = database_->FindAll();
    stats.total_patterns = all_ids.size();

    float total_confidence = 0.0f;
    float total_size = 0.0f;

    for (const auto& id : all_ids) {
        auto pattern_opt = database_->Retrieve(id);
        if (pattern_opt.has_value()) {
            const auto& pattern = pattern_opt.value();

            // Count by type
            switch (pattern.GetType()) {
                case PatternType::ATOMIC:
                    stats.atomic_patterns++;
                    break;
                case PatternType::COMPOSITE:
                    stats.composite_patterns++;
                    break;
                case PatternType::META:
                    stats.meta_patterns++;
                    break;
            }

            // Accumulate confidence
            total_confidence += pattern.GetConfidenceScore();

            // Estimate size (rough approximation)
            total_size += pattern.GetData().GetFeatures().Dimension() * sizeof(float);
        }
    }

    if (stats.total_patterns > 0) {
        stats.avg_confidence = total_confidence / stats.total_patterns;
        stats.avg_pattern_size_bytes = total_size / stats.total_patterns;
    }

    // Get storage stats
    stats.storage_stats = database_->GetStats();

    return stats;
}

// ============================================================================
// Maintenance
// ============================================================================

void PatternEngine::Compact() {
    database_->Compact();
}

void PatternEngine::Flush() {
    database_->Flush();
}

void PatternEngine::RunMaintenance() {
    if (!config_.enable_auto_refinement) {
        return;
    }

    auto all_ids = database_->FindAll();

    // Check for patterns that need splitting
    std::vector<PatternID> to_split;
    for (const auto& id : all_ids) {
        if (refiner_->NeedsSplitting(id)) {
            to_split.push_back(id);
        }
    }

    // Split patterns that are too general
    for (const auto& id : to_split) {
        refiner_->SplitPattern(id, 2);
    }

    // Check for similar patterns that should be merged
    for (size_t i = 0; i < all_ids.size(); ++i) {
        for (size_t j = i + 1; j < all_ids.size(); ++j) {
            if (refiner_->ShouldMerge(all_ids[i], all_ids[j])) {
                refiner_->MergePatterns({all_ids[i], all_ids[j]});
                break;  // After merge, pattern IDs change
            }
        }
    }
}

// ============================================================================
// Snapshot & Restore
// ============================================================================

bool PatternEngine::SaveSnapshot(const std::string& path) {
    // For now, this is a simplified implementation
    // In a real system, you would serialize all engine state including:
    // - All patterns from database
    // - Statistics
    // - Configuration
    // - Index state (if applicable)

    // Flush database first
    Flush();

    // For memory backend, we can't easily save snapshot
    // For persistent backend, the database file itself is the snapshot
    if (config_.database_type == "persistent") {
        // The database is already persisted at config_.database_path
        // We could copy it to the snapshot path here
        return true;
    }

    // Memory backend doesn't support snapshots in this simple implementation
    return false;
}

bool PatternEngine::LoadSnapshot(const std::string& path) {
    // Simplified implementation
    // In a real system, you would deserialize all engine state

    if (config_.database_type == "persistent") {
        // Database automatically loads from config_.database_path
        return true;
    }

    return false;
}

} // namespace dpan
