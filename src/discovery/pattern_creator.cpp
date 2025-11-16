// File: src/discovery/pattern_creator.cpp
#include "pattern_creator.hpp"
#include <stdexcept>
#include <algorithm>

namespace dpan {

// ============================================================================
// PatternCreator Implementation
// ============================================================================

PatternCreator::PatternCreator(std::shared_ptr<PatternDatabase> database)
    : database_(database) {
    if (!database_) {
        throw std::invalid_argument("PatternCreator requires non-null database");
    }
}

PatternID PatternCreator::CreatePattern(
    const PatternData& data,
    PatternType type,
    float initial_confidence) {

    if (initial_confidence < 0.0f || initial_confidence > 1.0f) {
        throw std::invalid_argument("initial_confidence must be in range [0.0, 1.0]");
    }

    // Generate new pattern ID
    PatternID new_id = GeneratePatternID();

    // Create pattern node
    PatternNode node(new_id, data, type);

    // Set initial parameters
    node.SetActivationThreshold(default_activation_threshold_);
    node.SetConfidenceScore(initial_confidence);

    // Initialize statistics
    InitializeStatistics(node);

    // Store in database
    if (!database_->Store(node)) {
        throw std::runtime_error("Failed to store pattern in database");
    }

    return new_id;
}

PatternID PatternCreator::CreateCompositePattern(
    const std::vector<PatternID>& sub_patterns,
    const PatternData& composite_data) {

    if (sub_patterns.empty()) {
        throw std::invalid_argument("Composite pattern requires at least one sub-pattern");
    }

    // Verify all sub-patterns exist
    for (const auto& sub_id : sub_patterns) {
        if (!database_->Exists(sub_id)) {
            throw std::invalid_argument("Sub-pattern does not exist in database");
        }
    }

    // Generate new pattern ID
    PatternID composite_id = GeneratePatternID();

    // Create composite pattern node
    PatternNode node(composite_id, composite_data, PatternType::COMPOSITE);

    // Add sub-patterns
    for (const auto& sub_id : sub_patterns) {
        node.AddSubPattern(sub_id);
    }

    // Set initial parameters
    node.SetActivationThreshold(default_activation_threshold_);
    node.SetConfidenceScore(default_initial_confidence_);

    // Initialize statistics
    InitializeStatistics(node);

    // Store in database
    if (!database_->Store(node)) {
        throw std::runtime_error("Failed to store composite pattern in database");
    }

    return composite_id;
}

PatternID PatternCreator::CreateMetaPattern(
    const std::vector<PatternID>& pattern_instances,
    const PatternData& meta_data) {

    if (pattern_instances.empty()) {
        throw std::invalid_argument("Meta-pattern requires at least one pattern instance");
    }

    // Verify all pattern instances exist
    for (const auto& instance_id : pattern_instances) {
        if (!database_->Exists(instance_id)) {
            throw std::invalid_argument("Pattern instance does not exist in database");
        }
    }

    // Generate new pattern ID
    PatternID meta_id = GeneratePatternID();

    // Create meta-pattern node
    PatternNode node(meta_id, meta_data, PatternType::META);

    // Add pattern instances
    for (const auto& instance_id : pattern_instances) {
        node.AddSubPattern(instance_id);
    }

    // Set initial parameters (meta-patterns typically have higher thresholds)
    node.SetActivationThreshold(std::min(1.0f, default_activation_threshold_ * 1.2f));
    node.SetConfidenceScore(default_initial_confidence_);

    // Initialize statistics
    InitializeStatistics(node);

    // Store in database
    if (!database_->Store(node)) {
        throw std::runtime_error("Failed to store meta-pattern in database");
    }

    return meta_id;
}

void PatternCreator::SetInitialActivationThreshold(float threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("threshold must be in range [0.0, 1.0]");
    }
    default_activation_threshold_ = threshold;
}

void PatternCreator::SetInitialConfidence(float confidence) {
    if (confidence < 0.0f || confidence > 1.0f) {
        throw std::invalid_argument("confidence must be in range [0.0, 1.0]");
    }
    default_initial_confidence_ = confidence;
}

void PatternCreator::InitializeStatistics(PatternNode& node) {
    // Statistics are already initialized in PatternNode constructor
    // This method is here for future extensibility if we need custom initialization

    // Set base activation (starts at 0)
    node.SetBaseActivation(0.0f);

    // Confidence is set by caller or defaults
    // Access count starts at 0 (handled by constructor)
}

PatternID PatternCreator::GeneratePatternID() {
    // Get all existing pattern IDs
    auto all_ids = database_->FindAll();

    if (all_ids.empty()) {
        return PatternID(1);  // Start from 1
    }

    // Find maximum ID
    uint64_t max_id = 0;
    for (const auto& id : all_ids) {
        max_id = std::max(max_id, id.value());
    }

    return PatternID(max_id + 1);
}

} // namespace dpan
