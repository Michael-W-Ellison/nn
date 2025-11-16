// File: src/discovery/pattern_creator.hpp
#pragma once

#include "core/pattern_node.hpp"
#include "storage/pattern_database.hpp"
#include <memory>
#include <vector>

namespace dpan {

/// PatternCreator - Creates new patterns in the database
///
/// Handles creation of atomic, composite, and meta patterns with
/// proper initialization of statistics and parameters.
class PatternCreator {
public:
    /// Constructor
    /// @param database Pattern database to store patterns in
    explicit PatternCreator(std::shared_ptr<PatternDatabase> database);

    /// Create a new atomic pattern
    /// @param data Pattern data
    /// @param type Pattern type (default: ATOMIC)
    /// @param initial_confidence Initial confidence score [0, 1]
    /// @return Created pattern ID
    PatternID CreatePattern(
        const PatternData& data,
        PatternType type = PatternType::ATOMIC,
        float initial_confidence = 0.5f
    );

    /// Create a composite pattern from sub-patterns
    /// @param sub_patterns IDs of sub-patterns that compose this pattern
    /// @param composite_data Data representing the composite pattern
    /// @return Created composite pattern ID
    PatternID CreateCompositePattern(
        const std::vector<PatternID>& sub_patterns,
        const PatternData& composite_data
    );

    /// Create a meta-pattern (pattern of patterns)
    /// @param pattern_instances IDs of pattern instances
    /// @param meta_data Data representing the meta pattern
    /// @return Created meta-pattern ID
    PatternID CreateMetaPattern(
        const std::vector<PatternID>& pattern_instances,
        const PatternData& meta_data
    );

    /// Set default activation threshold for new patterns
    /// @param threshold Activation threshold [0, 1]
    void SetInitialActivationThreshold(float threshold);

    /// Set default initial confidence for new patterns
    /// @param confidence Initial confidence [0, 1]
    void SetInitialConfidence(float confidence);

    /// Get default activation threshold
    float GetInitialActivationThreshold() const { return default_activation_threshold_; }

    /// Get default initial confidence
    float GetInitialConfidence() const { return default_initial_confidence_; }

private:
    std::shared_ptr<PatternDatabase> database_;
    float default_activation_threshold_{0.5f};
    float default_initial_confidence_{0.5f};

    /// Initialize statistics for a new pattern node
    void InitializeStatistics(PatternNode& node);

    /// Generate next available pattern ID
    PatternID GeneratePatternID();
};

} // namespace dpan
