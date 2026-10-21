// File: src/association/prediction_scorer.hpp
#pragma once

#include "core/types.hpp"

namespace dpan {

/// Learned (or rule-based) scoring of prediction candidates
///
/// AssociationLearningSystem::SetPredictionScorer() blends a scorer's output
/// with association strength when ranking predictions:
///   score = (1 - weight) * strength + weight * Score(...)
///
/// Implementations must be thread-safe: Score() may be called concurrently.
class PredictionScorer {
public:
    virtual ~PredictionScorer() = default;

    /// Likelihood in [0, 1] that `candidate` follows `source`
    /// @param association_strength Strength (context-adjusted if a context
    ///        is given) of the source -> candidate association
    /// @param context Current context, or nullptr
    virtual float Score(PatternID source,
                        PatternID candidate,
                        float association_strength,
                        const ContextVector* context) const = 0;
};

}  // namespace dpan
