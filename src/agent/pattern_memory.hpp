// File: src/agent/pattern_memory.hpp
//
// The DPAN side of the agent (docs/AGENT_PLAN.md, section 3). Fused states
// are stored as patterns: each pattern is a prototype (a point in a fixed
// random projection of the fused vector) with a PatternID. A new state joins
// its nearest pattern when it is much closer than the typical distance
// between states, otherwise it becomes a new pattern; when the memory is
// full the least recently used pattern is replaced.
//
// Patterns collect what feedback says about the states they hold (class
// weights, target means) and the input values seen with them. Transitions
// between consecutive patterns are kept as associations in an
// AssociationLearningSystem, with strengths that approximate transition
// probabilities; associations from states to action patterns are
// strengthened or weakened by the sign of the rewards the actions receive.
// Predictions are k-nearest-pattern votes (over all patterns
// or only recently labelled ones) and association sequence predictions.
//
// Not thread-safe.
#pragma once

#include "association/association_learning_system.hpp"
#include "core/types.hpp"
#include "memory/utility_calculator.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace dpan {
namespace agent {

class PatternMemory {
public:
    struct Config {
        size_t dimension{64};         ///< Random projection size
        size_t capacity{10000};       ///< Maximum number of patterns
        size_t memory_budget_bytes{0};  ///< Bound on MemoryBytes() (0 = capacity only)
        size_t recency_half_life{10000};  ///< Steps after which a pattern's recency score halves
        float prune_fraction{0.05f};  ///< Share of patterns removed when the store is full
        float merge_ratio{0.5f};      ///< Join a pattern closer than merge_ratio x typical distance
        float label_rate{0.1f};       ///< Minimum update rate of class weights and target means
        float transition_rate{0.1f};  ///< Update rate of transition strengths
        uint32_t seed{1};
    };

    /// @param input_size Fused vector length
    /// @param class_outputs Number of CLASS outputs
    /// @param value_sizes Size of each VALUE output
    /// @param observed_size Length of the input values remembered per pattern
    PatternMemory(size_t input_size, size_t class_outputs, std::vector<size_t> value_sizes,
                  size_t observed_size, const Config& config);

    /// Project a fused vector into pattern space
    std::vector<float> Project(const std::vector<float>& fused) const;

    struct Placement {
        PatternID id;
        float nearest_distance;  ///< Before placement; infinity if memory was empty
        bool created;
    };

    /// Store a projected state: join the nearest pattern or create one
    /// @param observed Input values of the state (NaN = missing)
    Placement Insert(const std::vector<float>& point, uint64_t step, const std::vector<float>& observed);

    /// Distance from `point` to the nearest pattern (infinity if empty)
    float NearestDistance(const std::vector<float>& point) const;

    void AddLabel(PatternID id, size_t class_output, size_t class_index);
    void AddNegative(PatternID id, size_t class_output, size_t class_index);
    void AddValue(PatternID id, size_t value_output, const std::vector<float>& values);

    /// Class probabilities voted by the k nearest patterns with labels for
    /// this output (nullopt if none have labels). horizon > 0 limits the
    /// vote to patterns that received one of the output's last `horizon`
    /// labels (a short-term memory); 0 = all patterns.
    std::optional<std::vector<float>> ClassVote(const std::vector<float>& point, size_t class_output,
                                                size_t class_count, size_t k, size_t horizon = 0) const;

    /// Votes of the short-term (last `horizon` labels) and long-term (all
    /// labelled patterns) memory, from one scan
    struct Votes {
        std::optional<std::vector<float>> short_term;
        std::optional<std::vector<float>> long_term;
    };
    Votes ClassVotes(const std::vector<float>& point, size_t class_output, size_t class_count, size_t k,
                     size_t horizon) const;
    Votes ValueVotes(const std::vector<float>& point, size_t value_output, size_t k, size_t horizon) const;

    /// Target means voted by the k nearest patterns with targets (horizon
    /// as for ClassVote)
    std::optional<std::vector<float>> ValueVote(const std::vector<float>& point, size_t value_output,
                                                size_t k, size_t horizon = 0) const;

    /// Strengthen the transition from -> to, weaken from's other transitions
    void RecordTransition(PatternID from, PatternID to);

    /// Input values expected next: the remembered values of the patterns
    /// that most often follow `current`, weighted by transition strength
    std::optional<std::vector<float>> PredictNextObserved(PatternID current) const;

    /// Strengthen (reward > 0) or weaken (reward < 0) the association
    /// between a stored state and an action pattern. New associations start
    /// at 0.5. Action patterns are IDs chosen by the caller; they are not
    /// stored states.
    void ReinforceAction(PatternID state, PatternID action, float reward);

    /// Association strength from `state` to each action (0.5 if none)
    std::vector<float> ActionStrengths(PatternID state, const std::vector<PatternID>& actions) const;

        /// Utility of a stored pattern (memory manager score: accesses,
    /// recency in steps, association strength, label purity)
    float Utility(PatternID id, uint64_t step) const;

    /// Estimated memory use: patterns, projection and associations
    size_t MemoryBytes() const;

    /// Write / restore all patterns, associations and counters. LoadState
    /// expects a memory constructed with the same sizes and config.
    /// @throws std::runtime_error on malformed data
    void SaveState(std::ostream& out) const;
    void LoadState(std::istream& in);

    /// Largest pattern ID in use (0 if none)
    PatternID::ValueType MaxPatternId() const;

    /// Change the byte budget (takes effect at the next insertion)
    void SetMemoryBudget(size_t bytes) { config_.memory_budget_bytes = bytes; }

    /// Patterns removed for capacity or budget so far
    uint64_t Pruned() const { return pruned_; }

    bool Contains(PatternID id) const { return slots_.count(id) > 0; }
    size_t Size() const { return slots_.size(); }
    float TypicalDistance() const { return typical_distance_; }
    const AssociationLearningSystem& Associations() const { return associations_; }

private:
    struct Prototype {
        PatternID id;
        std::vector<float> center;
        uint32_t count{0};
        uint64_t last_used{0};
        std::vector<std::vector<float>> class_weights;  // per CLASS output
        std::vector<uint32_t> class_counts;
        std::vector<uint64_t> class_serials;             // label serial of the last label
        std::vector<std::vector<float>> value_means;     // per VALUE output
        std::vector<uint32_t> value_counts;
        std::vector<uint64_t> value_serials;
        std::vector<float> observed;                     // NaN = not seen
    };

    static float Distance(const std::vector<float>& a, const std::vector<float>& b);
    static float SquaredDistance(const float* a, const float* b, size_t n, float bound);
    std::optional<std::vector<float>> ClassFromNeighbors(const std::vector<std::pair<size_t, float>>& neighbors,
                                                         size_t class_output, size_t class_count) const;
    std::optional<std::vector<float>> ValueFromNeighbors(const std::vector<std::pair<size_t, float>>& neighbors,
                                                         size_t value_output) const;

    /// The k nearest prototypes for two filters, from one scan
    template <typename UsableA, typename UsableB>
    std::pair<std::vector<std::pair<size_t, float>>, std::vector<std::pair<size_t, float>>>
    NearestTwo(const std::vector<float>& point, size_t k, UsableA usable_a, UsableB usable_b) const;
    float Utility(const Prototype& p, uint64_t step) const;
    size_t PrototypeBytes(const Prototype& p) const;
    bool OverBudget() const;
    void Prune(uint64_t step);
    void AfterAssociationWrites(size_t writes);
    Prototype* Find(PatternID id);
    void Remove(size_t slot);

    /// Nearest prototypes satisfying `usable`, nearest first
    template <typename Usable>
    std::vector<std::pair<size_t, float>> Nearest(const std::vector<float>& point, size_t k,
                                                  Usable usable) const;

    Config config_;
    size_t input_size_;
    size_t class_outputs_;
    std::vector<size_t> value_sizes_;
    size_t observed_size_;
    std::vector<float> projection_;  // input_size x dimension
    std::vector<Prototype> prototypes_;
    std::unordered_map<PatternID, size_t, PatternID::Hash> slots_;
    std::vector<uint64_t> class_serial_;  // labels received per CLASS output
    std::vector<uint64_t> value_serial_;  // targets received per VALUE output
    float typical_distance_{0.0f};
    bool has_typical_distance_{false};
    AssociationLearningSystem associations_;
    UtilityCalculator utility_;
    size_t prototype_bytes_{0};          // running total of PrototypeBytes over stored patterns
    size_t writes_since_compact_{0};     // association updates leave retired edges until Compact()
    uint64_t pruned_{0};
};

}  // namespace agent
}  // namespace dpan
