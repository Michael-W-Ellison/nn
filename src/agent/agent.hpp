// File: src/agent/agent.hpp
//
// dpan::agent::Agent - a drop-in learner (docs/AGENT_PLAN.md).
//
// The host describes its environment once (EnvironmentSpec), then streams
// observations and sends feedback when it has some:
//
//   Agent agent(spec);
//   StepId step = agent.Observe({{"temperature", 21.5}, {"weather", std::string("rain")}});
//   Prediction p = agent.Predict(step, "kind");
//   ...
//   agent.Feedback(step, "label", std::string("b"));   // may arrive later
//
// Each output is predicted by a neural head on a shared trunk and by the
// DPAN pattern memory (nearest patterns, association sequences); the two are
// blended by their recent accuracy. Until a head has seen
// cold_start_samples samples, predictions come from the patterns only.
//
// Supported in this version: CLASS (LABEL, OUTCOME), VALUE (TARGET),
// NEXT_OBSERVATION and ANOMALY outputs. ACTION outputs and REWARD feedback
// are rejected (Phase 3); Save/Load are Phase 5.
//
// All public methods are thread-safe (one internal mutex).
#pragma once

#include "agent/encoders.hpp"
#include "agent/environment.hpp"
#include "core/clock.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dpan {
namespace agent {

using StepId = uint64_t;

struct AgentConfig {
    EncoderConfig encoder;
    uint32_t seed{42};

    // Network and training (section 4 of the plan)
    float learning_rate{3e-3f};
    float min_learning_rate{1e-4f};
    size_t min_trunk_width{32};
    size_t max_trunk_width{512};
    float gradient_clip_norm{5.0f};
    size_t train_every{8};           ///< Observations between training steps
    size_t batch_size{64};
    float new_sample_share{0.5f};    ///< Share of each batch taken from the newest samples
    size_t replay_capacity{50000};
    size_t cold_start_samples{200};  ///< Samples a head needs before it is used
    size_t feedback_timeout{1000};   ///< Steps a step stays open for Predict/Feedback
    size_t plateau_interval{200};    ///< Training steps per learning-rate check
    size_t plateau_patience{10};     ///< Checks without improvement before the rate drops
    float plateau_factor{0.5f};

    // Pattern memory
    size_t pattern_capacity{10000};
    size_t pattern_dimension{64};
    size_t neighbors{5};
    float merge_ratio{0.5f};
    /// The pattern side votes with a long-term memory (all patterns) and a
    /// short-term memory (patterns given one of the output's last
    /// pattern_label_horizon labels or targets) and uses whichever has been
    /// more accurate recently, so it follows changes in the labelling
    size_t pattern_label_horizon{1000};

    /// Rate of the running accuracy/error estimates that weight the blend
    float quality_rate{0.02f};

    /// Recent anomaly scores kept as the percentile baseline
    size_t anomaly_baseline{1000};

    /// Limit for CLASS outputs whose classes are learned from labels
    size_t max_classes{1000};
};

struct Prediction {
    std::string output;
    OutputType type{OutputType::CLASS};

    /// False if nothing has been learned for this output yet (or the
    /// observation could not be encoded); other fields are then empty
    bool available{false};

    /// CLASS: most likely class, all classes and their probabilities
    std::string label;
    std::vector<std::string> classes;
    std::vector<float> probabilities;

    /// VALUE: predicted values. NEXT_OBSERVATION: the next values of the
    /// NUMERIC and TIME_SERIES inputs, in spec order, in input units
    /// (NaN where nothing can be predicted yet).
    std::vector<float> values;

    /// ANOMALY: percentile of this observation's anomaly score among recent
    /// observations (0 = most normal, 1 = most unusual)
    float score{0.0f};

    /// 0-1: class probability, 1 / (1 + recent standardized squared error),
    /// or for ANOMALY how full the percentile baseline is
    float confidence{0.0f};

    /// Share of the neural network in the result (the rest is patterns)
    float neural_weight{0.0f};
};

struct OutputStats {
    std::string name;
    OutputType type{OutputType::CLASS};
    uint64_t samples{0};        ///< Feedback (or self-supervised) samples received
    size_t classes{0};
    /// Running accuracy (CLASS) or standardized squared error (VALUE,
    /// NEXT_OBSERVATION) of each side; NaN if not measured
    float neural_quality{0.0f};
    float pattern_quality{0.0f};
    float neural_weight{0.0f};  ///< Current blend weight of the neural side
};

struct AgentStats {
    uint64_t steps{0};
    uint64_t feedback{0};
    uint64_t ignored_feedback{0};  ///< For steps observed before the inputs could be encoded
    uint64_t training_steps{0};
    size_t pending_steps{0};
    size_t replay_size{0};
    size_t patterns{0};
    size_t associations{0};
    bool network_ready{false};
    size_t fused_size{0};
    size_t trunk_width{0};
    float learning_rate{0.0f};
    std::vector<OutputStats> outputs;
};

enum class LogLevel { DEBUG, INFO, WARNING };
using Logger = std::function<void(LogLevel, const std::string&)>;

class Agent {
public:
    /// @throws std::invalid_argument if the spec or config is invalid or
    ///         uses features this version does not support
    explicit Agent(EnvironmentSpec spec, AgentConfig config = {});
    ~Agent();
    Agent(Agent&&) noexcept;
    Agent& operator=(Agent&&) noexcept;
    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;

    /// Learn from an observation (channels may be missing) and open a step
    /// for Predict and Feedback. Trains every train_every observations.
    /// @throws std::invalid_argument for unknown channels or bad values
    StepId Observe(const Observation& observation);

    /// Prediction for an open step
    /// @throws std::invalid_argument for an unknown output
    /// @throws std::out_of_range if the step is unknown or has expired
    Prediction Predict(StepId step, const std::string& output) const;

    /// Feedback for an open step:
    ///   LABEL    class name (string) or index of a declared class (number)
    ///   TARGET   number or vector of the output's size
    ///   OUTCOME  number: nonzero = the prediction for that step was correct.
    ///            Refers to the class last returned by Predict for the step
    ///            (or the current prediction if Predict was not called).
    ///            A CLASS output trained only by OUTCOME must list its
    ///            classes; it predicts from the first step.
    /// @throws std::invalid_argument for an unknown feedback name or bad value
    /// @throws std::out_of_range if the step is unknown or has expired
    void Feedback(StepId step, const std::string& feedback, const Value& value);

    /// Start a new episode: clears time-series windows and sequence links
    void EndEpisode();

    AgentStats Stats() const;

    const EnvironmentSpec& Spec() const;
    const AgentConfig& Config() const;

    /// Installs `clock` as the process-wide clock (dpan::SetClock); nullptr
    /// restores the system clock
    void SetClock(std::shared_ptr<const Clock> clock);

    void SetLogger(Logger logger);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace agent
}  // namespace dpan
