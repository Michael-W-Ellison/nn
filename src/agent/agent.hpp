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
// Actions: Act() chooses a discrete action for a step; REWARD (or OUTCOME)
// feedback rewards it. Steps of an episode are linked: the next observed
// step is the next state; EndEpisode(true) marks the last action terminal.
// The network learns Q-values (Double DQN); state -> action associations
// reinforced by reward sign break ties and choose actions during the cold
// start. ActContinuous() chooses the action of a CONTINUOUS_ACTION output
// (values within bounds); a TD3 actor-critic on the fused state learns it.
//
// Save/Load write and read one versioned checkpoint of everything learned.
//
// All public methods are thread-safe (one internal mutex).
#pragma once

#include "agent/encoders.hpp"
#include "agent/environment.hpp"
#include "core/clock.hpp"

#include <cstdint>
#include <functional>
#include <limits>
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
    size_t train_every{8};           ///< New trainable samples between training steps (observations with actions)
    size_t batch_size{64};
    float new_sample_share{0.5f};    ///< Share of each batch taken from the newest samples
    size_t replay_capacity{50000};
    size_t cold_start_samples{200};  ///< Samples a head needs before it is used
    size_t feedback_timeout{1000};   ///< Steps a step stays open for Predict/Feedback
    size_t plateau_interval{200};    ///< Training steps per learning-rate check
    size_t plateau_patience{10};     ///< Checks without improvement before the rate drops
    float plateau_factor{0.5f};

    // Actions (Double DQN)
    // When the spec has ACTION outputs these replace train_every,
    // learning_rate and min_trunk_width: learning from rewards needs more
    // updates per observation and a smaller, steadier step
    size_t action_train_every{1};
    float action_learning_rate{2e-4f};
    size_t action_min_trunk_width{64};
    size_t action_return_steps{3};       ///< Rewards summed before bootstrapping (multi-step returns)
    float discount{0.99f};               ///< Weight of future rewards
    size_t target_sync{500};             ///< Training steps between target network copies
    float exploration_start{1.0f};       ///< Probability of a random action at first...
    float exploration_end{0.05f};        ///< ...falling linearly to this
    size_t exploration_steps{10000};     ///< ...over this many exploring Act calls per output

    // Continuous actions (TD3). The actor and two critics read the fused
    // state; actions are scaled to [-1, 1] inside, to each output's bounds
    // outside. Uses discount, batch_size and action_train_every.
    size_t continuous_hidden{128};                ///< Units per hidden layer (actor and critics)
    float continuous_actor_learning_rate{1e-3f};
    float continuous_critic_learning_rate{1e-3f};
    size_t continuous_random_steps{1000};         ///< First exploring ActContinuous calls: uniform random actions
    float continuous_exploration_noise{0.5f};     ///< Then Gaussian noise of this std (fraction of the half-range)
    float continuous_target_noise{0.2f};          ///< Target policy smoothing noise (fraction of the half-range)
    float continuous_noise_clip{0.5f};
    size_t continuous_policy_delay{2};            ///< Critic updates per actor update
    float continuous_tau{0.005f};                 ///< Rate at which target networks follow the live ones
    /// Actor penalty weight x (|z| - threshold)^2 on each pre-tanh action
    /// output z beyond the threshold; keeps the actor from sticking at a
    /// bound, where the tanh gradient vanishes (0 turns it off)
    float continuous_saturation_penalty{0.001f};
    float continuous_saturation_threshold{2.0f};
    /// TD3 updates per training step (each further one on its own batch from
    /// the replay buffer). Two learn the pendulum (environment I) more
    /// reliably than one, at twice the cost.
    size_t continuous_updates_per_step{2};

    // Concept drift (section 4, rule 5): a Page-Hinkley test on each
    // output's prequential network loss (ANOMALY: reconstruction error;
    // not ACTION). On a detection, until the loss is back near its earlier
    // level or drift_recovery_max_samples have passed: the learning rate is
    // multiplied by drift_learning_rate_factor, drift_new_sample_share of
    // each batch comes from the last drift_recent_window samples, and the
    // pattern side votes with its short-term memory only.
    bool detect_drift{true};
    double drift_delta{0.5};          ///< Tolerated increase, in standard deviations of the loss
    double drift_threshold{20.0};     ///< Accumulated increase that signals a drift
    size_t drift_warmup{50};          ///< Losses observed before testing (after each detection)
    float drift_learning_rate_factor{1.5f};
    float drift_new_sample_share{0.9f};
    size_t drift_recent_window{1000};
    size_t drift_recovery_max_samples{3000};

    // Safety (rule 7): a copy of the networks is kept every
    // snapshot_interval observations while training is healthy; a training
    // step that leaves a weight or loss NaN or infinite, or a loss above
    // divergence_loss, restores it
    size_t snapshot_interval{1000};
    float divergence_loss{1e6f};

    // Capacity (rule 6): when the learning rate would drop at a plateau,
    // after capacity_min_samples samples, while the network does worse than
    // the nearest-pattern vote on some output, the trunk is widened by
    // capacity_growth (up to max_trunk_width) without changing its outputs
    bool grow_capacity{true};
    size_t capacity_min_samples{10000};
    float capacity_growth{1.5f};

    /// Bound on the agent's memory (replay buffer, pattern store, pending
    /// steps, networks). After the fixed parts, 80% goes to the replay
    /// buffer and 20% to the pattern store; each removes its least useful
    /// entries to stay within its share.
    size_t memory_budget_bytes{256u << 20};

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

    /// CLASS outputs: the network's blend weight is 1 / (1 + exp(s x (L_n -
    /// L_p))), with L_n and L_p the running log-loss of the network and of
    /// the pattern memory (-log of the true class's probability, at
    /// quality_rate). With quality_rate 0.02 (about 50 labels), s = 10 gives
    /// the side whose log-loss is 0.1 lower about 73% of the weight, 0.3
    /// lower 95%. 0: weights by squared running accuracy instead.
    float class_blend_sharpness{10.0f};

    /// Recent anomaly scores kept as the percentile baseline
    size_t anomaly_baseline{1000};

    /// Limit for CLASS outputs whose classes are learned from labels
    size_t max_classes{1000};

    // Held-out samples (environments without action outputs): one in
    // holdout_every new trainable samples, chosen by a hash of the raw
    // observation (so a repeated observation is always held out or never),
    // is kept out of training. Every holdout_check_interval training steps
    // the network's loss on the newest holdout_capacity of them is compared
    // with the last check: if it rose, training slows to half as often (up
    // to every max_training_slowdown x train_every samples); if it fell,
    // twice as often again. A drift clears them. 0 turns this off.
    size_t holdout_every{20};
    size_t holdout_capacity{500};
    size_t holdout_check_interval{100};
    size_t holdout_min_samples{50};      ///< Held-out samples needed before a check counts
    size_t max_training_slowdown{8};
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

    /// ACTION: `label` is the greedy action, `classes` the actions,
    /// `probabilities` the chance Act(explore) picks each one, `values` the
    /// Q-values (empty during the cold start)
    ///
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
    /// CLASS: running log-loss of each side (NaN otherwise or if not measured)
    float neural_log_loss{std::numeric_limits<float>::quiet_NaN()};
    float pattern_log_loss{std::numeric_limits<float>::quiet_NaN()};
    float neural_weight{0.0f};  ///< Current blend weight of the neural side
    float exploration{0.0f};    ///< ACTION: current probability of a random action.
                                ///< CONTINUOUS_ACTION: noise std (fraction of the half-range; 1 = uniform random)
    uint32_t drift_events{0};
    bool recovering{false};     ///< Recovering from a detected drift
};

struct AgentStats {
    uint64_t steps{0};
    uint64_t episodes{0};          ///< EndEpisode calls
    uint64_t feedback{0};
    uint64_t ignored_feedback{0};  ///< For steps observed before the inputs could be encoded
    uint64_t training_steps{0};
    size_t pending_steps{0};
    size_t replay_size{0};
    size_t held_out_samples{0};    ///< Samples kept out of training to check the training rate
    size_t training_slowdown{1};   ///< Training every training_slowdown x train_every new samples
    size_t patterns{0};
    uint64_t patterns_pruned{0};
    uint64_t drift_events{0};
    bool recovering{false};
    uint64_t rollbacks{0};          ///< Weights restored from a snapshot
    uint64_t widenings{0};          ///< Trunk widenings
    size_t associations{0};
    size_t memory_bytes{0};         ///< Estimated memory in use
    size_t memory_budget_bytes{0};
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
    /// for Predict and Feedback. Without action outputs, trains once per
    /// train_every new trainable samples (labels, targets, outcomes,
    /// self-supervised samples), so observations without feedback do not
    /// train; with action outputs, every train_every observations.
    /// @throws std::invalid_argument for unknown channels or bad values
    StepId Observe(const Observation& observation);

    /// Prediction for an open step
    /// @throws std::invalid_argument for an unknown output
    /// @throws std::out_of_range if the step is unknown or has expired
    Prediction Predict(StepId step, const std::string& output) const;

    /// Choose an action of an ACTION output for an open step. With explore,
    /// a random action with the current exploration probability, otherwise
    /// the action with the highest Q-value (association strength during the
    /// cold start). Calling it again for the same step and output returns
    /// the action already chosen.
    /// @throws std::invalid_argument for an unknown output or one that is not ACTION
    /// @throws std::out_of_range if the step is unknown or has expired
    std::string Act(StepId step, const std::string& action_output, bool explore = true);

    /// Choose the action of a CONTINUOUS_ACTION output for an open step:
    /// `size` values within the output's bounds. With explore, uniform
    /// random actions for the first continuous_random_steps exploring calls,
    /// then the actor's action plus Gaussian noise; otherwise the actor's
    /// action. Calling it again for the same step and output returns the
    /// action already chosen.
    /// @throws std::invalid_argument for an unknown output or one that is not CONTINUOUS_ACTION
    /// @throws std::out_of_range if the step is unknown or has expired
    std::vector<float> ActContinuous(StepId step, const std::string& action_output, bool explore = true);

    /// Feedback for an open step:
    ///   LABEL    class name (string) or index of a declared class (number)
    ///   TARGET   number or vector of the output's size
    ///   REWARD   number: reward for the action Act (or ActContinuous) chose
    ///            at that step (rewards for one step add up)
    ///   OUTCOME  ACTION and CONTINUOUS_ACTION outputs: reward +1 (nonzero) or -1 (zero).
    ///            CLASS outputs: number: nonzero = the prediction for that step was correct.
    ///            Refers to the class last returned by Predict for the step
    ///            (or the current prediction if Predict was not called).
    ///            A CLASS output trained only by OUTCOME must list its
    ///            classes; it predicts from the first step.
    /// @throws std::invalid_argument for an unknown feedback name or bad value
    /// @throws std::out_of_range if the step is unknown or has expired
    /// @throws std::logic_error for REWARD/OUTCOME on an ACTION (or
    ///         CONTINUOUS_ACTION) output when Act (ActContinuous) was not
    ///         called for the step
    void Feedback(StepId step, const std::string& feedback, const Value& value);

    /// End the current episode: clears time-series windows and sequence
    /// links. terminal = the episode ended in a terminal state, so the last
    /// action's value has no future part; false = it was cut off (e.g. a
    /// time limit), so the last action is not learned from unless its next
    /// state was observed.
    void EndEpisode(bool terminal = true);

    AgentStats Stats() const;

    /// Save everything the agent has learned and its pending steps to one
    /// versioned checkpoint: spec, config, encoder statistics and
    /// vocabularies, networks with optimizer state, replay buffer, patterns,
    /// associations, counters and random state. A loaded agent continues
    /// exactly as the saved one would have. The logger and metrics hook are
    /// not saved.
    /// @return false if the file cannot be written (an existing file is
    ///         only replaced by a complete checkpoint)
    bool Save(const std::string& path) const;
    /// @throws std::runtime_error if writing fails
    void Save(std::ostream& out) const;

    /// @throws std::runtime_error for a missing, malformed or unsupported checkpoint
    static Agent Load(const std::string& path);
    static Agent Load(std::istream& in);

    const EnvironmentSpec& Spec() const;
    /// Settings in effect (with the action_* settings applied when the
    /// spec has ACTION outputs)
    const AgentConfig& Config() const;

    /// Installs `clock` as the process-wide clock (dpan::SetClock); nullptr
    /// restores the system clock
    void SetClock(std::shared_ptr<const Clock> clock);

    /// Called with the agent's state when it logs; must not call the agent
    void SetLogger(Logger logger);

    /// Called with Stats() every `every_steps` observations (outside the
    /// agent's lock, so it may call the agent); nullptr removes it
    void SetMetricsHook(std::function<void(const AgentStats&)> hook, size_t every_steps = 1000);

private:
    friend struct AgentTestAccess;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace agent
}  // namespace dpan
