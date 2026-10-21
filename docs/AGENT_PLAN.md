# Plan: `dpan::Agent`, a drop-in learner

Goal: one component that a host program can add without designing a
network, and that learns from whatever combination of inputs, feedback and
outputs its environment provides.

Decisions this plan is based on:

- The host program declares each channel's type; the agent infers sizes,
  ranges and vocabularies.
- Outputs are predictions, classes and discrete actions; continuous actions
  are a later phase.
- CPU only, inputs up to about 64×64 images and short text, optional BLAS.
- C++ on Linux first, then a C API and Python bindings, then Windows and
  macOS.

## 1. Interface

The host program describes its environment once, then streams observations
and sends feedback when it has some.

```cpp
namespace dpan::agent {

enum class InputType    { NUMERIC, CATEGORICAL, TEXT, IMAGE, TIME_SERIES };
enum class FeedbackType { LABEL, TARGET, REWARD, OUTCOME };
enum class OutputType   { CLASS, VALUE, ACTION, NEXT_OBSERVATION, ANOMALY };

struct InputSpec    { std::string name; InputType type; size_t size = 0;            // 0 = infer from data
                      std::vector<std::string> categories;                          // empty = open vocabulary
                      size_t width = 0, height = 0, image_channels = 1; };
struct FeedbackSpec { std::string name; FeedbackType type; std::string output; };    // which output it trains
struct OutputSpec   { std::string name; OutputType type;
                      std::vector<std::string> classes_or_actions; size_t size = 1; };

struct EnvironmentSpec { std::vector<InputSpec> inputs; std::vector<FeedbackSpec> feedback;
                         std::vector<OutputSpec> outputs; };

using Value = std::variant<double, std::vector<float>, std::string, Image>;          // per channel

class Agent {
public:
    explicit Agent(EnvironmentSpec spec, AgentConfig config = {});
    StepId Observe(const std::map<std::string, Value>& observation);                 // missing channels allowed
    Prediction Predict(StepId step, const std::string& output) const;                // value + confidence
    std::string Act(StepId step, const std::string& action_output);                  // discrete action
    void Feedback(StepId step, const std::string& feedback, const Value& value);    // may arrive later
    void EndEpisode();
    bool Save(const std::string& path) const;  static Agent Load(const std::string& path);
    AgentStats Stats() const;   void SetClock(std::shared_ptr<Clock>);   void SetLogger(Logger);
};
}
```

- `Agent` is the only class a host program needs. `StepId` links feedback
  that arrives later to the observation it concerns.
- The spec is validated when the agent is constructed: every output must be
  reachable from a feedback channel, except NEXT_OBSERVATION and ANOMALY,
  which are self-supervised.

## 2. Input type → encoder

Each input channel gets an encoder. Encoder outputs are concatenated into
one fused vector that feeds a shared network trunk.

| Input type | Encoder |
|---|---|
| NUMERIC (scalar or vector) | Running mean and variance normalization, clipped to ±5σ. Missing values become 0 plus a "missing" indicator. |
| CATEGORICAL, known set | One-hot vector |
| CATEGORICAL, open vocabulary | Feature hashing into 64 buckets (configurable), followed by a learned embedding layer |
| TEXT | Lowercase. Word and character tri-gram features hashed into 1,024 buckets, normalized, then a learned embedding layer. |
| TIME_SERIES | A window of the last 16 values per series, plus their differences and summary statistics |
| IMAGE (up to 64×64) | Scale to [0,1], then convolution and pooling layers (Phase 6). Until then: downsample to 16×16 and flatten. |

- **Trunk size:** 2 Dense+ReLU layers whose width is twice the fused size,
  kept between 32 and 512.
- **What gets inferred:** sizes left at 0, value ranges and vocabularies are
  learned from the first observations. Categories are not guessed, because
  the host declares each channel's type.

## 3. Feedback type → learner

Each output gets its own head on the shared trunk. A head is trained only
when its feedback arrives; heads without feedback for a sample contribute no
loss.

| Feedback → output | Neural learner | DPAN component | Combined prediction |
|---|---|---|---|
| LABEL → CLASS | Softmax head, cross-entropy loss | Fused states stored as patterns; vote by the k nearest patterns' labels (fast search) | Mix of the two, weighted by each one's recent accuracy |
| TARGET → VALUE | Regression head, MSE loss | Average of the k nearest patterns' targets | Same weighting, by recent error |
| REWARD → ACTION | Double DQN: Q-value head, target network synced every 500 steps, ε-greedy exploration falling from 1.0 to 0.05 | Associations between states and actions, reinforced by reward sign | Neural Q-values; associations break ties |
| OUTCOME (correct/incorrect) | A correct outcome counts as the label for the earlier prediction; an incorrect one is a negative example for the predicted class | `Reinforce(correct)` on associations; the neural prediction scorer is trained on outcomes | As above |
| No feedback → NEXT_OBSERVATION | Head predicting the next fused state, MSE loss | Association sequence prediction | Neural head, with an association fallback |
| No feedback → ANOMALY | Autoencoder reconstruction error | Distance to the nearest stored pattern | Score converted to a percentile against a running baseline |

**Cold start:** until a head has seen 200 samples (configurable), outputs
come from the DPAN side (nearest patterns and associations). Every
prediction carries a confidence value.

## 4. Stream learning rules

1. **Each `Observe`:** encode the inputs and update the normalization
   statistics; store the fused state as a pattern and record the activation
   for association learning; hold the step in a pending buffer until its
   feedback arrives; expire a pending step after `feedback_timeout`
   (default 1,000 steps).
2. **Trainable samples go to the replay buffer:** supervised feedback makes
   a step trainable when it arrives, self-supervised learning on the next
   observation, rewards as (state, action, reward, next state) transitions.
   The buffer holds 50,000 samples by default, drops low-utility samples
   first (memory manager utility scores), keeps classes balanced, and keeps
   recent samples alongside a random sample of older ones.
3. **Training cadence:** every 8 observations, one mini-batch of 32, half
   newest samples and half replay. Gradient clipping (norm 5), then Adam
   with a learning rate that drops when progress plateaus.
4. **Mixed channels:** losses from all heads with samples are summed, each
   divided by its running average so no head dominates.
5. **Concept drift:** a Page-Hinkley test on each head's loss. On
   detection: temporarily raise the learning rate, raise the share of new
   samples, and lower the weight of nearest-pattern votes until accuracy
   recovers.
6. **Capacity:** if the loss plateaus while still high after at least
   10,000 samples, widen the trunk in a function-preserving way, up to the
   size limit.
7. **Safety:** a NaN or diverging loss rolls the weights back to the last
   in-memory snapshot (every 1,000 steps); memory budgets bound the replay
   buffer and pattern store; the memory manager prunes patterns.

## 5. Phases

| Phase | Deliverables | Done when |
|---|---|---|
| **0. Groundwork** | A `Clock` interface replacing every `Timestamp::Now()` call. Portable little-endian serialization. Saving optimizer state. Gradient clipping and learning-rate schedules. A public way to add associations (no `const_cast`). CMake options to make SQLite and zlib optional. | A run driven by a fake clock is fully reproducible. Serialization round-trips on a fixed byte-level fixture. The library builds with SQLite and zlib off. |
| **1. Spec and encoders** | `EnvironmentSpec`, `Value`, validation, all encoders except convolutional images | Unit tests per encoder. Sizes and vocabularies are inferred correctly. Missing values are handled. Invalid specs are rejected with clear errors. |
| **2. Agent: supervised and self-supervised** | `Agent`, step IDs, delayed feedback, trunk and heads, replay, training cadence, nearest-pattern blending, NEXT_OBSERVATION and ANOMALY outputs | Reference environments A–D pass. |
| **3. Reinforcement** | Double DQN head, exploration, episodes, reward and outcome handling | Reference environments E–F pass. |
| **4. Robustness and self-tuning** | Drift detection, capacity growth, rollback, resource budgets, metrics and logging hooks | Recovers from an abrupt drift within 1,000 samples. Memory stays within budget over 1 million steps. Injected NaN values are rolled back. |
| **5. Full persistence** | One versioned checkpoint: spec, config, encoder statistics and vocabularies, weights, optimizer state, replay buffer, patterns, associations, co-occurrence data, random-number state | After save and load, predictions match an uninterrupted run exactly (same seed and clock). |
| **6. Images and speed** | Conv2D, max-pooling and flatten layers (gradient-checked). Optional BLAS. Multithreaded batches. | Reference environment G passes. A training step is ≥3× faster with BLAS on a 512-wide trunk. |
| **7. Embedding** | Shared-library build. C API with an opaque handle and the spec as JSON. Python bindings (pybind11, NumPy). Builds tested on GCC and Clang (Linux), then Windows and macOS. | Environments A–F pass through the Python bindings; the C API example runs. |
| **8. Continuous actions** | Actor-critic method (TD3 or SAC style) with action bounds in `OutputSpec` | A continuous-control reference task passes. |

Phases 0–2 give a usable drop-in learner. Phase 3 adds learning from
rewards; phases 4–5 make it reliable for long-running use.

## 6. Reference environments

All agents use the same default `AgentConfig`. Each environment runs as an
integration test with a fixed seed and a fake clock.

| | Environment | Channels | Pass criterion (proposed) |
|---|---|---|---|
| A | Stream classification with drift (3 classes, numeric and categorical inputs) | inputs → LABEL → CLASS | Accuracy ≥ 90% when each sample is scored before training on it, after the first 2,000 samples; recovers within 1,000 samples after the drift |
| B | Stream regression | numeric → TARGET → VALUE | Error ≤ 1.2× that of a model trained offline on the same data |
| C | Time-series forecasting | TIME_SERIES → NEXT_OBSERVATION | Mean squared error ≥ 20% lower than predicting the last value |
| D | Anomaly detection (injected anomalies) | numeric and TEXT, no feedback → ANOMALY | Area under the ROC curve ≥ 0.9 |
| E | 5×5 gridworld | numeric → REWARD → ACTION | Optimal path in ≥ 95% of episodes after 500 episodes |
| F | Cart-pole (simulator included in the tests) | numeric → REWARD → ACTION | Average reward ≥ 195 over 100 episodes, reached within 1,000 episodes |
| G | Small images (synthetic shapes, up to 64×64) | IMAGE and categorical → LABEL → CLASS | Accuracy ≥ 95% |
| H | Mixed: text and numeric inputs, label and outcome feedback | combination | Accuracy ≥ 85% |

## 7. Known limits

- One default configuration will not be optimal for every domain; the
  reference suite shows it is acceptable across tasks, not best in each.
- Learning from rewards needs many more interactions than learning from
  labels.
- Images are limited to small sizes on the CPU.
- The host still declares its channels; the agent learns everything else.

## Status

| Phase | State |
|---|---|
| 0. Groundwork | Complete. Clock (`core/clock.hpp`), little-endian serialization (`core/binary_io.hpp`) with byte-level fixtures, optimizer `SaveState`/`LoadState`, `ClipGradientNorm`, learning-rate schedules, `AssociationLearningSystem::AddAssociation`/`SetAssociationStrength`/`RemoveAssociation`, CMake options `DPAN_WITH_SQLITE` and `DPAN_WITH_ZLIB`. |
| 1. Spec and encoders | Complete. `src/agent/environment.hpp` (spec, `Value`, validation), `src/agent/encoders.hpp` (encoders, `ObservationEncoder`). Notes below. |
| 2–8 | Not started |

### Phase 1 implementation notes

Decisions made where sections 1–2 do not specify:

- Every channel's encoding ends with a missing indicator: 1 when the
  channel is absent, the fraction of NaN elements for NUMERIC and
  TIME_SERIES vectors, else 0.
- CATEGORICAL with known categories has an extra "unknown category" slot.
  A known category may also be given by its index (a number).
- Open-vocabulary categories and text are hashed with FNV-1a plus a
  splitmix64 finalizer (not `std::hash`), so buckets are the same on every
  platform. The learned embedding layers after them belong to the agent's
  network (Phase 2); encoders report the width to use (`EmbeddingSize()`:
  8 for categories, 32 for text).
- TIME_SERIES: each observation adds one value per series. Per series:
  16 standardized values (padded with the oldest value), 15 differences,
  and the window's mean, standard deviation, minimum and maximum; then the
  window fill fraction. A missing value repeats the previous one.
  `ResetEpisode()` clears the windows.
- IMAGE: the pixel range divisor (1, 255, 65535 or the maximum) is inferred
  from the largest pixel seen. Images are converted to the spec's channel
  count and resampled to 16×16 by area averaging. All images of a channel
  must have the same size (declared or inferred from the first one).
- `Encode()` on an encoder is const and returns the encoding the value
  will have once learned from, so prediction-time encoding changes no
  state. `ObservationEncoder::Encode` validates the whole observation
  before learning anything from it.
- An `ObservationEncoder` is not ready (returns no fused vector) until the
  sizes of NUMERIC and TIME_SERIES channels declared with size 0 have been
  seen.
- CLASS outputs may leave their class list empty (learned from labels);
  ACTION outputs must list their actions. OUTCOME feedback may train CLASS
  or ACTION outputs.
- Encoder state is not yet saved (Phase 5).
