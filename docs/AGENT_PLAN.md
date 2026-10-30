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
enum class OutputType   { CLASS, VALUE, ACTION, NEXT_OBSERVATION, ANOMALY, CONTINUOUS_ACTION };

struct InputSpec    { std::string name; InputType type; size_t size = 0;            // 0 = infer from data
                      std::vector<std::string> categories;                          // empty = open vocabulary
                      size_t width = 0, height = 0, image_channels = 1; };
struct FeedbackSpec { std::string name; FeedbackType type; std::string output; };    // which output it trains
struct OutputSpec   { std::string name; OutputType type;
                      std::vector<std::string> classes_or_actions; size_t size = 1;
                      std::vector<float> low, high; };                              // CONTINUOUS_ACTION bounds

struct EnvironmentSpec { std::vector<InputSpec> inputs; std::vector<FeedbackSpec> feedback;
                         std::vector<OutputSpec> outputs; };

using Value = std::variant<double, std::vector<float>, std::string, Image>;          // per channel

class Agent {
public:
    explicit Agent(EnvironmentSpec spec, AgentConfig config = {});
    StepId Observe(const std::map<std::string, Value>& observation);                 // missing channels allowed
    Prediction Predict(StepId step, const std::string& output) const;                // value + confidence
    std::string Act(StepId step, const std::string& action_output);                  // discrete action
    std::vector<float> ActContinuous(StepId step, const std::string& action_output); // continuous action
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
| CATEGORICAL, open vocabulary | 1,024 slots (configurable): one per category in order of first appearance, hashing only once all slots are taken; followed by a learned embedding layer |
| TEXT | Lowercase. Word and character tri-gram features hashed into 1,024 buckets, normalized, then a learned embedding layer. |
| TIME_SERIES | A window of the last 16 values per series, plus their differences and summary statistics |
| IMAGE (up to 64×64) | Scale to [0,1] at native size (longer side above `max_image_size`, default 64, is area-resampled down), then convolution and pooling layers to an embedding of `image_embedding_size` (default 64). |

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
| LABEL → CLASS | Softmax head, cross-entropy loss | Fused states stored as patterns; vote by the k nearest patterns' labels (fast search) | Mix of the two, weighted by each one's recent log-loss (`class_blend_sharpness`; recent accuracy until the log-loss blend change below) |
| TARGET → VALUE | Regression head, MSE loss | Average of the k nearest patterns' targets | Same weighting, by recent error |
| REWARD → ACTION | Double DQN: Q-value head (Huber loss), target network synced every 500 training steps, 3-step returns cut at exploratory actions, ε-greedy exploration falling from 1.0 to 0.05 | Associations between states and actions, reinforced by reward sign | Neural Q-values; associations break ties and choose during the cold start |
| REWARD → CONTINUOUS_ACTION | TD3: deterministic actor and two critics on the fused state, two updates per training step, target policy smoothing, actor and target updates every 2nd critic update; uniform random actions for the first 1,000 exploring steps, then Gaussian noise (std 0.5 of the half-range); 2 × 128 hidden units | — | The actor's action, scaled to the bounds |
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
3. **Training cadence:** every 8 new trainable samples (a step's first
   label, target or outcome, or a self-supervised sample; observations
   without feedback add none), one mini-batch of 64, half newest samples
   and half replay. Gradient clipping (norm 5), then Adam with a learning
   rate that drops when progress plateaus. When the environment has ACTION
   or CONTINUOUS_ACTION outputs: every observation, learning rate 2e-4,
   trunk at least 64 wide; Q-value losses do not drive the plateau check.
   (Every 8 observations in all environments until the "training follows
   new samples" change below.) Without action outputs, one sample in 20
   (by a hash of the raw observation) is held out of training, and the
   training interval doubles (up to 8×) when the network's loss on the
   held-out samples rises between checks and halves when it falls (see
   "Change: held-out check of the training rate" below).
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
| I | Pendulum swing-up (Pendulum-v1 dynamics, simulator included in the tests) | numeric → REWARD → CONTINUOUS_ACTION | Mean return of 20 greedy episodes ≥ −200 after 200 training episodes |

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
| 2. Agent: supervised and self-supervised | Complete. `src/agent/agent.hpp`. Reference environments A–D pass (results and notes below). |
| 3. Reinforcement | Complete. `Agent::Act`, REWARD and OUTCOME feedback for ACTION outputs, episodes. Reference environments E–F pass (results and notes below). |
| 4. Robustness and self-tuning | Complete. Drift detection and recovery, trunk widening, rollback, memory budgets, metrics hook (results and notes below). |
| 5. Full persistence | Complete. `Agent::Save` / `Agent::Load`: one versioned checkpoint; a loaded agent continues exactly as the saved one (results and notes below). |
| 6. Images and speed | Complete. `Conv2D`, `MaxPool2D`, `Flatten` (`src/nn/convolution.hpp`), optional BLAS and multithreaded matrix products (`src/nn/compute.hpp`), convolutional image embeddings in the agent. Reference environment G passes (results and notes below). |
| 7. Embedding | Complete on Linux (GCC and Clang). C API shared library `dpan_c` (`src/capi/dpan_c.h`), JSON forms of spec, config, values, predictions and statistics (`src/agent/spec_json.hpp`), Python package `dpan` (`python/`). A–F pass through the Python bindings; the C example runs. Windows and macOS are not built or tested (results and notes below). |
| 8. Continuous actions | Complete. `CONTINUOUS_ACTION` outputs with bounds (`OutputSpec::low` / `high`), learned with TD3 (`src/agent/actor_critic.hpp`); `Agent::ActContinuous`, `dpan_agent_act_continuous`, Python `act_continuous`. Reference environment I passes (results and notes below). |
| 9. Memory | Complete. A GRU after the trunk (`memory_units`, off by default; `src/nn/recurrent.hpp`), trained on windows of steps with backpropagation through time. Memory tasks M1, M2 and M4 pass at the defaults, M3 with its own window settings (results below). |
| 10. Sentence meaning | Complete. TEXT channels add a word sequence (first-appearance vocabulary, optional GloVe vectors) read by two GRUs, one in each direction, in the network (`src/agent/word_sequence.hpp`). T1–T3 pass for seeds 1–3; T1 with GloVe uses an ensemble of 4 agents (`ensemble_size`) and, changed after the first runs, the mean over 3 sentence orders (results below). |

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

### Phase 2 results

Reference environments (`tests/agent/reference_environments_test.cpp`,
default `AgentConfig`, fixed seeds, `ManualClock`):

| | Criterion | Result |
|---|---|---|
| A | Accuracy ≥ 90%: samples 2,000–6,000; the 200 samples ending 1,000 after the drift; samples 7,000–9,000 | 98.7%, 97.5%, 98.0% |
| B | Test error ≤ 1.2× an offline model of the same shape (Adam, up to 300 epochs, early stopping) | 1.14× |
| C | Next-value error ≤ 0.8× the last-value baseline (steps 3,000–6,000) | 0.11× (MSE 0.016 vs 0.148) |
| D | Area under the ROC curve ≥ 0.9 (2% injected anomalies of three kinds) | 0.97 |

### Phase 2 implementation notes

- **Components.** `AgentNetwork` (per-channel embeddings, trunk, one head
  per output), `AnomalyAutoencoder`, `PatternMemory` (DPAN side),
  `ReplayBuffer`. `Agent` owns them and holds pending steps for delayed
  feedback.
- **Batch size 64** instead of 32: with 32, B reached 1.22× (no blend weight
  between the two sides reached 1.2×); with 64, 1.14×. Decided with the
  user. Default learning rate 3e-3, halved after 10 checks of 200 training
  steps without improvement, not below 1e-4.
- **Pattern memory.** Fused states are projected to 64 dimensions (fixed
  Gaussian random projection) and stored as prototypes with PatternIDs. A
  state joins its nearest prototype when closer than half the running
  typical nearest distance; otherwise it becomes a new prototype; at
  capacity (10,000) the least recently used one is replaced. Search is
  exact (linear scan of the projected prototypes), not LSH. Prototypes are
  not mirrored into a `PatternDatabase` yet (needed for the memory manager
  in Phase 4).
- **Sequences.** Transitions between consecutive prototypes are
  associations in an `AssociationLearningSystem` (CAUSAL edges, set through
  `AddAssociation`/`SetAssociationStrength`, strengths approximating
  transition probabilities); its time-window co-occurrence learning and
  automatic maintenance are not used.
- **Short- and long-term pattern votes.** The pattern side votes with all
  prototypes and with only those given one of the output's last 1,000
  labels, and uses whichever has been more accurate recently (as in
  SAM-kNN). Without this, A did not recover after its drift: old prototypes
  kept their labels. The network does not detect drift yet (Phase 4); after
  A's drift its accuracy is still about 43% 3,000 samples later, and the
  blend relies on the pattern side.
- **Blending.** CLASS: weights proportional to the squared running
  accuracies. VALUE and NEXT_OBSERVATION: inversely proportional to the
  running standardized squared errors. Running estimates use rate 0.02 and
  are measured on each sample before learning from it. Before a head has
  200 samples its weight is 0. (An error-minimizing weight from the error
  covariance was tried; its estimates were too noisy and it did worse.)
- **NEXT_OBSERVATION** predicts the next values of the NUMERIC and
  TIME_SERIES inputs (in input units), not the whole fused state; other
  input types are not predicted. The pattern side predicts the remembered
  input values of the likely next prototypes.
- **ANOMALY** scores are computed when the observation arrives, before the
  agent learns from it: the percentile of the autoencoder reconstruction
  error and of the distance to the nearest prototype, each against the
  last 1,000 scores, averaged once the autoencoder has 200 samples (pattern
  distance only before). The autoencoder bottleneck is a quarter of the
  input size, between 1 and 64.
- **OUTCOME** feedback refers to the class `Predict` last returned for the
  step (or the current prediction). Correct: trained as a label. Incorrect:
  trained as a negative example (loss −log(1 − p)), and the prototype's
  weight for that class is lowered. A CLASS output trained only by OUTCOME
  must list its classes; its network head predicts from the start so that
  outcomes have predictions to refer to.
- **Replay.** Samples are stored sparsely. When the buffer is full, a new
  sample replaces one of 8 random candidates: the one from the most
  frequent class, oldest first. Utility scores from the memory manager are
  not used yet (Phase 4).
- **Not in Phase 2:** ACTION outputs and REWARD feedback (Phase 3, below),
  `Save`/`Load` (Phase 5). `SetClock` sets the
  process-wide clock.

### Phase 3 results

Reference environments E and F (`tests/agent/reference_environments_test.cpp`,
default `AgentConfig`). Because reinforcement learning results vary with
the random seed, both were also run with other seeds (outside the test
suite):

| | Criterion | Test seed | Other seeds |
|---|---|---|---|
| E | Shortest path in ≥ 95 of 100 greedy episodes after 500 training episodes | 97 | 100, 95, 99, 95, 96 (5 of 5 pass) |
| F | 100-episode average reward ≥ 195 within 1,000 episodes | episode 574 | episode 412; best averages 193.3 and 189.4 (1 of 3 pass) |

F's criterion is measured on the training episodes themselves, so 5% of
its actions are random. A–D were run again with the Phase 3 changes and
still pass (A 98.7% / 97.5% / 98.0%, B 1.13×, C 0.11×, D 0.97).

### Phase 3 implementation notes

- **API.** `Act(step, output, explore = true)` chooses an action (the same
  one again for the same step). `Feedback` with REWARD adds to that step's
  reward; OUTCOME on an ACTION output is a reward of +1 or −1. The next
  observed step of an episode is the next state; `EndEpisode(true)` marks
  the last action terminal, `EndEpisode(false)` (a cut-off episode)
  discards it unless its next state was observed. `Predict` on an ACTION
  output gives the greedy action, the action probabilities under the
  current exploration rate, and the Q-values.
- **Settings for action environments (decided with the user).** With the
  Phase 2 settings, E passed on 1 of 6 seeds and F did not learn: an
  episode of cart-pole gave the network about one update per 8 steps.
  When the spec has ACTION outputs: training every observation
  (`action_train_every`), learning rate 5e-4 (`action_learning_rate`; 2e-4 since Phase 6),
  trunk width at least 64 (`action_min_trunk_width`). Q-value losses are
  excluded from the plateau learning-rate check (their targets move by
  design; including them drove the rate to its minimum).
- **Fixed input statistics (all environments).** NUMERIC and TIME_SERIES
  statistics stop changing after 5,000 values per element
  (`EncoderConfig::statistics_limit`). Statistics that kept moving changed
  the encoding of states already in the replay buffer; with raw inputs in a
  test, cart-pole learned much better (best average 182 vs 40).
- **Multi-step returns (decided with the user, beyond the plan).** Plain
  Double DQN solved F in about half the runs. Targets now sum up to 3
  discounted rewards before bootstrapping (`action_return_steps`), stopping
  at a terminal step and before an exploratory action (Watkins's Q(λ)
  cut), so they estimate the greedy policy. Without the cut, F passed on 4
  of 4 seeds but E on 4 of 6. Samples link to the next step's sample with
  a weak reference, so late rewards are included and streams without
  episodes do not keep old samples alive.
- **Pattern side.** Each action has a PatternID; associations from state
  patterns to actions start at 0.5 and move toward 1 or 0 with the sign of
  each reward (rate 0.1). They break exact ties between Q-values and choose
  the greedy action before the Q head has 200 transitions.
- **Loss scaling.** Dividing each head's loss by its running average (rule
  4) is kept as in Phase 2. Normalizing the factors to average 1 was tried
  during Phase 3; it did not help F and stopped OUTCOME-only learning.
- **Cost.** Action environments train once per observation: F's test runs
  about 65,000 steps in about a minute.

### Phase 4 results

`tests/agent/robustness_test.cpp` (default `AgentConfig` except where noted):

| Criterion | Test | Result |
|---|---|---|
| Recovers from an abrupt drift within 1,000 samples | Environment A, labels rotate at sample 6,000 | 1 detection, none before the drift; the network's own accuracy 1,000 samples after: 98.4% (Phase 3: about 43% after 3,000); combined accuracy on samples 800–1,000 after: 99.5% |
| Memory stays within budget over 1 million steps | Numeric and open-vocabulary inputs (a new category almost every step), CLASS, NEXT_OBSERVATION and ANOMALY outputs, 16 MiB budget | Peak estimate 14.9 MB of 16 MB (checked every 10,000 steps); resident memory of the test process 25.0 MB at step 200,000 and the same at step 1,000,000; about 3,000 patterns held, 932,866 pruned over the run. Runs in about 5.5 minutes (label `long`; `ctest -LE long` skips it). |
| Injected NaN values are rolled back | A NaN written into a network weight, and into the anomaly autoencoder | Restored at the next training step (1 rollback each); predictions stay finite; at least 480 of the next 500 predictions correct |

Also tested: a regression whose relation flips (network error 0.007 before,
0.027 1,000 samples after, in squared standard deviations), trunk widening
on a task where the network trails the nearest-pattern vote (32 → 108
units, outputs unchanged by each widening), the metrics hook, and the
Page-Hinkley detector alone (no alarm in 100,000 stationary losses with a
30% error rate; at most 2 with rare 2% errors; an increase from 5% to 60%
errors is found within 30 samples). Reference environments A–F still pass
(A: 98.7% / 99.5% / 98.3%).

### Phase 4 implementation notes

- **Drift detection.** Page-Hinkley on each output's prequential network
  loss (CLASS: error 0/1; VALUE, NEXT_OBSERVATION: squared standardized
  error; ANOMALY: reconstruction error; not ACTION, whose loss changes by
  design). Each loss is standardized with the mean and deviation since the
  last detection (after 50 warm-up losses) and clipped to ±3, so rare
  errors of an accurate model cannot trigger it; δ = 0.5, λ = 20 (λ = 10
  gave a false alarm within 100,000 noisy losses).
- **Recovery** lasts until the output's loss is back within 10% (+0.01) of
  its level before the drift, or 3,000 samples: the learning rate is reset
  to its base value (undoing plateau drops) and multiplied by 1.5, 90% of
  each batch comes from the last 1,000 trainable samples, the pattern side
  votes with its short-term memory only, and the blend's running estimates
  start over. A factor of 3 was measured to be worse than 1 or 1.5 in A.
  The plan's "lower the weight of nearest-pattern votes" is applied to the
  long-term votes: the short-term pattern memory is the part that adapts
  fastest.
- **Pre-drift labels are dropped from replay** (beyond the plan) for the
  drifted CLASS or VALUE output: targets of samples observed more than 50
  steps before the detection. Without this, the network recovered during
  the recovery period and then fell back (accuracy 56–75% with repeated
  detections) once old labels returned with replay. After a drift that
  only changes the input distribution, still-valid labels are lost too and
  the network relearns them.
- **Pre-drift labels and targets are also forgotten by the pattern memory**
  (added with scenario S4): every pattern's class weights or target means
  for the drifted output are cleared, so the pattern side votes only with
  feedback received after the detection (see "S4 change" below).
- **Capacity.** When the learning rate would drop at a plateau, after
  10,000 samples, while the network trails the pattern side by a clear
  margin on some output (2 points of accuracy, or 10% more error) — the
  test used for "loss still high", since the irreducible loss is unknown —
  both trunk layers are widened by 1.5× (Net2WiderNet: new units copy
  existing ones; copies share the outgoing weights in slightly unequal
  shares summing to 1), only if the larger model still fits the memory
  budget. Optimizer state of the network is reset. Widening stops for good
  when the last one did not narrow the gap by at least 10%: in the
  1-million-step test, without this rule the trunk grew to 512 units (7
  widenings) while still trailing, steps became 6 times slower and the
  replay buffer shrank to 11,000 samples to make room.
- **Rollback.** A copy of the networks is kept when they are built and
  every 1,000 observations while their weights are finite. A training step
  that leaves a weight or loss non-finite, or a loss above 10⁶, restores
  the copy (with the target network). A relative divergence test was not
  used: a real drift raises the loss by orders of magnitude.
- **Memory budget** (`memory_budget_bytes`, 256 MiB by default). Fixed
  parts are reserved first: the networks (weights, gradients, two Adam
  moments; live, snapshot and target copies) and the pending steps (with
  an upper bound for each step's sample). The rest goes 80% to the replay
  buffer and 20% to the pattern store, and is split again after a
  widening. The replay buffer measures samples in bytes (again when they
  change) and evicts as in Phase 2 until within its share. The pattern
  store removes the 5% of patterns with the lowest utility from the memory
  module's `UtilityCalculator` (accesses, recency in steps with a 10,000
  step half-life, average association strength, label purity as
  confidence), with their associations; `UtilityCalculator` gained a
  `CalculateUtility` overload that takes these inputs directly. Updated
  associations are retired, not freed, until the matrix is compacted; the
  pattern store compacts it once retired edges outnumber live ones (or
  exceed 10% of its budget). The replay buffer still does not use utility
  scores; its eviction is class-balanced and random among old samples.
- **Speed.** Nearest-pattern search computes both votes (short- and
  long-term) in one scan, takes the anomaly distance from the placement
  scan, and stops summing a distance once it exceeds the current k-th
  best.
- **Hooks.** `SetLogger` (warnings for drifts and rollbacks, information
  for recoveries, widenings, learning-rate drops) and `SetMetricsHook`
  (`Stats()` every N observations, called outside the agent's lock).
  `Stats()` reports memory use against the budget, drift events, recovery,
  rollbacks, widenings and pruned patterns.
- **Testing.** `agent/agent_testing.hpp` (`AgentTestAccess::InjectNaN`)
  writes a NaN into a weight, for tests of the rollback.

### Phase 5 results

`tests/agent/persistence_test.cpp`: an agent runs part of a stream, is
saved and loaded, and both continue on the same data; every prediction,
action, probability, value, score and statistic is compared bit for bit.

| Scenario | Result |
|---|---|
| All input types (numeric with inferred size, known and open categories, text, time series, image), CLASS with learned classes, VALUE, NEXT_OBSERVATION, ANOMALY; labels and targets 7 steps late (pending steps cross the save); OUTCOME feedback; saved after a learning-rate drop | Identical over 1,500 further steps |
| Saved before the input sizes are known | Identical over 600 steps |
| Saved while recovering from a drift | Identical until step 4,500 (about 2,000 further steps) |
| Reinforcement learning (gridworld), saved in the middle of an episode, with exploration | Identical over 3,000 steps |

The tests detect missing state: with the agent's random state, the replay
buffer's random state, the associations or the numeric input statistics
left out of loading, at least one scenario fails. The learning-rate
schedule's state round-trips exactly in a unit test; in these streams a
fresh schedule happens to reach the same state at the next check. Load also
rejects a wrong file type, an unsupported version and truncated files
(`std::runtime_error`); `Save` returns false when the file cannot be
written and replaces an existing checkpoint only with a complete one
(written to a temporary file, then renamed).

### Phase 5 implementation notes

- **Format.** `DPAG`, version 1 (little-endian, `core/binary_io.hpp`),
  then the spec, the configuration in effect (all fields), and the state:
  encoder statistics, windows and vocabularies; per output the learned
  classes, action pattern IDs, target statistics, running quality
  estimates, drift detector and recovery state; the network, target
  network and snapshot (weights, head layout, Adam state of every
  optimizer), the autoencoder and its snapshot; the pattern store
  (projection, patterns in store order, label serials, counters, every
  association with its type and strength); every sample once, with the
  replay buffer, newest queue, recent window, pending steps and next-step
  links referring to it by number, so shared samples stay shared; the
  learning-rate schedule, counters, anomaly baselines, and the full state
  of both random-number generators. The logger and metrics hook are not
  saved. The plan's "co-occurrence data" is the association store: the
  agent does not use the co-occurrence tracker (Phase 2 notes).
- **Exactness.** Three changes made behaviour depend only on saved state:
  ties between transition strengths are broken by pattern position, not
  by the association store's internal order (which depends on pattern ID
  values; IDs come from a process-wide counter and differ between a run
  and a copy loaded in the same process); a pattern's average association
  strength is summed in sorted order; memory estimates count element
  sizes, not vector capacities (which differ after loading). Results of
  A–F and the Phase 4 tests are unchanged, except the million-step test:
  peak estimate still 14.9 MB of 16 MB; resident memory 26.3 MB at step
  200,000 and at step 1,000,000 (the replay buffer now holds about 31,000
  samples instead of 29,000). Loading reserves pattern IDs up
  to the largest one restored, so new patterns cannot reuse them in
  another process.
- **Supporting changes.** `LearningRateSchedule` gained `SaveState` /
  `LoadState` (state of `ReduceOnPlateau`); encoders, `AgentNetwork`,
  `AnomalyAutoencoder`, `PatternMemory`, `Sample` and `ReplayBuffer` gained
  serialization.
- **Size.** A checkpoint holds the replay buffer, so it is roughly the
  memory the agent uses (bounded by `memory_budget_bytes`).

### Phase 6 results

| Check | Result |
|---|---|
| G: 32×32 grey images of four shapes (square, circle, triangle, plus; size 4–9 pixels, random position, Gaussian noise) and a day/night category (night images darker and noisier) → CLASS; 5,000 samples, accuracy on the last 1,000 | 0.969 at the test seed (71); 0.970, 0.966, 0.973 at seeds 72–74 |
| Training step, 257 inputs → trunk 512, batch 64, median of 3 × 20 steps (`tests/agent/performance_test.cpp`) | 18.9 ms without BLAS on 1 thread, 8.92 ms without BLAS on 4 threads, 3.27 ms with OpenBLAS: 5.8× and 2.7× faster. The test requires ≥ 3× against 1 thread and is skipped without BLAS. |
| Gradient checks (`tests/nn/convolution_test.cpp`) | `Conv2D` (padding 1 at stride 1 and 2), `MaxPool2D` (non-overlapping and overlapping) agree with central differences |
| Default build (no BLAS) | All 1,930 tests pass. Before the learning-rate change below, A–F gave the same results as before Phase 6. |
| Build with `DPAN_WITH_SQLITE=OFF`, `DPAN_WITH_ZLIB=OFF` | All tests pass (long, reference, robustness and performance labels excluded) |
| ThreadSanitizer | No reports in the compute, convolution and agent tests |
| BLAS build | All 1,930 tests pass (with the learning-rate change below). |

**E stability and the action learning rate.** With
`action_learning_rate` 5e-4 (the Phase 3 setting), the BLAS build failed E
at the test seed (93 of 100). E was close to its threshold in both builds;
greedy episodes on the shortest path, out of 100, at seeds 51 (the test
seed) and 1–11:

| Build | 51 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No BLAS | 97 | 100 | 95 | 99 | 95 | 96 | 100 | 98 | 99 | 93 | 97 | 92 |
| BLAS | 93 | 98 | 96 | 100 | 99 | 97 | 99 | 100 | 94 | 90 | 94 | 89 |

(BLAS sums in a different order, so results differ in the last bits and
training follows a different path.) The missed episodes contain moves into
a wall or away from the goal. Their Q-values are 0.05–0.1 below the best
move's, and the network's errors at learning rate 5e-4 exceed that. Other
settings, over the same 12 seeds and both builds (lowest / mean result):

| Setting | No BLAS | BLAS |
|---|---|---|
| Phase 3 settings | 92 / 96.8 | 89 / 95.8 |
| `target_sync` 250 / 1000 | 90 / 96.5, 84 / 93.3 | 82 / 96.4, 77 / 89.8 |
| `exploration_steps` 5,000 / 20,000 | 71 / 91.5, 87 / 96.2 | 71 / 92.8, 90 / 96.9 |
| `exploration_end` 0.02 / 0.1 | 68 / 93.8, 86 / 96.6 | 82 / 96.6, 92 / 96.8 |
| `action_return_steps` 1 | 83 / 92.9 | 59 / 91.9 |
| `action_learning_rate` 3e-4 | 93 / 97.7 | 95 / 98.1 |
| `action_learning_rate` 2.5e-4 | 95 / 98.4 | 93 / 98.1 |
| `action_learning_rate` 2e-4 | 95 / 98.3 | 97 / 99.0 |
| `action_learning_rate` 1.5e-4 | 98 / 99.1 | 92 / 98.8 |

The default `action_learning_rate` is now **2e-4**. At seeds 12–23 it gives
lowest 98 / mean 99.5 (no BLAS) and 98 / 99.3 (BLAS); 5e-4 gave 86 / 97.6
and 89 / 97.3. Test seeds after the change:

| | No BLAS | BLAS |
|---|---|---|
| E (seed 51) | 100 of 100 (was 97) | 99 of 100 (was 93) |
| F (seed 61) | solved at episode 811, best average 195.02 (was 574) | solved at episode 548, best average 195.7 (was 449) |

F learns more slowly at 2e-4. At seeds 61–66 it reaches 195 within 1,000
episodes at 5 of 6 seeds without BLAS and 4 of 6 with BLAS; at 5e-4 it was
3 of 6 in each build. At the test seed without BLAS, F's best average
(195.02) is just above the threshold. A–D and G do not use the action
learning rate; their results are unchanged.

### Phase 6 implementation notes

- **Layers.** Images are rows in height × width × channels order (the
  layout of `ImageEncoder`). `Conv2D(shape, filters, kernel, stride,
  padding)` uses im2col: the patches of the batch form one matrix, so the
  forward pass, the weight gradient and the input gradient are one matrix
  product each. `MaxPool2D(shape, pool, stride)` records the winning input
  of each output for the backward pass. `Flatten` is an identity that checks
  the row length. All three serialize through `Sequential` and `Layer::Deserialize`.
- **Agent.** Images are no longer resampled to 16×16 by default:
  `EncoderConfig::image_size` 0 keeps the native size, and images whose
  longer side exceeds `max_image_size` (64) are area-resampled to fit;
  `image_size` > 0 resamples to a square as before. Each image channel has
  an embedding branch: up to three blocks of 3×3 convolution (8, 16, 32
  filters) + ReLU + 2×2 max pooling, while the smaller side is at least 8,
  then a dense layer to `image_embedding_size` (64). Text and categorical
  embeddings are unchanged (one dense layer, same initial weights).
- **BLAS.** `cmake -DDPAN_WITH_BLAS=ON` (default OFF) links BLAS through
  CMake's `FindBLAS` (vendor OpenBLAS unless `BLA_VENDOR` is set) and needs
  `cblas.h` (for example the `libopenblas-dev` package). `MatMul`,
  `MatMulTransposeA` and `MatMulTransposeB` then call `cblas_sgemm`.
  `nn::BlasAvailable()` reports whether it was built in; `nn::SetUseBlas`
  switches it off and on at run time.
- **Threads.** Without BLAS, the three products split output rows across
  threads when a product has at least 2²⁰ multiply-adds. Each output element
  is summed in the same order as on one thread, so results do not depend on
  the thread count. `nn::SetThreadCount` / `nn::ThreadCount` (default: the
  hardware thread count, at most 8).
- **Checkpoint.** Version 2: the configuration holds `max_image_size` and
  `image_embedding_size`, and embeddings are stored as layer sequences.
  Version 1 checkpoints are rejected.

### Phase 7 results

| Check | Result |
|---|---|
| A–F through the Python bindings (`python/tests/test_reference.py`; ctest `python.*`, labels `python` and `reference`) | All pass: A 0.9895 / 0.9950 / 0.9850; B ratio 1.139; C ratio 0.119; D 0.971; E 97 of 100; F solved at episode 716 (best average 195.26) |
| C example (`examples/c_api/agent_c_example.c`, compiled as C99; ctest `agent_c_example`) | Runs: classification 1000 of the last 1000 correct; gridworld 20 of 20 greedy episodes on a shortest path after save and load |
| C API tests (`tests/capi/c_api_test.cpp`, linked only to `dpan_c`) | 6 tests pass: status codes and messages, per-thread error message, builder and JSON observations give identical results, save/load gives identical actions and Q-values, images and vector targets |
| Python API tests (`python/tests/test_api.py`) | 6 tests pass |
| JSON tests (`tests/agent/json_test.cpp`) | 10 tests pass, including numbers under a decimal-comma C++ locale |
| Installed package | A C program built with `find_package(DPAN)` and `dpan::c_api` (`examples/consumer/c_main.c`) links only `libdpan_c.so` and runs |
| `libdpan_c.so` exports | The 29 `dpan_*` functions only (linker version script) |
| GCC 13.3 build with `DPAN_BUILD_PYTHON=ON` | All 1,954 tests pass (the million-step test excluded) |
| Clang 18.1 build with `DPAN_BUILD_PYTHON=ON` | All 1,954 tests pass (the million-step test excluded) |
| `BUILD_SHARED_LIBS=ON` | Configures and builds (before this phase it stopped with a CMake dependency-cycle error); 1,934 tests pass (long, reference, robustness and performance labels excluded). `examples/consumer` built with `add_subdirectory` and `BUILD_SHARED_LIBS=ON` runs both programs. |
| Warnings (`-Wall -Wextra -Wpedantic`, GCC and Clang) | None in the new sources, except one from pybind11's `PYBIND11_MODULE` macro under Clang `-Wpedantic` |
| Windows, macOS | Not built or tested |

The Python environments use NumPy's random generator, so their data differ
from the C++ tests'. Run outside the test suite at three other seeds each
(seed + 1, + 2, + 3): B ratios 1.047, 1.100, 1.133 (pass); E 100, 100, 100
(pass); F solved at episode 528, then not solved within 1,000 episodes
(best averages 187.6 and 158.0). F's sensitivity to the seed is the one
recorded in Phase 6.

### Phase 7 implementation notes

- **Shared library.** The module libraries are static (core, storage,
  similarity and discovery depend on each other, which CMake allows only
  between static libraries); they are now declared `STATIC`, so a parent
  project with `BUILD_SHARED_LIBS=ON` can still add DPAN. The shared library
  is the C API, `dpan_c` (`DPAN_BUILD_C_API`, default ON, CMake target
  `dpan::c_api`, versioned `libdpan_c.so.0.1`). It contains the agent and the
  modules it uses; on ELF platforms a version script exports only `dpan_*`.
  It does not depend on SQLite or zlib.
- **C API** (`src/capi/dpan_c.h`). Opaque `dpan_agent` and
  `dpan_observation` handles. The spec and configuration are JSON; an
  observation is built with typed setters (number, vector, text, image) or
  given as JSON; feedback as a number, vector, text or JSON value;
  predictions and statistics are returned as JSON. `dpan_agent_act` returns
  the action's name (owned by the agent) and index. Every function returns
  a `dpan_status`; `dpan_last_error()` holds the message per thread. Strings
  returned through `char**` are freed with `dpan_string_free`.
  `dpan_clock_use_manual` / `dpan_clock_advance` / `dpan_clock_use_system`
  control the process clock.
- **JSON** (`src/agent/json.hpp`, `src/agent/spec_json.hpp`). A small parser
  and writer (no dependency): duplicate members, unknown members and
  unknown type names are errors that name their place (`inputs[2]: unknown
  member "sizes"`); numbers are read and written in the classic locale.
  Config fields are read by their C++ names, with type and range checks.
- **Python** (`DPAN_BUILD_PYTHON`, default OFF). Package `dpan` with the
  extension `dpan._core` (pybind11 2.13.6 from the system or fetched),
  written to `<build>/python/dpan`; `DPAN_PYTHON_INSTALL_DIR` installs it.
  `dpan.Agent(spec, config)` takes dicts or JSON text; observation values
  are numbers, strings, sequences or NumPy arrays (1-D vector, 2-D or 3-D
  image as height × width [× channels]), `None` for a missing channel.
  `Prediction` has attributes; `stats()`, `spec` and `config` are dicts.
  Errors: invalid spec, config or value → `ValueError` or `TypeError`;
  unknown step → `IndexError`; wrong order (e.g. a reward before `act`) →
  `RuntimeError`; `save` failure → `OSError`. The GIL is released while
  the agent works. `use_manual_clock`, `advance_clock(seconds)`,
  `use_system_clock` control the process clock.
- **Tests.** `python.*` tests run with `python3 -m unittest` (no pytest);
  the reference ones have the label `reference`, so `ctest -LE reference`
  skips them as it skips the C++ ones.

### Phase 8 results

| Check | Result |
|---|---|
| I: pendulum swing-up (`tests/agent/reference_environments_test.cpp`, seed 81) | Mean return of 20 greedy episodes −111.3 (threshold −200); last 20 training episodes −176.3; 44,000 steps, about 50 s |
| I in the BLAS build | −110.5 |
| I through the Python bindings (`python.i_pendulum_swing_up`, NumPy data) | −153.9 |
| A–G | The same results as before this phase, in the default and the BLAS build |
| TD3 unit tests (`tests/agent/actor_critic_test.cpp`) | One-step task: the actor is within 0.15 of the best action, and its action is in the top 5% of the critic's values; bootstrapping through the target critics; save/load and copies continue identically; delayed actor updates |
| Agent tests (`tests/agent/continuous_test.cpp`) | Spec validation and JSON, bounds, repeated `ActContinuous` calls, wrong calls rejected, a one-step task learned within 0.6 (action range 0–10), rollback of non-finite weights |
| Persistence (`PersistenceTest.ContinuousActionLearnerContinuesExactlyMidEpisode`) | Saved mid-episode, after the random start: identical over 1,500 further steps. Leaving out the restored pending actions, or changing the restored update counter, makes it fail. |
| C API (`CApiTest.ContinuousActions`), Python API (`test_continuous_actions`) | Pass |
| GCC and Clang builds | All 1,968 tests pass (the million-step test excluded) |
| BLAS build | 1,958 of 1,960 pass before a fix to the TD3 unit test, which then passes. The other failure, `CliBenchmark.DetailedPrediction_Overhead` (a timing benchmark this phase did not change), passed 3 of 3 times when run alone. |

I at other seeds (outside the test suite): C++ seeds 1–16 give −147.1,
−144.1, −154.1, −156.7, −131.5, −171.1, −181.5, −157.1, **−210.6**, −131.7,
−129.9, −134.1, −136.7, −159.9, −168.9, −154.1 (15 of 16 pass); through
Python, seeds 82–84 give −181.1, −186.4, **−473.6** (2 of 3 pass).

### Phase 8 implementation notes

- **Spec.** `OutputType::CONTINUOUS_ACTION` with `size` values and bounds
  `low` / `high` (both empty = [−1, 1]); trained by REWARD or OUTCOME (±1)
  feedback. JSON: `{"type": "continuous_action", "size": n, "low": [...],
  "high": [...]}`.
- **Learner** (`src/agent/actor_critic.hpp`). TD3 on the fused state: actor
  (2 × 64 ReLU, tanh output; 128 units since the S7 change below) and two critics (2 × 64 ReLU on [state,
  action]), each with a target copy; one critic update per training step
  (mean squared error toward reward + discount × min of the target critics
  at the smoothed target action, noise std 0.2 clipped to 0.5); every 2nd
  step the actor follows the first critic's gradient and the targets move
  by τ = 0.005. Adam, learning rate 1e-3 for both; no gradient clipping.
  Two such updates per training step (`continuous_updates_per_step`; the
  second on its own batch from the replay buffer; see "Environment I
  stability" below).
  Actions are in [−1, 1] inside and scaled to the bounds outside. One-step
  returns (the discrete learner's multi-step returns are not used).
- **Acting.** `Agent::ActContinuous(step, output, explore)`: uniform random
  actions for the first `continuous_random_steps` (1,000) exploring calls,
  then the actor's action plus Gaussian noise (std
  `continuous_exploration_noise` = 0.1 of the half-range; 0.5 since the S7
  change below), clipped to the
  bounds; without explore, the actor's action. Calling it again for the
  same step returns the same action. `Act` on such an output (and
  `ActContinuous` on an ACTION output) throws `std::invalid_argument`.
- **Prediction.** `values` is the greedy action, `score` the first critic's
  value of it, `confidence` 1 − the exploration noise (0 during the random
  start). `Stats().outputs[i].exploration` is the noise std (1 during the
  random start). The pattern memory has no part in continuous actions.
- **Settings.** `continuous_hidden`, `continuous_actor_learning_rate`,
  `continuous_critic_learning_rate`, `continuous_random_steps`,
  `continuous_exploration_noise`, `continuous_target_noise`,
  `continuous_noise_clip`, `continuous_policy_delay`, `continuous_tau`,
  `continuous_updates_per_step`, `continuous_saturation_penalty`,
  `continuous_saturation_threshold` (see "S7 seed 4 change" below);
  `discount`, `batch_size` and the action settings (`action_train_every`:
  training every observation) are shared with discrete actions.
- **Safety and memory.** The actor-critics are in the snapshots and the
  rollback; non-finite weights or losses roll back. Their loss is not
  compared with `divergence_loss`, because its scale follows the rewards
  (pendulum Q-values reach about −1,600). Their parameters count toward the
  memory budget (live, target and snapshot copies).
- **Checkpoint.** Version 3: bounds in the spec, the TD3 settings, the
  actor-critics with their optimizer state and update counters, the actions
  in samples and pending steps. Version 4 adds
  `continuous_updates_per_step` to the configuration; version 6 adds the
  saturation settings to the configuration and the actor-critics; earlier
  versions are rejected.
- **APIs.** C: `dpan_agent_act_continuous(agent, step, output, explore,
  values, capacity, count)`; it checks `capacity` before choosing. Python:
  `agent.act_continuous(step, output, explore=True)` returns a list.
- **Supporting change.** `nn::Sequential::Backward` returns the gradient
  with respect to its input (the actor update needs it through the critic).

### Environment I stability

With the Phase 8 settings, I passed at its test seed but not reliably:
over C++ seeds 1–48, 6 runs ended below −200 and 2 of them had greedy
episodes that never swung the pendulum up (returns −960 to −1,690); through
Python, 6 of 24 seeds (81–104) ended below −200. A trace of such an episode
shows the pendulum spinning at the speed limit (−8) with the torque at its
bound (−2) in every step: the greedy policy has a region where it
saturates.

Settings and changes tried (outside the test suite; greedy mean return of
20 episodes after 200 training episodes per seed; "stuck" = greedy episodes
below −800). The first block was run on 16 seeds including the six
failures above, later ones on 24 (those 16 and 22, 27, 42, 12–16):

| Change | Seeds | Runs < −200 | Stuck episodes | Worst |
|---|---|---|---|---|
| Phase 8 settings | 16 | 6 | 8 | −580.7 |
| Batch 128 | 16 | 6 | 11 | −654.5 |
| 128 hidden units | 16 | 2 | 0 | −215.8 |
| Actor learning rate 3e-4 | 16 | 2 | 6 | −432.8 |
| Exploration noise 0.2 | 16 | 1 | 0 | −201.0 |
| Exploration noise 0.3 | 16 | 2 | 3 | −361.0 |
| Random start 2,000 steps | 16 | 1 | 0 | −203.5 |
| Noise 0.2 and random start 2,000 | 16 | 1 | 0 | −209.1 |
| Exploration noise 0.2, all 48 seeds | 48 | 3 | 10 | −690.8 |
| 128 hidden units, 16 further seeds | 16 | 1 | 8 | −749.2 |
| Linear actor output with inverted gradients at the bounds (Hausknecht & Stone 2016) | 24 | 6 | 0 | −248.1 |
| ... with noise 0.2 / batch 128 / 128 units / random start 2,000 / policy delay 1 / learning rates 2e-3 / τ 0.01 / target noise 0.1 | 24 | 4–11 | 0–3 | −226.8 to −400.3 |
| tanh actor with a penalty on its input beyond ±2.5 | 24 | 6 | 5 | −442.9 |
| tanh actor with a minimum gradient of 0.1 toward the inside | 24 | 5 | 10 | −735.2 |
| Greedy actions from the target actor (inverted gradients) | 24 | 5 | 4 | −433.9 |
| Inverted gradients, two updates per step | 24 | 6 | 0 | −258.5 |
| **tanh actor, two updates per step** | 24 | **3** | **0** | **−215.4** |

Inverted gradients removed the stuck episodes in C++ (0 over 48 seeds;
spread of the 48 results 30.9 against 67.6) but lowered the mean (−169.1
against −164.7), and through Python it ended 11 of 24 seeds below −200
(against 6), so it was not adopted. Two updates per step was adopted
(`continuous_updates_per_step` = 2):

| | Phase 8 settings | Two updates per step |
|---|---|---|
| C++ seeds 1–48: runs < −200 | 6 | 4 |
| C++ seeds 1–48: stuck episodes | 8 | 0 |
| C++ seeds 1–48: worst run | −580.7 | −230.9 |
| C++ seeds 1–48: mean, spread (std) | −164.7, 67.6 | −158.9, about 27.8 |
| C++ seeds not used to choose (24): runs < −200, mean | 0, −142.5 | 1, −151.2 |
| Python seeds 81–104: runs < −200 | 6 | 1 |
| Python seeds 81–104: mean, worst | −186.2, −473.6 | −171.2, −511.8 |
| Test seed (C++ 81) | −111.3 | −122.9 (BLAS build −123.0) |
| Time of the I test | about 50 s | about 94 s |

All 1,968 tests pass with the change (the million-step test excluded).

Remaining: runs below −200 are 5 of 72 (C++ and Python together) instead
of 12 of 72; on the C++ seeds that were not used to choose the setting, the
new setting is slightly worse (1 run below −200 against 0); one Python run
(seed 85) still ends at −511.8.

### Scenario tests

`tests/agent/scenarios_test.cpp`: ten uses beyond the reference
environments, each with the default `AgentConfig`, seeds 1–3, and a pass
criterion fixed before the first run. They are skipped unless
`DPAN_SCENARIOS` is set, so the regular suite does not depend on them:
`DPAN_SCENARIOS=1 ctest --test-dir build -L scenarios` (label `scenarios`;
each test prints its metrics).

| | Use | Criterion | Before the changes below | Now |
|---|---|---|---|---|
| S1 | Text classification, 3 classes, keywords with filler words and misspellings | accuracy ≥ 0.90 (last 1,000 of 3,000) | 0.94–0.96 | same |
| S2 | 3 regression outputs, 5 inputs, 10% of input values missing | R² ≥ 0.7 for every output | 0.77–0.89 | same |
| S3 | Plan environment H: text + numeric inputs, labels for 30% of steps, outcomes for the rest, 4 classes; an ANOMALY output alongside | accuracy ≥ 0.85 (last 2,000 of 6,000, non-anomalous steps), anomaly AUC ≥ 0.8 | 0.94–0.95, AUC 0.89–0.92 | same |
| S4 | Regression whose target function changes at step 4,000 | MSE over steps 5,000–6,000 ≤ 2 × MSE over 3,000–4,000 | 0.026–0.028 vs limits 0.024–0.025 (fails) | 0.013–0.014 |
| S5 | 250 open-vocabulary user IDs, 5 imbalanced classes (60/20/10/5/5%), 50 new users from step 4,000 | mean per-class recall ≥ 0.8 (last 2,000 of 8,000) | 0.45–0.55 (fails) | 1.0 |
| S6 | Contextual bandit, 4 contexts, 3 actions, outcomes only | greedy action best in 4/4 contexts after 3,000 steps | 4/4 | same |
| S7 | Continuous 2-D reaching, velocity in [−0.1, 0.1]², absolute positions as input | mean final distance of 20 greedy episodes ≤ 0.1 after 300 training episodes | 0.24–0.28 (fails) | 0.065–0.073 |
| S8 | Image regression: x position of a blob in a 16×16 image | R² ≥ 0.8 | 0.9995–0.9996 | same |
| S9 | Two related time series (the second lags the first by 3 steps) | MSE ≤ 0.5 × last value, both series | 0.066–0.076 | same |
| S10 | Inputs at scale 10⁶ with offsets 10⁷ | accuracy ≥ 0.90 | 0.996–0.998 | same |

S6 passes, but its success rate while exploring is about 0.5 after 3,000
steps: the exploration rate falls over `exploration_steps` (10,000).

**S5 and the category encoding.** S5 failed because open-vocabulary
categories were hashed into 64 buckets, so 250 users shared buckets with
users of other classes (with 1,024 hashed buckets it passed: mean recall
0.90 and 0.91). Categories now get their own slot in the order they first
appear, and hashing is used only once all slots are taken; the default
`category_buckets` is 1,024 (was 64). Supporting changes:

- Wider one-hot inputs made the memory reservation per pending step (a
  dense copy of the fused vector and a dense upper bound for its sample)
  take most of a small budget: in the million-step test (16 MiB, an
  open-vocabulary channel with new categories at every step) the pattern
  store fell to 1 pattern and the replay buffer to 2,184 samples. Pending
  steps no longer keep a dense copy (the sample holds the fused vector
  sparsely), and the reservation counts at most `MaxNonZero()` values per
  sample (1 for a one-hot channel). The test now ends with 2,010 patterns
  and 22,425 replay samples (2,958 and 28,868 with 64 buckets), within the
  budget.
- Cost: on that stream, 50,000 steps take 18.4 s with 1,024 slots against
  12.6 s with 64, and 52.1 s against 15.7 s with an ANOMALY output (the
  autoencoder reconstructs the full fused vector; 256 slots: 17.3 s and
  30.1 s). The million-step test takes about 29 minutes instead of 5; its
  timeout is now 3,600 s.
- Checkpoint version 5: the category slots are in the encoder state; pending
  steps have no dense fused vector.
- Reference environments A–I give the same results as before these
  changes (none has an open-vocabulary channel). All 1,999 tests pass with
  GCC and Clang (the million-step test run separately; the scenario tests
  skipped).

**S7.** Findings before the change below (outside the test suite):

- With the relative position (target − position) as input instead of the
  absolute positions, S7 passes at 300 episodes (final distances 0.010 and
  0.052), so the actor-critic can solve the task; it is slow to learn the
  difference of two inputs.
- TD3 on a one-step task with a 2-D action learns the best action (mean
  error 0.105, as with one dimension), so multi-dimensional actions work.
- Learning continues past 300 episodes: at 1,000 episodes seeds 2 and 3
  reach 0.067 and 0.080, but seed 1 collapses (0.17 at episode 500, 1.04 by
  episode 1,000); during the collapse the share of greedy actions at a bound
  rises from about 0.15 to 0.44. The critic's value at the episode start is
  −24 to −28 against realised (50-step) returns of −11 to −14.
- Settings at 300 episodes, seeds 1–3 (final distance): 128 hidden units
  0.10, 0.11, 0.23; 256 units 0.39, 0.08, 0.15; learning rates 3e-3 0.13,
  0.11, 0.13; random start 3,000 steps 0.19, 0.10, 0.12; four updates per
  step 0.26, 0.09, 0.19; τ 0.01 0.23, 0.17, 0.26; learning rates 3e-3 with
  128 units 0.14, 0.04, 0.12. None passes at all three seeds; no default
  was changed.

**S4.** The agent detects the change (one drift event) and recovers from
an error of 5.3 right after it to 0.026–0.028 at 1,000–2,000 steps after;
the criterion asks for at most twice the error before the change (0.024–
0.025; the noise floor is 0.01).

**S7 change: exploration noise 0.5 and 128 hidden units.** More
exploration noise was the one setting that moved S7 toward the criterion
(final distance at seeds 1–3: noise 0.3 0.13, 0.10, 0.14; 0.5 0.12, 0.07,
0.10; 0.7 0.10, 0.07, 0.11). Decaying the noise from 0.5 to 0.1 over the
first 10,000 exploring steps was worse (0.14–0.17), so the noise helps
through more varied training data throughout, not only early exploration.
Two combinations passed at seeds 1–3; both were checked on the pendulum
(environment I) because the settings apply to every continuous output:

| Setting | S7 seeds 1–3 | S7 seeds 4–11 | Pendulum, the 24 C++ seeds used before: runs < −200, stuck episodes, worst |
|---|---|---|---|
| Phase 8 settings | 0.24–0.28 | — | 3, 0, −215.4 |
| Noise 0.5, learning rates 3e-3 | 0.080, 0.094, 0.051 | 8 of 8 pass | 2, **19**, −1,293.5 (rejected) |
| **Noise 0.5, 128 hidden units** | 0.065, 0.073, 0.073 | 7 of 8 pass (seed 4: 0.50) | **0, 0, −195.6** |

The defaults are now `continuous_exploration_noise` 0.5 (was 0.1) and
`continuous_hidden` 128 (was 64). Further checks of the new defaults:

| | Before | After |
|---|---|---|
| Pendulum, 24 C++ seeds not used before (17–47): runs < −200, worst, mean | 1, −230.9, −151.2 | 1, −205.8, −150.3 |
| Pendulum through Python, seeds 81–104: runs < −200, worst, mean | 1, −511.8, −171.2 | 0, −186.2, −157.4 |
| Environment I test (seed 81) | −122.9 | −117.1 |
| Time of the I test alone | about 94 s | about 315 s |
| Scenario tests passing | 24 of 30 | 27 of 30 (S4 fails) |

The 128-unit networks cost about 3–4× per training step for continuous
outputs; the timeouts of the reference tests are now 1,800 s (C++ and
Python). Exploring episodes score lower with the larger noise (pendulum:
last 20 training episodes −454.9); greedy actions are unaffected, and
`Prediction::confidence` of a continuous output is now 0.5 after the random
start (1 − the noise). All 1,999 tests pass (the million-step test is
unaffected: it has no continuous output).

**S4 change: the pattern memory forgets pre-drift feedback.** After the
change of target function the network recovered quickly (its error back to
about 0.011 by step 5,000), but the pattern side did not: its standardized
error was still 0.06 at step 6,000, against 0.008 before the change, and it
kept 9–18% of the blend. Each pattern's target is a running mean that moves
10% toward each new target the pattern receives, and in a 2-D input space
with thousands of patterns most receive few new targets, so the votes stayed
close to the old function; the short-term vote did not help, because it
only selects patterns that received some recent target, and their means
still mostly reflected old ones. A fresh agent trained on the new function
alone reaches 0.0164 over its samples 1,000–2,000, so the shortfall was
recovery, not learning speed.

On a detected drift, the agent already dropped pre-drift targets of the
drifted output from the replay buffer; it now also clears every pattern's
class weights (CLASS) or target means (VALUE) for that output
(`PatternMemory::ForgetLabels` / `ForgetValues`). Until new feedback
arrives the pattern side does not vote, and the network predicts alone.

| | Before | After |
|---|---|---|
| S4 seeds 1–3: MSE 1,000–2,000 steps after the change (limits 0.0244–0.0250) | 0.0257, 0.0275, 0.0270 | 0.0140, 0.0132, 0.0134 |
| S4 seeds 1–3: MSE over the first 200 steps after the change | 5.27–5.37 | 0.87–1.27 |
| S4 seeds 4–11 (not used to choose) | not run | 8 of 8 pass (0.0134–0.0155) |
| Scenario tests passing | 27 of 30 | 30 of 30 |
| Reference environment A (accuracy after the drift) | 0.983 | 0.9835 (others unchanged; they detect no drift) |

All 1,999 tests pass.

**S7 seed 4 change: a penalty on saturated actor outputs.** At seed 4 the
actor drove both actions to a bound and S7 ended at 0.50. At the bound the
tanh output's slope is close to 0, so the critic's gradient no longer reaches
the actor and it stays there. The actor's loss now has a penalty
`continuous_saturation_penalty` × (|z| − `continuous_saturation_threshold`)²
on each pre-tanh output z beyond the threshold (default 2.0, where tanh is
0.96). Below the threshold the actor is unchanged, so actions can still
come close to the bounds. On a task whose best action is the bound, the
actor's z now levels off at about 3.6 (weight 0.001), against 5.5 and
rising without the penalty (`SaturationPenaltyKeepsTheActorOffTheBound`).

Candidates were checked on S7 and on the pendulum, because they change
every continuous output:

| Actor change | S7 seeds 1–11 | Pendulum, 24 C++ seeds: runs < −200, stuck episodes, worst, mean | Python seed 81 (the I test) |
|---|---|---|---|
| None (before) | 10 of 11 (seed 4: 0.50) | 0, 0, −195.6, −161.4 | −153.6 |
| Linear output, clipped, inverted gradients | 11 of 11 | 5, 2, −289.5, −176.2 | not run |
| tanh with a minimum inward slope of 0.1 | 11 of 11 | 3, 3, −288.2, −171.8 | not run |
| tanh with a minimum inward slope of 0.03 | 11 of 11 | 2, 7, −637.7, −181.6 | not run |
| Penalty, weight 0.005 | 11 of 11 | 2, 1, −251.9, −164.3 | −154.1 |
| Penalty, weight 0.003 | 11 of 11 | 0, 0, −196.0, −158.4 | −252.8 (fails; seeds 82–104: 0 of 23 fail) |
| Penalty, weight 0.002 | 11 of 11 | 1, 0, −217.0, −160.7 | −153.2 |
| **Penalty, weight 0.001** | **11 of 11 (0.036–0.090; seed 4: 0.046)** | **1, 0, −210.7, −160.5** | **−146.3** |

Larger weights (0.01 to 10, thresholds 1.5 to 3) also fixed seed 4; on the
pendulum, weight 0.01 had 1 run below −200 (−212.0), 0.1 had 1 (−215.2),
and 1 (threshold 3) had 3 and 15 stuck episodes. The default is weight 0.001: no
candidate kept every pendulum run of the 24 above −200 while passing the
Python I test, and 0.001 has the smallest worst run of those that pass it.
The environment I test gives −115.7 (C++) and −146.3 (Python).

### Real-world datasets

Four public datasets through the Python package, outside the test suite
(scripts not in the repository). The agent uses the default `AgentConfig`
(seeds 1–3), streams the training rows in a random order with each label
right after its prediction, then predicts the test rows without feedback.
The offline models are scikit-learn models fitted on the same training rows.

| Dataset (input type) | Split | Agent, 1 pass | Agent, 3 passes | Offline models |
|---|---|---|---|---|
| Adult census income (6 numeric, 8 categorical) | UCI split, 32,561 / 16,281 | 0.843–0.848 | 0.832–0.837 | logistic regression 0.853, gradient boosting 0.871, always "≤50K" 0.764 |
| SMS Spam Collection (text) | random 80/20, 4,459 / 1,115 | 0.981–0.985 | 0.979–0.984 | TF-IDF + logistic regression 0.984, naive Bayes 0.984 |
| UCI handwritten digits (8×8 image) | random 80/20, 1,437 / 360 | 0.981–0.983 | 0.983–0.992 | logistic regression 0.981, 3-nearest neighbours 0.986, SVM 0.994 |
| Melbourne daily minimum temperature (time series) | forecasts for 1988–1990 | MAE 1.99–2.03 °C | — | yesterday's value 2.01, linear model on the last 16 days 1.80 |

Accuracy unless stated. Results before the change below.

**Adult.** Findings (seed 1 unless stated):

- The network alone reaches 0.843 after one pass, against 0.857 for an
  offline 64×64 network after one epoch (0.853 after 10, 0.831 after 50).
  Training every 16 or 32 observations gives 0.855 and 0.857; a learning
  rate of 1e-3 gives 0.852. Drift detection (no drift events), log-scaling
  the capital gain and loss values, and the share of newest samples in a
  batch did not help. The agent trains on each row about 8 times per pass
  (a batch of 64 every 8 rows), which over-fits this noisy data.
- Over three passes the network's accuracy on rows it has already seen
  rises 0.84 → 0.90 → 0.93 while its test accuracy falls 0.843 → 0.827 →
  0.815: it memorises the training rows. The pattern side alone gives 0.814
  → 0.818 → 0.826.
- The blend weight of the network stays at 0.48–0.56 (squared running
  accuracies of 0.85 and 0.82 are nearly equal), so the weaker pattern side
  takes half of each prediction: with training every 32 rows the network
  alone gives 0.857 (seed 1), the blend 0.848–0.852 (seeds 1–3).
- Training followed observations, so the agent kept training on stored
  samples while it observed the unlabelled test rows (about 2,000 of 6,105
  training steps in a one-pass run).

**Change: training follows new samples.** Without action outputs, a
training step now comes after every `train_every` new trainable samples
instead of every `train_every` observations; more feedback for a stored
sample does not count. Environments with ACTION or CONTINUOUS_ACTION outputs
still train every observation: applied there too, the first step of each
episode (which completes no transition) no longer trained, and S7 failed at
seed 3 (0.110 against 0.090; seeds 1–11: 10 of 11 pass).

| Seeds 1–3 | Before | After |
|---|---|---|
| Adult, 1 pass | 0.843–0.848 | 0.847–0.849 |
| Adult, 3 passes | 0.832–0.837 | 0.834–0.838 |
| SMS, 1 pass | 0.981–0.985 | 0.981–0.986 |
| Digits, 1 pass | 0.981–0.983 | 0.983–0.989 |
| S7 seeds 1, 3, 4 | 0.06923, 0.09019, 0.04551 | identical |

During a labelled pass the agent still trains as often as before, so the
over-training per row and the over-fitting over repeated passes remain;
the change removes only the training on observations without feedback.

**Change: held-out check of the training rate.** Training less often
helped Adult but hurt the less noisy tasks (`train_every`, seeds 1–3):

| | 8 | 16 | 32 |
|---|---|---|---|
| Adult, 1 pass / 3 passes | 0.847–0.849 / 0.834–0.838 | 0.846–0.851 / 0.837–0.842 | 0.847–0.848 / 0.845–0.850 |
| B: error / offline (≤ 1.2), C++ / Python | 1.148 / 1.111 | 1.198 / 1.236 (fails) | 1.260 / 1.234 (both fail) |
| G (≥ 0.95) | 0.975 | 0.963 | 0.942 (fails) |
| S2, lowest R² (≥ 0.7) | 0.776–0.789 | 0.772–0.781 | 0.676–0.683 (fails) |

The loss on reused against first-use samples within a batch did not tell
the tasks apart (its ratio was 0.41–0.49 for G, 0.77–0.78 for Adult's first
pass, and near 1 on Adult's repeated passes, where its first-use samples are
repeats). The training rate now follows held-out samples instead: without
action outputs, a sample whose raw observation hashes to 0 modulo
`holdout_every` (20) is kept out of training (the pattern memory still
learns from it), so a repeated observation is held out every time. Every
`holdout_check_interval` (100) training steps, the network's mean loss on
the newest `holdout_capacity` (500) held-out samples is compared with the
last check (once there are `holdout_min_samples`, 50): if it rose (mean
relative change over the heads > 0), training happens half as often (up to
every `max_training_slowdown` (8) × `train_every` samples); if it fell,
twice as often. A drift clears the held-out samples and restores the full
rate. `AgentStats::held_out_samples` and `training_slowdown` report them.
Checkpoint version 7 adds the held-out samples, each sample's observation
hash and held-out mark, and the controller state.

| Seeds 1–3 | Before | After |
|---|---|---|
| Adult, 1 pass | 0.847–0.849 | 0.844–0.850 |
| Adult, 3 passes | 0.834–0.838 | 0.845–0.847 |
| B: C++ / Python (≤ 1.2) | 1.148 / 1.111 | 1.170 / 1.144 |
| G (≥ 0.95) | 0.975 | 0.971 |
| C: error / last value, C++ / Python | 0.110 / 0.122 | 0.118 / 0.120 |
| S2, lowest R² | 0.776–0.803 | 0.774–0.807 |
| S4, MSE 1,000–2,000 steps after the change | 0.0131–0.0140 | 0.0136–0.0145 |
| SMS, 1 pass / 3 passes | 0.981–0.986 / 0.980–0.984 | 0.983–0.985 / 0.979–0.987 |
| Digits, 1 pass / 3 passes | 0.983–0.989 / 0.981–0.989 | 0.981–0.992 / 0.986–0.994 |
| Temperatures, MAE | 1.95–2.04 | 2.00–2.03 |

The check prevents most of the over-fitting over repeated passes; a single
pass is unchanged. The tasks that need much training lose a little,
mostly because the held-out twentieth is never trained on. A tolerance of
1% on the change gave the same results except on Adult (3 passes
0.843–0.847).

**Change: log-loss blend for CLASS outputs.** The network's weight in a
CLASS prediction was n² / (n² + p²) with n and p the running accuracies of
the two sides, so 0.85 against 0.82 gave 0.52, and on Adult the weaker
pattern side took half of every prediction. It is now
1 / (1 + exp(s × (L_n − L_p))), with L_n and L_p each side's running
log-loss (−log of the true class's probability, floored at 1e-3, at
`quality_rate`; at `class_blend_rate` since Phase 10) and s = `class_blend_sharpness` (10; 0 restores the
accuracy weights). The short- or long-term pattern memory is still chosen
by accuracy. `OutputStats::neural_log_loss` and `pattern_log_loss` report
the two log-losses; checkpoint version 8 stores them.

Before the held-out check this lost on repeated passes (Adult pass 3: 0.818–0.833
at s = 10–50 against 0.833–0.835), because the network's log-loss on rows
it had memorised looked best and it took nearly all the weight. With the
held-out check (seeds 1–3):

| | Accuracy blend | s = 10 | s = 25 |
|---|---|---|---|
| Adult, pass 1 | 0.844–0.850 | 0.849–0.855 | 0.852–0.856 |
| Adult, pass 2 | 0.848–0.853 | 0.850–0.852 | 0.848–0.851 |
| Adult, pass 3 | 0.844–0.846 | 0.841–0.847 | 0.839–0.844 |
| A (C++): before / 800–1,000 after / after the drift | 0.984 / 0.995 / 0.984 | 0.985 / 0.995 / 0.983 | 0.985 / 0.995 / 0.983 |
| A (Python) | 0.990 / 1.000 / 0.986 | 0.990 / 0.990 / 0.985 | 0.990 / 0.990 / 0.984 |
| G | 0.971 | 0.972 | 0.972 |
| S1 / S3 / S5 / S10 | 0.946–0.958 / 0.946–0.952 / 1.0 / 0.997–0.999 | 0.949–0.957 / 0.945–0.950 / 1.0 / 0.997–0.999 | 0.947–0.957 / 0.946–0.950 / 1.0 / 0.997–0.999 |
| SMS, 1 / 3 passes | 0.983–0.985 / 0.979–0.987 | 0.980–0.986 / 0.979–0.987 | 0.979–0.986 / 0.970–0.987 |
| Digits, 1 / 3 passes | 0.981–0.992 / 0.986–0.994 | 0.983–0.989 / 0.986–0.994 | 0.981–0.986 / 0.983–0.994 |

s = 10: Adult's first pass at or above logistic regression (0.853) for two
of three seeds, later passes within noise, the other CLASS tasks within
noise. The weight follows the last ~50 labels and varies between seeds
(0.57–0.98 at the end of Adult's first pass).


## Phase 9: memory

Goal: tasks whose answer depends on observations from earlier steps that
the current observation does not show.

**Design.** A GRU layer (`src/nn/recurrent.hpp`) after the trunk, enabled by
`memory_units` (0 = off, the default): its state h_t = GRU(trunk(x_t),
h_{t−1}) starts at 0 at the beginning of each episode (after `EndEpisode`; a
stream without episodes is one sequence), and a Dense+ReLU layer on [trunk
output, h_t] feeds the heads.

- *Online:* each observation advances the state with the current weights;
  the step's sample keeps the state before it (h_{t−1}) and after it (h_t).
  Predictions use h_t.
- *History:* the newest `memory_history` (20,000) steps are kept in order
  (also those without feedback, whose inputs the state needs).
- *Training:* each training step unrolls the network over `memory_batch`
  windows of up to `memory_window` consecutive steps, with
  backpropagation through time: one window ends at the newest sample (it
  covers the steps since the last training step), the others at samples
  drawn from replay. A window starts at the episode start (state 0) or from
  the state stored before its first step; it does not reach back past an
  episode start. Every step of a window that is in the replay buffer and
  has a target adds to the loss (held-out samples, evicted ones and targets
  dropped after a drift are context only), except the first
  `memory_burn_in` (0) steps of a window that starts inside an episode.
  Environments with action outputs use `action_memory_window` (8) and
  `action_memory_batch` (32) instead of `memory_window` (64) and
  `memory_batch` (8).
- *Initialization:* the GRU's update-gate biases follow "chrono"
  initialization (Tallec & Ollivier 2018) up to the window length, so its
  units start out keeping their state for 2 to `memory_window` steps.
- *Q-learning targets:* windows of environments with action outputs extend
  `action_return_steps` steps past their end; the next state's value uses
  each network's own state there when that step is in a window (both the
  online and the target network are unrolled over the windows), else the
  state stored when the step was observed.
- Continuous actions (TD3) and the pattern memory do not use the state.
- Checkpoint version 9: the memory settings, each sample's states and
  episode-start mark, the history and the current state.

**Tasks** (`tests/agent/memory_test.cpp`; criteria fixed before the first
run; the memoryless agent is run on each once to show it fails):

| | Task | Pass |
|---|---|---|
| M1 | Delayed recall: a cue (A–D) shown only at an episode's first step, then D steps with a 2-value noise input and no cue; a label for the cue at the last step. D = 5, 20, 50; 3,000 episodes each | accuracy at the last step over the last 500 episodes ≥ 0.9 (chance 0.25) |
| M2 | T-maze: a corridor of 10 cells, a cue (left/right) only at the first cell; actions forward, left, right; at the junction turning toward the cue ends the episode with reward +4, the other way with −1; left/right in the corridor and forward at the junction cost −0.1; cut off after 30 steps. 2,000 training episodes | 100 greedy episodes after training: ≥ 90% turn toward the cue |
| M3 | Cart-pole (environment F) observing only the cart position and pole angle (NUMERIC, so no differences are computed) | average reward of 100 consecutive episodes ≥ 195 within 2,000 episodes |
| M4 | Running count: episodes of 50 steps, each step one event a, b or c (uniform); a VALUE output with the number of a's so far in the episode, as a target at every step; 1,000 episodes | mean absolute error over the last 100 episodes ≤ 0.5 |

### Phase 9 results

`memory_units` 32; the memoryless agent (defaults) for comparison:

| | Without memory | With memory | Time |
|---|---|---|---|
| M1, delay 5 / 20 / 50 | 0.244 / 0.280 / 0.264 | 1.000 / 1.000 / 1.000 | 2 s / 12 s / 29 s |
| M2 | 59 of 100 | 100 of 100 | 210 s |
| M3 | best average 44.6 | 195 after 535 episodes, with `action_memory_window` 4 and `action_memory_batch` 64; best average 186.3 at the defaults (8, 32) | 506 s |
| M4 | 4.27 | 0.081 | 14 s |

M3 passes only with settings that make M2 fail (M2 at 64 windows of 4
steps: 59 and 61 of 100), so the action defaults are those that pass M2,
and the M3 test sets its own. Steps toward these results (M3: best
100-episode average unless solved):

| Change | M1 delay 50 | M2 | M3 | M4 |
|---|---|---|---|---|
| First version: update-gate bias 1, Q-learning targets from stored states, heads directly on [trunk, state], window ends spread over the batch (8 windows of 64 steps) | 0.238 | 96 | 24.8 | 0.083 |
| Chrono initialization | 1.000 | 100 | 32.2 | 0.056 |
| Q-learning targets from states unrolled over the windows | | | 37.2 (16 × 8: 69.9) | |
| One window at the newest sample, the others at replayed ones | | | 79.1 | |
| Dense+ReLU layer on [trunk, state] before the heads | 1.000 | 100 | 128.7 | 0.081 |
| M3 windows × length: 32 × 8 / 16 × 16 with burn-in 8 / 64 × 8 / 64 × 8 with burn-in 4 / 64 × 4 | | | 186.3 / 160.9 / 169.5 / 160.3 / solved after 535 episodes | |

With full cart-pole observations (velocities included) the memoryless
agent reached a best average of 183.3 in 2,000 episodes on M3's seed, the
agent with memory and 64 windows of 2 steps solved it after 286 episodes,
and with the first version's 8 windows of 64 steps it reached 134.9: the
number of independent windows, not the memory itself, limited Q-learning.
Cost: with memory a training step of an environment with action outputs
unrolls 256 steps (32 × 8) instead of training 64 samples; the memory tests
take about 13 minutes together (label `memory`; `ctest -LE memory` skips them).

## Phase 10: sentence meaning

Goal: TEXT inputs whose meaning depends on word order, negation and word
meaning, not only on which words occur. Before this phase a TEXT channel was
a bag of hashed words and within-word character tri-grams (1,024 buckets):
word order was lost, and on SST-2 (below) the agent reached 0.644–0.698
(seeds 1–3, one pass; 0.749 and 0.777 with 8,192 buckets) against 0.818 for
logistic regression on bag-of-words.

**Design.** A TEXT channel keeps its hashed features and adds a word
sequence:

- *Encoder:* the first `text_sequence_length` (32; 0 = off) words become
  word ids (0 = no word). Ids are slots in order of first appearance, up to
  `text_vocabulary` (20,000; later new words share one id). If
  `text_word_vectors` names a word-vector file (GloVe text format: a word and
  its values per line; an optional "count dimension" header line), its
  first `text_word_vectors_limit` (40,000) words take the first ids, with
  their vectors, scaled by one factor so that their mean row norm is
  `text_word_vector_norm` (10; 0 = as in the file). Words are split as for the hashed features (ASCII letters
  and digits, lower-cased; "n't" becomes "n" and "t").
- *Network* (`src/agent/word_sequence.hpp`): each id is looked up in a table
  of word vectors: the file's vectors, trained further unless
  `train_word_vectors` is false (then fixed and shared between copies of
  the network); learned vectors (`text_word_dimension`, 50, without a file)
  for the other words. A GRU (`text_sequence_units`, 64) reads the words in
  order and, with `text_bidirectional` (on), a second one from the last word
  to the first; each GRU's state after its last step and the mean of its
  states over the words, and the mean word vector, are added to the
  channel's embedding.
- The pattern memory, the anomaly autoencoder and TD3 read the hashed
  features only (the word ids are not quantities).
- Checkpoints (version 12) hold the vocabulary and the file's vectors;
  loading needs no file.

**Tests** (`tests/agent/text_test.cpp`, label `text`; criteria fixed before
the first run; seeds 1–3). T1 and T3 run when `DPAN_SST2_DIR` (train.tsv and
test.tsv, "sentence\tlabel") and `DPAN_WORD_VECTORS` (an uncompressed GloVe
file) are set; with `DPAN_TEXT_BASELINE` set every test runs without the word
sequence and only prints its results; `DPAN_TEXT_CONFIG` (JSON) changes
settings.

| | Test | Pass |
|---|---|---|
| T1 | SST-2 (Stanford Sentiment Treebank, binary, 6,920 training / 1,821 test sentences): one pass over the training sentences, labels after each prediction, then the test sentences | with GloVe (6B, 100 dimensions): test accuracy ≥ 0.818 (bag-of-words logistic regression, regularization chosen on the 872 dev sentences) for every seed (changed after the first runs, see "Sentence order" below: the mean over 3 orders of the training sentences, with `ensemble_size` 4); without a word-vector file: ≥ 0.75 for every seed |
| T2 | Negation and word order (generated): (a) "the ⟨noun⟩ was [not] ⟨adjective⟩", label = the adjective's polarity, reversed by "not"; (b) "the ⟨animal⟩ chased the ⟨animal⟩" and "the ⟨animal⟩ was chased by the ⟨animal⟩", label = the chaser (6 animals). 3,000 training sentences, 1,000 test sentences each, no word-vector file | test accuracy ≥ 0.95 for (a) and (b); bag-of-words features cannot tell the two orders in (b) apart |
| T3 | Unseen synonyms: "the ⟨noun⟩ was ⟨adjective⟩"; training uses 4 positive and 4 negative adjectives, the test 4 positive and 4 negative adjectives never seen in training | with GloVe: test accuracy ≥ 0.85 |

**Results** (seeds 1 / 2 / 3; GloVe 6B 100d, first 40,000 words; "bag of
words" = `text_sequence_length` 0 with the other settings of this phase;
"one GRU" = the first version, forward GRU only, vectors as in the file):

| | Pass | Two GRUs, norm 10 (final) | One GRU | Bag of words |
|---|---|---|---|---|
| T1, GloVe | ≥ 0.818 | 0.825 / 0.823 / 0.829 (pass) | 0.806 / 0.789 / 0.808 | 0.699 / 0.718 / 0.680 |
| T1, no file | ≥ 0.75 | 0.783 / 0.795 / 0.783 (pass) | 0.764 / 0.784 / 0.775 | 0.699 / 0.718 / 0.680 |
| T2a negation | ≥ 0.95 | 1.0 / 1.0 / 1.0 (pass) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| T2b word order | ≥ 0.95 | 0.963 / 1.0 / 1.0 (pass) | 0.944 / 0.979 / 0.985 | 0.517 / 0.506 / 0.502 |
| T3 unseen synonyms | ≥ 0.85 | 1.0 / 0.864 / 0.885 (pass) | 0.927 / 0.989 / 0.853 | 0.604 / 0.681 / 0.618 |

T1 with GloVe depends on the order of the training sentences: a Python run
of the same data and seeds (numpy permutation instead of `std::shuffle`)
gives 0.802 / 0.827 / 0.804, so the criterion holds for the C++ test's
order only. One SST-2 pass with GloVe takes about 3 minutes (about 1
minute with one GRU).

**Sentence order.** Six runs (both orders, seeds 1–3) give a mean of 0.818
and a minimum of 0.802 with the defaults. After one pass the network scores
0.982 on a sample of training sentences against 0.802 on the test sentences
(Python order, seed 1); the blend weight stays at 0.93–0.99, and averaging
the predictions of the last 2–8 checkpoints adds at most 0.005. None of
these raised the minimum to 0.818 (six runs each, min / mean):

| | Min | Mean |
|---|---|---|
| defaults | 0.802 | 0.818 |
| `text_dropout` 0.25 (optional dropout of the reader's output) | 0.808 | 0.816 |
| `text_word_dropout` 0.1 (a word read as an empty vector) | 0.791 | 0.811 |
| `learning_rate` 2e-3, `text_word_vector_norm` 7 | 0.792 | 0.813 |
| `text_buckets` 64 | 0.787 | 0.809 |
| `text_sequence_units` 32 | 0.800 | 0.808 |
| `text_word_vectors_limit` 100,000 | 0.792 | 0.816 |
| `text_attention` (states pooled by learned attention weights) | 0.802 | 0.813 |

Attention pooling was chosen from a PyTorch prototype of the reader under
the same training regime (12 runs: mean 0.819, minimum 0.809, against
0.812 and 0.787 for the default reader, 0.805 and 0.752 for a convolution
over 3–5 word windows); in the agent it gave the result above and T3 at
seed 3 0.776, so it is off by default.

More passes over the training sentences (`DPAN_SST2_PASSES`; later passes
in new orders) lower the result, as on Adult:

| Passes | Min | Mean |
|---|---|---|
| 1 | 0.802 | 0.818 |
| 2 | 0.786 | 0.803 |
| 3 | 0.794 | 0.803 |
| 2, `text_dropout` 0.5, `text_word_dropout` 0.25 | 0.792 | 0.801 |

`ensemble_size` K (default 1) runs K complete agents (seeds `seed`,
`seed` + 7919, ...; each with its own networks, pattern memory, replay
buffer and training schedule) on the same observations and feedback and
averages their predictions; it is not available with ACTION outputs or
`memory_units`, each agent uses `memory_budget_bytes`, and training takes
K times as long. A first version that kept K networks inside one agent
(sharing its pattern memory, newest samples and training schedule) gave
less (K = 4: min 0.807, mean 0.820). Six runs:

| Agents | Min | Mean |
|---|---|---|
| 1 | 0.802 | 0.818 |
| 4 | 0.817 | 0.824 |
| 5 | 0.815 | 0.826 |
| 8 | 0.820 | 0.829 |

With 4 or 5 agents five of the six runs pass (Python order seed 1 or 3 at
0.815–0.817); with 8 all six pass (Python order 0.823 / 0.835 / 0.832, C++
test 0.820 / 0.832 / 0.833), with a margin of 0.002 at the lowest.

**Criterion change.** After these runs the T1 criterion with GloVe was
restated as the mean over 3 orders of the training sentences per seed (the
test's first order and two more, `std::shuffle` with seed × 1000 + k).
With one agent this fails for seeds 1 and 2; with 4 agents every run is at
least 0.822:

| Seed | 1 agent: orders 1 / 2 / 3, mean | 4 agents: orders 1 / 2 / 3, mean |
|---|---|---|
| 1 | 0.825 / 0.817 / 0.809, 0.817 | 0.824 / 0.828 / 0.822, 0.825 |
| 2 | 0.823 / 0.820 / 0.802, 0.815 | 0.829 / 0.835 / 0.823, 0.829 |
| 3 | 0.829 / 0.811 / 0.824, 0.821 | 0.829 / 0.835 / 0.829, 0.831 |

T1's GloVe runs therefore use `ensemble_size` 4: training and prediction
take about 4 times as long (about 12 minutes per order and seed on one
core) and each agent has its own memory budget. The agent's default stays
1.

Bag-of-words features are enough for T2a (the network learns the "not" ×
adjective interaction); T2b and T3 need the word sequence.

Findings:

- *T2b.* The held-out check keeps 1 in 20 observations out of training, by
  a hash of the observation, so the same sentences for every seed. T2b has
  60 distinct sentences, so about 4 are never trained on; at seed 1 the
  network answers each of them like the trained sentence with the same two
  animals in the same form (for example "the bear chased the cat" → cat).
  With `holdout_every` 0 T2b reaches 1.0; with `text_sequence_units` 128,
  0.981 / 1.0 / 0.984, but SST-2 then takes about 4 times as long with no
  consistent gain (0.780 / 0.809 / 0.799 without a file). The mean of the
  GRU states was added to the output for this test; it did not change the
  T2b range (0.944–0.985 with it, 0.947–0.979 without). A forward GRU's
  final state holds the last words most strongly; the second GRU, reading
  from the last word to the first, does the same for the first words, and
  with it T2b reaches 0.963 / 1.0 / 1.0.
- *T1 with GloVe.* With the vectors fixed, GloVe gave no gain over learned
  vectors (0.783 / 0.789 / 0.795 against 0.786 / 0.801 / 0.799 at
  `quality_rate` 0.02, Python runs). For comparison, logistic regression on the mean
  GloVe vector of a sentence reaches 0.753, and on bag-of-words plus the
  mean vector 0.823. Training the vectors (now the default) and the
  separate blend rate below gave 0.809 / 0.825 / 0.809 in a Python run with
  a different sentence order and 0.806 / 0.789 / 0.808 in the C++ test.
  More training steps (`train_every` 4 or 2) and `learning_rate` 1e-3 did
  not help (0.633–0.789). With two GRUs: 0.818 / 0.815 / 0.808, and T3 at
  seed 3 fell to 0.830 (all of its errors on two unseen adjectives, "poor"
  and "lousy"); with the vectors fixed, T1 0.789 / 0.787 / 0.793 and T3
  seed 3 0.815. GloVe rows have a mean norm of 5.3, learned vectors start
  near 1. Scaling the file's vectors (one factor) to mean norm m:

  | m | T1, GloVe | T3 |
  |---|---|---|
  | 1 | 0.779 / 0.816 / 0.806 | 0.798 / 0.762 / 0.728 |
  | 5.3 (as in the file) | 0.818 / 0.815 / 0.808 | 1.0 / 1.0 / 0.830 |
  | 8 | 0.820 / 0.823 / 0.828 | 0.970 / 0.966 / 0.885 |
  | 10 | 0.825 / 0.823 / 0.829 | 1.0 / 0.864 / 0.885 |
  | 12 | 0.818 / 0.815 / 0.821 | 1.0 / 0.873 / 0.885 |
  | 20 | 0.799 / 0.815 / 0.790 | 0.986 / 0.890 / 0.885 |

  Larger vectors change less, relative to their size, when they are
  trained, so the unseen synonyms stay closer to the trained ones; 8–10
  pass both tests.
- *T1 without a file* (one GRU) depends on the order of the training sentences: the
  C++ test (`std::shuffle`) gives 0.764 / 0.784 / 0.775, a Python run of
  the same data and seeds (numpy permutation) 0.788 / 0.801 / 0.799. The
  blend change below does not lower it: with `class_blend_rate` 0.02 the
  C++ test gives 0.763 / 0.776 / 0.775 and the Python run 0.786 / 0.801 /
  0.799. (The other parts of the change apply to OUTCOME feedback and to
  word-vector files only.)

**Change: separate blend rate.** On SST-2 the network's weight in the CLASS
blend fell from 0.90 to 0.23 over the last 20 training sentences of one run,
so the test used mostly the pattern memory: at `quality_rate` 0.02 the
running log-losses follow about the last 50 labels, and one confident miss
(log-loss up to 6.9) moves them by up to 0.14. They now use
`class_blend_rate` (0.005, about 200 labels; at most 0.035 per miss);
`quality_rate` still drives the running accuracies. With the slower rate an
agent trained by OUTCOME feedback only stayed at chance
(`AgentTest.LearnsFromOutcomesOnly`, 250 of 500): the log-losses were
scored on correct outcomes only, where both sides predicted the same class
with probability near 1. A wrong outcome now scores −log(1 − p(predicted))
on both sides (the true class's log-loss when there are two classes).
Checkpoint version 11 stores the rate.

With these changes the other CLASS tasks stay within noise (seeds 1–3;
before = the s = 10 column of the log-loss blend table under "Real-world datasets"):

| | Before | After |
|---|---|---|
| Adult, pass 1 / 2 / 3 | 0.849–0.855 / 0.850–0.852 / 0.841–0.847 | 0.851–0.855 / 0.849–0.853 / 0.841–0.845 |
| A (C++): before / 800–1,000 after / after the drift | 0.985 / 0.995 / 0.983 | 0.985 / 0.995 / 0.984 |
| A (Python) | 0.990 / 0.990 / 0.985 | 0.989 / 0.990 / 0.985 |
| G | 0.972 | 0.972 |
| S1 / S3 / S10 | 0.949–0.957 / 0.945–0.950 / 0.997–0.999 | 0.948–0.956 / 0.942–0.948 / 0.997–0.999 |
| SMS, 1 / 3 passes | 0.980–0.986 / 0.979–0.987 | 0.984–0.987 / 0.979–0.986 |
| Digits, 1 / 3 passes | 0.983–0.989 / 0.986–0.994 | 0.983–0.989 / 0.986–0.994 |

SMS, S1 and S3 have TEXT inputs, so their "after" includes the word
sequence (one GRU). With two GRUs and norm 10: S1 0.947–0.949, S3
0.942–0.949, SMS 0.987–0.989 / 0.981–0.984 (1 / 3 passes). The full suite
(scenarios included) passes.


## Phase 11: predictor-guided settings search

Goal: find out whether the agent can choose better settings for a task by
learning, from settings it has tried, to predict which untried settings
will do well, and whether that beats trying settings at random with the
same number of trials.

**Tasks and data.** Adult (income, tabular) and SMS Spam (text), with the
splits of the real-world runs above. The search sees only the training
rows, split again (fixed permutation, seed 0) into 80% search-training and
20% validation rows; the test rows are used only for the final check. A
candidate's score is its validation accuracy after one pass over the
search-training rows (agent seed 1).

**Search space** (both tasks; the last two for SMS only):
`learning_rate` (log-uniform 3e-4 to 1e-2), `train_every` {2, 4, 8, 16},
`batch_size` {32, 64, 128}, `min_trunk_width` {16, 32, 64, 128},
`new_sample_share` {0.25, 0.5, 0.75}, `class_blend_sharpness` {0, 5, 10,
25}, `neighbors` {3, 5, 10, 20}, `gradient_clip_norm` {1, 5, 20},
`holdout_every` {0, 20}, `text_buckets` {256, 1024, 4096},
`text_sequence_length` {0, 32}.

**Methods** (40 trained candidates each, per task):

- *Random:* 40 settings drawn at random.
- *Predictor-guided:* the same first 12 random settings, then 7 rounds of
  4. Each round a new agent (NUMERIC input: the settings scaled to [0, 1],
  `learning_rate` on a log scale; VALUE output: validation accuracy;
  `cold_start_samples` 5, `holdout_every` 0) is trained on all measured
  settings (30 passes) and predicts 2,000 random untried settings; the 3
  with the highest prediction and 1 random one are measured.

Each method chooses the setting with the highest validation accuracy. Each
search is repeated with 3 search seeds (different random settings). Each
chosen setting, and the defaults, are then trained on all training rows
(one pass, agent seeds 1–3) and scored on the test rows.

**Criteria** (fixed before the first run; mean test accuracy over 3 search
seeds × 3 agent seeds, defaults over 3 agent seeds):

| | Pass |
|---|---|
| G1: guided against random | guided ≥ random − 0.001 on both tasks and ≥ random + 0.002 on at least one |
| G2: guided against defaults | guided ≥ defaults + 0.002 on both tasks |

Also reported: the best validation accuracy found after each number of
trials (mean over search seeds) for both methods.

**Results** (mean test accuracy; random and guided: 3 search seeds × 3
agent seeds; defaults: 3 agent seeds):

| | Defaults | Random | Guided | Guided − random | Guided − defaults |
|---|---|---|---|---|---|
| Adult | 0.8527 | 0.8540 | 0.8554 | +0.0014 | +0.0027 |
| SMS | 0.9880 | 0.9870 | 0.9874 | +0.0004 | −0.0006 |

G1 (guided against random) is **not met**: the guided search is not worse
than random on either task, but its gain (Adult +0.0014) is below 0.002.
G2 (guided against defaults) is **not met**: on SMS the chosen settings are
0.0006 below the defaults.

Best validation accuracy after n trials (mean over search seeds):

| | 12 | 20 | 28 | 40 |
|---|---|---|---|---|
| Adult, random | 0.8544 | 0.8544 | 0.8554 | 0.8556 |
| Adult, guided | 0.8544 | 0.8551 | 0.8551 | 0.8558 |
| SMS, random | 0.9914 | 0.9936 | 0.9940 | 0.9940 |
| SMS, guided | 0.9914 | 0.9922 | 0.9936 | 0.9936 |

The predictor did not find good settings faster than random draws. The
settings themselves matter little on these tasks: across all searches the
best validation accuracy is within about 0.002 (Adult) and 0.003 (SMS) of
the defaults' test accuracy, and on SMS the validation gains (0.994
against the defaults' 0.988 test accuracy) did not carry over to the test
rows. One choice recurs: `class_blend_sharpness` 25 in 9 of the 12 chosen
settings (default 10).



## Phase 12: games

Goal: measure how well the agent learns to play games from rewards alone,
with its ACTION outputs (Double DQN, default settings). Criteria fixed
before the first run; seeds 1–3; every seed must pass. Tests:
`python/tests/test_games.py`, run when `DPAN_GAMES` is set (the classic
control games also need the `gymnasium` package; with
`gymnasium[box2d]` for LunarLander).

**Classic control** (Gymnasium; the observation as one NUMERIC input, one
ACTION output, the game's reward as REWARD feedback, `EndEpisode(true)` when
the game ends, `EndEpisode(false)` at the time limit). Training for a fixed
number of steps with exploration; then 100 episodes without exploration
(new game seeds). Pass: their mean return reaches Gymnasium's
`reward_threshold` for the game.

| | Game | Training steps | Pass: mean return of 100 greedy episodes |
|---|---|---|---|
| C1 | CartPole-v1 | 300,000 | ≥ 475 |
| C2 | Acrobot-v1 | 200,000 | ≥ −100 |
| C3 | MountainCar-v0 | 500,000 | ≥ −110 |
| C4 | LunarLander-v3 | 600,000 | ≥ 200 |

**Board games** (written in the test; the board from the agent's side as one
NUMERIC input, +1 own stone, −1 the opponent's, 0 empty; one ACTION output
per cell or column; reward +1 for a win, −1 for a loss, 0 otherwise, at the
agent's move that ends the game or after the opponent's reply; a move to a
full cell or column loses the game). The agent moves first in half of the
games. Opponents: *random* (a random legal move) and a rule-based one
(tic-tac-toe: perfect play by minimax; Connect Four: win if it can, else
block the agent's immediate win, else a random legal move). Training games
alternate between the two opponents.

| | Game | Training games | Pass (games without exploration) |
|---|---|---|---|
| B1 | Tic-tac-toe | 30,000 | against random (1,000 games): losses ≤ 1% and wins ≥ 80%; against minimax (200 games): losses ≤ 5% |
| B2 | Connect Four (6 × 7) | 50,000 | against random (1,000 games): wins ≥ 95%; against the rule-based opponent (1,000 games): wins ≥ 50% |

Reported for comparison: a random player's mean return in each classic
control game, and random and rule-based players against each other in the
board games.

B1's win threshold against random play was first 85%. Before any agent run,
the baselines showed that minimax (perfect play, a random choice among
equally good moves) wins 85% of 500 games against random play, so the
threshold was lowered to 80%: 85% would have required beating perfect play
or matching it within sampling noise.

**Results** (seeds 1 / 2 / 3; default settings):

| | Game | Pass | Result | Random play |
|---|---|---|---|---|
| C1 | CartPole-v1 | ≥ 475 | 34.9 / 294.3 / 106.8 (**not met**) | 20.9–23.5 |
| C2 | Acrobot-v1 | ≥ −100 | −89.5 / −495.2 / −82.4 (**not met** at seed 2) | −496.9 to −499.5 |
| C3 | MountainCar-v0 | ≥ −110 | −200.0 / −200.0 / −200.0 (**not met**) | −200.0 |
| C4 | LunarLander-v3 | ≥ 200 | 115.6 / 185.1 / 40.1 (**not met**) | −188.8 to −194.3 |
| B1 | Tic-tac-toe | see above | against random wins 0.942 / 0.939 / 0.931, losses 0.010 / 0.009 / 0.007; against minimax losses 0.045 / 0.050 / 0.020 (pass) | random against random: wins 0.474, losses 0.409 |
| B2 | Connect Four | see above | against random wins 0.947 / 0.922 / 0.927 (**not met**); against the rule-based opponent wins 0.532 / 0.422 / 0.484 (**not met** at seeds 2 and 3) | rule-based against random: wins 0.965 |

The agent learns something in every game but MountainCar (all greedy
episodes reach the 200-step limit: the car never reached the goal, whose
reward is the only signal). The results vary strongly between seeds
(CartPole 34.9 to 294.3; Acrobot −495.2 against −82.4 and −89.5), and in
CartPole seed 1 300,000 training steps took 10,428 episodes (29 steps each),
so the policy was poor through much of the training as well. Only
tic-tac-toe passes.

**Two changes after these results.**

1. *Keep the best policy* (`action_keep_best`, default true;
   `action_best_episodes`, default 50): `Act` without exploration uses a
   copy of the network taken when the mean return of the last 50 episodes
   with exploration was highest. Not with `memory_units`. The copy is in
   checkpoints.
2. *Action settings chosen on the games.* Tuning runs used seeds 11–13 (not
   the test seeds) and C1, C2 and C4, with keep-best on. Mean return of 100
   greedy episodes, seeds 11 / 12 / 13:

   | Settings | CartPole | Acrobot | LunarLander |
   |---|---|---|---|
   | defaults | 213 / 304 / 500 | −499 / −90 / −97 | 210 / 132 / 269 |
   | `target_sync` 2000 | 496 / 500 / 488 | −86 / −500 / −91 | 175 / 139 / 124 |
   | `action_learning_rate` 0.0005 | 327 / 259 / 148 | −101 / −83 / −500 | 211 / 234 / 233 |
   | `action_learning_rate` 0.0001 | 86 / 500 / 217 | −500 / −90 / −106 | 258 / 220 / 213 |
   | `replay_capacity` 200,000 | 495 / 422 / 422 | −497 / −85 / −95 | 214 / 191 / 237 |
   | `exploration_steps` 50,000 | 492 / 92 / 500 | −115 / −88 / −500 | 216 / 218 / 77 |
   | `target_sync` 2000, rate 0.0001 | 495 / 482 / 252 | −500 / −88 / −120 | 244 / 215 / 196 |
   | `target_sync` 1000 | 204 / 194 / 500 | −106 / −88 / −87 | 251 / 244 / 192 |
   | **`target_sync` 1000, replay 200,000** | 467 / 500 / 500 | −113 / −88 / −80 | 278 / 272 / 230 |
   | replay 200,000, rate 0.0001 | 86 / 500 / 500 | −500 / −99 / −370 | 234 / 248 / 243 |

   The last but one is the only setting whose mean over the three seeds
   reaches the threshold in all three games. The Huber loss was already in
   use. Adopted: a new setting `action_replay_capacity` (default 200,000)
   replaces `replay_capacity` when the spec has ACTION outputs and
   `memory_units` is 0 (other agents keep 50,000; with 200,000, memory task
   M3 was not solved in 2,000 episodes, with 50,000 it passes). `target_sync` 1000 as the default made
   reference E (64 of 100 shortest paths in C++, 78 in Python; needs 95)
   and Python F (best average 182.7; needs 195) fail, so the default stays
   500 and C1–C4 set `target_sync` 1000 in the test. Checkpoint version 17.

**Results after the two changes** (seeds 1 / 2 / 3):

| | Game | Pass | Result |
|---|---|---|---|
| C1 | CartPole-v1 | ≥ 475 | 500.0 / 309.5 / 327.2 (**not met** at seeds 2 and 3) |
| C2 | Acrobot-v1 | ≥ −100 | −84.9 / −89.8 / −91.8 (pass) |
| C3 | MountainCar-v0 | ≥ −110 | −195.5 / −187.6 / −199.3 (**not met**) |
| C4 | LunarLander-v3 | ≥ 200 | 213.4 / 229.8 / 244.8 (pass) |
| B1 | Tic-tac-toe | see above | against random wins 0.941 / 0.919 / 0.930, losses 0.012 / 0.027 / 0.006; against minimax losses 0.045 / 0.155 / 0.020 (**not met** at seeds 1 and 2) |
| B2 | Connect Four | see above | against random wins 0.969 / 0.960 / 0.952; against the rule-based opponent wins 0.526 / 0.550 / 0.488 (**not met** at seed 3) |

C2 and C4 now pass, C1 and B2 are closer, and MountainCar's greedy episodes
reach the goal in a few cases. B1 no longer passes. Undoing one change at a
time does not restore it: without keep-best, seed 2 loses 0.010 against
random and 0.060 against minimax; with `action_replay_capacity` 50,000,
seed 2 loses 0.018 and 0.110. Before the changes B1 passed with small
margins (seed 2: 0.009 and 0.050).
