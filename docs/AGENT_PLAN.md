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
| REWARD → ACTION | Double DQN: Q-value head (Huber loss), target network synced every 500 training steps, 3-step returns cut at exploratory actions, ε-greedy exploration falling from 1.0 to 0.05 | Associations between states and actions, reinforced by reward sign | Neural Q-values; associations break ties and choose during the cold start |
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
3. **Training cadence:** every 8 observations, one mini-batch of 64, half
   newest samples and half replay. Gradient clipping (norm 5), then Adam
   with a learning rate that drops when progress plateaus. When the
   environment has ACTION outputs: every observation, learning rate 5e-4,
   trunk at least 64 wide; Q-value losses do not drive the plateau check.
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
| 2. Agent: supervised and self-supervised | Complete. `src/agent/agent.hpp`. Reference environments A–D pass (results and notes below). |
| 3. Reinforcement | Complete. `Agent::Act`, REWARD and OUTCOME feedback for ACTION outputs, episodes. Reference environments E–F pass (results and notes below). |
| 4. Robustness and self-tuning | Complete. Drift detection and recovery, trunk widening, rollback, memory budgets, metrics hook (results and notes below). |
| 5. Full persistence | Complete. `Agent::Save` / `Agent::Load`: one versioned checkpoint; a loaded agent continues exactly as the saved one (results and notes below). |
| 6–8 | Not started |

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
  (`action_train_every`), learning rate 5e-4 (`action_learning_rate`),
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
