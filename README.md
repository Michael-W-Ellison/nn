# DPAN — Dynamic Pattern Association Network

DPAN is a C++17 library for learning from streams of data without a fixed
network topology. It stores observed inputs as **patterns** (feature vectors
with metadata), finds similar patterns, learns **associations** between
patterns that co-occur, and manages memory by utility (pruning, tiered
storage, decay and consolidation). This learning is incremental and driven
by co-occurrence statistics and reinforcement.

It also includes a conventional neural network module (dense layers trained
with backpropagation) that works on its own and plugs into DPAN as a learned
feature extractor, a learned similarity metric and a learned prediction
scorer.

## Modules

| Library (CMake target) | Purpose |
|---|---|
| `dpan::core` | `PatternID`, `PatternData`/`FeatureVector`, `PatternNode`, `PatternEngine` (end-to-end: extract, match, create, search) |
| `dpan::storage` | `PatternDatabase` interface; in-memory (optionally memory-mapped) and SQLite backends; `CreatePatternDatabase(config_file)` |
| `dpan::similarity` | Similarity metrics (cosine, Euclidean, Hausdorff, Chamfer, statistical, spectral, contextual) and similarity search |
| `dpan::discovery` | Pattern extraction, matching, creation and refinement |
| `dpan::association` | Co-occurrence tracking, association formation, reinforcement, competition, normalization, prediction, activation propagation |
| `dpan::memory` | Utility scoring, adaptive thresholds, pruning, active/warm/cold/archive tiers, decay, interference, consolidation |
| `dpan::learning` | Attention mechanisms (basic, context-aware, multi-head, self-attention) and pattern importance |
| `dpan::nn` | Neural networks: dense layers, 2-D convolution, max pooling, activations, dropout, losses, SGD/Adam, `Sequential` training and persistence; `NeuralEncoder`/`Autoencoder`, `EmbeddingSimilarity`, `NeuralPredictionScorer` |
| `dpan::agent` | Drop-in learner (`docs/AGENT_PLAN.md`): describe inputs, feedback and outputs in an `EnvironmentSpec`; `Agent` encodes observations, learns from labels, targets, outcomes, rewards and the stream itself, predicts classes, values, next observations and anomaly scores, and chooses discrete actions (Double DQN) and continuous actions (TD3). Detects concept drift and recovers, widens its network when it lacks capacity, rolls back on NaN or diverging training, and stays within a memory budget. `Save`/`Load` write one checkpoint from which an agent continues exactly. |
| `dpan::c_api` | C interface to the agent (`src/capi/dpan_c.h`): a shared library with an opaque handle; spec and configuration as JSON |
| `dpan` (Python) | Python bindings to the agent (`python/`, pybind11, NumPy arrays as inputs) |
| `dpan::dpan` | All C++ libraries above |

## Requirements

- CMake ≥ 3.20 and a C++17 compiler (GCC or Clang)
- SQLite3 and zlib development packages (optional, see `DPAN_WITH_SQLITE`
  and `DPAN_WITH_ZLIB`)
- A BLAS library with `cblas.h` (optional, see `DPAN_WITH_BLAS`)
- Python ≥ 3.8 with development files and NumPy (only for `DPAN_BUILD_PYTHON`)
- libyaml (only for the `dpan_cli` tool)
- Network access on first configure when building tests (GoogleTest is fetched)

## Building

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build -j8          # run the test suite
ctest --test-dir build -j8 -LE long # without the ~30-minute million-step test
DPAN_SCENARIOS=1 ctest --test-dir build -j8 -L scenarios   # scenario tests (skipped otherwise)
```

Options (defaults are ON when DPAN is the top-level project, OFF when it is
added with `add_subdirectory`):

| Option | Effect |
|---|---|
| `DPAN_BUILD_TESTS` | Build the GoogleTest suite |
| `DPAN_BUILD_EXAMPLES` | Build programs in `examples/` |
| `DPAN_BUILD_CLI` | Build the interactive `dpan_cli` |
| `DPAN_INSTALL` | Install rules and the `DPAN` CMake package (default ON) |
| `DPAN_WITH_SQLITE` | Build the SQLite `PersistentBackend` (default ON). When OFF, the `persistent`/`sqlite` database types throw, and `dpan_cli` is not built. |
| `DPAN_WITH_ZLIB` | Compress archive-tier files (default ON). When OFF, archive files are written uncompressed and compressed files written by a zlib build cannot be read. |
| `DPAN_BUILD_C_API` | Build the C API shared library `dpan_c` (default ON; target `dpan::c_api`) |
| `DPAN_BUILD_PYTHON` | Build the Python package `dpan` in `<build>/python` (default OFF). Needs the Python 3 development files; uses an installed pybind11 or fetches it. `DPAN_PYTHON_INSTALL_DIR` sets where `cmake --install` puts it. |
| `DPAN_WITH_BLAS` | Use BLAS (`cblas_sgemm`, OpenBLAS by default; set `BLA_VENDOR` for another) for matrix products (default OFF). Needs `cblas.h`, for example from `libopenblas-dev`. Without it, large products are split across threads (`nn::SetThreadCount`). |

### Time

All timestamps come from a process-wide clock (`core/clock.hpp`). By
default it is the system clock (microseconds since the Unix epoch). Install
a `ManualClock` with `SetClock` or `ScopedClock` to drive time from a
simulation or to make runs reproducible.

## Using DPAN in another project

Installed package:

```bash
cmake -S . -B build -DDPAN_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/opt/dpan
cmake --build build -j && cmake --install build
```

```cmake
find_package(DPAN 0.1 REQUIRED)            # with -DCMAKE_PREFIX_PATH=/opt/dpan
target_link_libraries(my_app PRIVATE dpan::dpan)
```

Source tree:

```cmake
add_subdirectory(path/to/dpan)
target_link_libraries(my_app PRIVATE dpan::dpan)
```

Headers are included by module path, e.g. `#include "core/pattern_engine.hpp"`.
`examples/consumer/` is a complete project that builds either way.

The module libraries are static. The C API (`dpan::c_api`) is a shared
library that contains everything the agent needs and exports only its
`dpan_*` functions.

## Drop-in agent

Describe the environment once, then stream observations and send feedback
when it is available (`examples/agent/agent_example.cpp`):

```cpp
#include "agent/agent.hpp"
using namespace dpan::agent;

EnvironmentSpec spec;
spec.inputs = {{"reading", InputType::NUMERIC}, {"source", InputType::CATEGORICAL}};
spec.outputs = {{"state", OutputType::CLASS}, {"unusual", OutputType::ANOMALY}};
spec.feedback = {{"label", FeedbackType::LABEL, "state"}};
Agent agent(spec);

StepId step = agent.Observe({{"reading", std::vector<float>{2.1f, 0.2f}}, {"source", std::string("a")}});
Prediction state = agent.Predict(step, "state");      // label, probabilities, confidence
float unusual = agent.Predict(step, "unusual").score;  // percentile among recent observations
agent.Feedback(step, "label", std::string("hot"));     // may arrive later
```

Actions are learned from rewards:

```cpp
spec.outputs = {{"move", OutputType::ACTION, {"up", "down", "left", "right"}}};
spec.feedback = {{"reward", FeedbackType::REWARD, "move"}};
...
StepId step = agent.Observe({{"position", std::vector<float>{x, y}}});
std::string move = agent.Act(step, "move");          // explores; Act(step, "move", false) is greedy
agent.Feedback(step, "reward", -0.04);
agent.EndEpisode(true);                              // when the episode ends (true = terminal)
```

Continuous actions (values within bounds) are learned with TD3:

```cpp
OutputSpec torque{"torque", OutputType::CONTINUOUS_ACTION};
torque.size = 1;
torque.low = {-2.0f};
torque.high = {2.0f};
spec.outputs = {torque};
spec.feedback = {{"reward", FeedbackType::REWARD, "torque"}};
...
std::vector<float> u = agent.ActContinuous(step, "torque");   // ActContinuous(step, "torque", false): greedy
agent.Feedback(step, "reward", reward);
```

Save and restore everything learned (a loaded agent continues exactly as
the saved one would):

```cpp
agent.Save("agent.dpag");                       // false if the file cannot be written
Agent restored = Agent::Load("agent.dpag");     // throws std::runtime_error if malformed
```

Sizes, value ranges, vocabularies and (unless listed) class names are
learned from the data. See `docs/AGENT_PLAN.md` for what is implemented.

### From C

`src/capi/dpan_c.h` (shared library `dpan_c`). The spec and configuration are
JSON (format in `src/agent/spec_json.hpp`); predictions and statistics come
back as JSON. Every function returns a `dpan_status`; `dpan_last_error()`
has the message. Complete program: `examples/c_api/agent_c_example.c`.

```c
#include "dpan_c.h"

dpan_agent* agent = NULL;
if (dpan_agent_create("{\"inputs\": [{\"name\": \"x\", \"type\": \"numeric\"}],"
                      " \"feedback\": [{\"name\": \"label\", \"type\": \"label\", \"output\": \"kind\"}],"
                      " \"outputs\": [{\"name\": \"kind\", \"type\": \"class\"}]}",
                      NULL, &agent) != DPAN_OK) {
    fprintf(stderr, "%s\n", dpan_last_error());
}
dpan_step step;
dpan_agent_observe_json(agent, "{\"x\": [0.5, 1.5]}", &step);   /* or dpan_observation_set_* */
char* prediction = NULL;
dpan_agent_predict_json(agent, step, "kind", &prediction);
dpan_string_free(prediction);
dpan_agent_feedback_text(agent, step, "label", "b");
dpan_agent_free(agent);
```

### From Python

Build with `-DDPAN_BUILD_PYTHON=ON`, then `PYTHONPATH=<build>/python`:

```python
import numpy as np
import dpan

agent = dpan.Agent({
    "inputs": [{"name": "x", "type": "numeric", "size": 4},
               {"name": "camera", "type": "image"}],
    "feedback": [{"name": "label", "type": "label", "output": "kind"}],
    "outputs": [{"name": "kind", "type": "class"}],
}, config={"seed": 7})
step = agent.observe({"x": [0.1, 0.2, 0.3, 0.4], "camera": np.zeros((32, 32))})
p = agent.predict(step, "kind")          # p.label, p.probabilities, p.confidence, ...
agent.feedback(step, "label", "b")
agent.save("agent.dpag"); restored = dpan.Agent.load("agent.dpag")
```

Actions: `agent.act(step, output)` returns the name of a discrete action;
`agent.act_continuous(step, output)` returns the values of a
`continuous_action` output. In C: `dpan_agent_act` and
`dpan_agent_act_continuous`.

Values may be numbers, strings, sequences or NumPy arrays (1-D: vector;
2-D or 3-D: image, height × width [× channels]); `None` marks a missing
channel. `dpan.use_manual_clock()` and `dpan.advance_clock(seconds)` make
runs reproducible.

## Example

```cpp
#include "association/association_learning_system.hpp"
#include "core/pattern_engine.hpp"

using namespace dpan;

PatternEngine::Config config;
config.similarity_metric = "cosine";
PatternEngine engine(config);

FeatureVector features(std::vector<float>{0.2f, 0.9f, 0.1f});
PatternID id = engine.CreatePattern(PatternData::FromFeatures(features, DataModality::NUMERIC));
auto similar = engine.FindSimilarPatterns(engine.GetPattern(id)->GetData(), 5);

AssociationLearningSystem associations;
associations.RecordPatternActivation(id);           // call for each observed pattern
associations.FormNewAssociations(engine.GetDatabase());
auto next = associations.PredictWithConfidence(id, 3);
```

Neural network:

```cpp
#include "nn/sequential.hpp"

using namespace dpan::nn;

Sequential model;
model.Emplace<Dense>(2, 8).Emplace<Tanh>().Emplace<Dense>(8, 1).Emplace<Sigmoid>();
Adam optimizer(0.05f);
model.Fit(inputs, targets, BinaryCrossEntropy(), optimizer);   // Matrix, one sample per row
std::vector<float> output = model.Predict({1.0f, 0.0f});
```

## Documentation

- `docs/DPAN_User_Guide.md`, `docs/DPAN_API_Reference.md` — usage and API
- `docs/DPAN_Neural_Network_Guide.md` — neural network module and its DPAN integration
- `docs/DPAN_CLI_Guide.md`, `QUICK_START_CLI.md` — the interactive CLI
- `docs/DPAN_Design_Document.md` — architecture and design

Status reports and analyses written during development are kept in
`docs/history/` as snapshots; they do not describe the current state.

## Known limitations

- Approximate search (`PatternMatcher` fast search, `ApproximateSearch`)
  retrieves candidates by angle between feature vectors (random-hyperplane
  LSH) and can miss matches; it is used by the matcher only for databases of
  at least `fast_search_min_patterns` (default 10,000) patterns.
- Interference between patterns is O(n²) per maintenance pass and is limited
  to `MemoryManager::Config::max_interference_patterns` patterns.
- Some heuristics are simplified (temporal variance estimate, pattern
  splitting by perturbation, type classification from pattern structure).
