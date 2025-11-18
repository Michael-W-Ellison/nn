# DPAN User Guide

**Dynamic Pattern Association Network - Complete User Guide**

**Version:** 1.0
**Date:** 2025-11-18

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Installation and Building](#installation-and-building)
5. [Using DPAN as a Library](#using-dpan-as-a-library)
6. [Using the CLI Interface](#using-the-cli-interface)
7. [Configuration Guide](#configuration-guide)
8. [Training Strategies](#training-strategies)
9. [Advanced Features](#advanced-features)
10. [Performance Tuning](#performance-tuning)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)
13. [Examples and Tutorials](#examples-and-tutorials)

---

## Introduction

### What is DPAN?

DPAN (Dynamic Pattern Association Network) is a novel artificial intelligence architecture that learns patterns and associations from raw input data without requiring labeled training sets. Unlike traditional neural networks that need supervised training, DPAN:

- **Self-organizes** through exposure to data
- **Discovers patterns** autonomously from unlabeled input
- **Forms associations** based on co-occurrence and temporal relationships
- **Continuously adapts** its internal representation
- **Learns incrementally** without catastrophic forgetting

### Key Features

**Pattern Learning**
- Automatic pattern extraction from raw bytes
- Hierarchical pattern organization (atomic, composite, meta)
- Similarity-based pattern matching
- Feature compression and indexing

**Association Learning**
- Co-occurrence tracking and analysis
- Multiple association types (causal, categorical, spatial, functional)
- Reinforcement-based strength adjustment
- Competitive dynamics and normalization

**Attention Mechanisms**
- Multi-head attention for diverse perspectives
- Context-aware pattern selection
- Importance-based weighting
- Configurable attention strategies

**Storage and Persistence**
- In-memory or SQLite-based persistent storage
- Efficient indexing for fast retrieval
- Session save/restore capabilities
- Automatic compaction and maintenance

### Use Cases

- **Conversational AI**: Learn language patterns from dialogue
- **Sequence Prediction**: Predict next elements in sequences
- **Pattern Recognition**: Identify recurring patterns in data streams
- **Anomaly Detection**: Detect deviations from learned patterns
- **Content Recommendation**: Suggest related content based on associations
- **Time Series Analysis**: Learn temporal patterns and relationships

---

## Getting Started

### Prerequisites

**Required:**
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14 or higher
- SQLite3 development libraries
- Standard C++ library

**Optional:**
- Git (for version control)
- Doxygen (for generating documentation)
- Valgrind (for memory debugging)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/dpan.git
cd dpan

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

# Run tests to verify installation
ctest --output-on-failure

# Run the CLI
./src/cli/dpan_cli
```

**30-Second CLI Tutorial:**

```bash
# Start DPAN CLI
./build/src/cli/dpan_cli

# Talk to DPAN
dpan> Hello
[Processing: "Hello"]
[Created 1 new pattern(s)]

dpan> How are you?
[Created 1 new pattern(s)]

dpan> I am fine
[Created 1 new pattern(s)]

# Check what was learned
dpan> /stats
# Shows statistics

# Make a prediction
dpan> /predict Hello
# Shows what DPAN expects after "Hello"

# Exit (auto-saves)
dpan> exit
```

See [QUICK_START_CLI.md](../QUICK_START_CLI.md) for more CLI details.

---

## Core Concepts

### Patterns

**Definition:** A pattern is an abstracted representation of input data, stored with metadata and features.

**Types:**
- **Atomic**: Indivisible, basic patterns extracted directly from input
- **Composite**: Patterns composed of multiple sub-patterns
- **Meta**: High-level patterns representing relationships between patterns

**Pattern Structure:**
```
Pattern
├── ID (unique 64-bit identifier)
├── Type (ATOMIC, COMPOSITE, META)
├── Data (compressed raw data + feature vector)
├── Confidence (0.0 - 1.0)
├── Timestamps (creation, last access)
├── Access count
└── Metadata (key-value pairs)
```

**Example:**
```cpp
// Text "Hello" becomes:
PatternNode {
    id: 12345678901234567890
    type: ATOMIC
    data: compressed("Hello") + features[64]
    confidence: 0.85
    created: 2025-11-18 10:30:00
    accessed: 2025-11-18 10:35:00
    access_count: 5
}
```

### Associations

**Definition:** An association is a weighted, directed connection between two patterns representing their relationship.

**Types:**
- **CAUSAL**: Pattern A typically precedes pattern B (temporal sequence)
- **CATEGORICAL**: Patterns A and B belong to the same category
- **SPATIAL**: Patterns appear in similar spatial configurations
- **FUNCTIONAL**: Patterns serve similar purposes or roles
- **COMPOSITIONAL**: One pattern contains or is part of the other

**Association Structure:**
```
Association
├── Source Pattern ID
├── Target Pattern ID
├── Type (CAUSAL, CATEGORICAL, SPATIAL, etc.)
├── Strength (0.0 - 1.0)
├── Creation timestamp
├── Last activation timestamp
└── Activation count
```

**Example:**
```
"Hello" → "How are you?" [CAUSAL, strength: 0.92]
"Hello" → "Hi" [CATEGORICAL, strength: 0.85]
```

### Learning Process

DPAN learns through a continuous cycle:

1. **Input Processing**
   - Raw bytes → Feature extraction → Pattern creation/matching
   - New patterns created if no match found above threshold

2. **Activation Recording**
   - When a pattern is matched, it's "activated"
   - Activation history is maintained with timestamps
   - Co-occurrence statistics are updated

3. **Association Formation**
   - Patterns that frequently co-occur form associations
   - Formation rules determine when/how associations are created
   - Type inference based on temporal/spatial relationships

4. **Reinforcement**
   - Successful predictions strengthen associations
   - Failed predictions weaken associations
   - Confidence scores updated based on accuracy

5. **Competition**
   - Associations compete for limited "strength budget"
   - Winner-take-all dynamics favor stronger associations
   - Prevents over-generalization

6. **Maintenance**
   - Time-based decay weakens unused associations
   - Weak associations below threshold are pruned
   - Strength normalization prevents inflation
   - Pattern consolidation merges similar patterns

### Context Vectors

**Purpose:** Represent the situational context in which patterns/associations are relevant.

Context is represented as a sparse multi-dimensional vector where each dimension has a name and value:

```cpp
ContextVector context;
context.Set("topic", 0.8);        // Conversation topic relevance
context.Set("formality", 0.6);    // Formality level
context.Set("sentiment", 0.3);    // Emotional tone
context.Set("time_of_day", 0.4);  // Temporal context
```

**Uses:**
- Context-sensitive pattern matching
- Context-aware predictions
- Association filtering by context
- Attention mechanism input

### Attention Mechanisms

**Purpose:** Dynamically weight patterns based on relevance, importance, and context.

Instead of treating all patterns equally, attention mechanisms compute dynamic importance scores:

```
Traditional: prediction = argmax(association_strength)

With Attention: prediction = argmax(
    association_weight * association_strength +
    attention_weight * attention_score
)
```

**Components:**
- **Query-Key Similarity**: How relevant is this pattern to the current query?
- **Pattern Importance**: Intrinsic value based on usage, confidence, associations
- **Context Alignment**: How well does it fit the current context?
- **Multi-Head Perspectives**: Multiple attention viewpoints combined

**Benefits:**
- Improved prediction accuracy
- Context-aware responses
- Better handling of ambiguity
- Interpretable attention weights

---

## Installation and Building

### Standard Build

```bash
# 1. Navigate to project directory
cd /path/to/dpan

# 2. Create build directory
mkdir -p build
cd build

# 3. Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_CLI=ON

# 4. Build
make -j$(nproc)

# 5. Optionally install system-wide
sudo make install
```

### Build Options

**CMAKE Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`, `Release`, `RelWithDebInfo` |
| `BUILD_TESTS` | `ON` | Build test suite |
| `BUILD_CLI` | `ON` | Build interactive CLI |
| `BUILD_EXAMPLES` | `OFF` | Build example programs |
| `ENABLE_BENCHMARKS` | `OFF` | Build performance benchmarks |
| `ENABLE_ASAN` | `OFF` | Enable AddressSanitizer (debug) |
| `ENABLE_TSAN` | `OFF` | Enable ThreadSanitizer (debug) |

**Example Debug Build:**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_ASAN=ON \
    -DBUILD_TESTS=ON

make -j$(nproc)
```

### Platform-Specific Instructions

**Linux (Ubuntu/Debian):**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libsqlite3-dev \
    git

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**macOS:**
```bash
# Install dependencies via Homebrew
brew install cmake sqlite3

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

**Windows (MSVC):**
```cmd
# Install dependencies (vcpkg)
vcpkg install sqlite3:x64-windows

# Build
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### Running Tests

```bash
# After building, from build directory:
ctest --output-on-failure

# Run specific test
./tests/core/pattern_engine_test

# Run with verbose output
ctest -V

# Run tests matching pattern
ctest -R "association.*" -V
```

Expected output:
```
Test project /home/user/nn/build
      Start  1: CoreTypesTest
 1/65 Test  #1: CoreTypesTest ....................   Passed    0.02 sec
      Start  2: PatternIDTest
 2/65 Test  #2: PatternIDTest ....................   Passed    0.01 sec
...
100% tests passed, 0 tests failed out of 444
```

---

## Using DPAN as a Library

### Minimal Example

```cpp
#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    // 1. Create pattern engine
    dpan::PatternEngine::Config engine_config;
    engine_config.database_type = "memory";
    dpan::PatternEngine engine(engine_config);

    // 2. Create association learning system
    dpan::AssociationLearningSystem::Config assoc_config;
    dpan::AssociationLearningSystem assoc_system(assoc_config);

    // 3. Process some inputs
    std::vector<std::string> inputs = {
        "Hello",
        "How are you?",
        "I am fine",
        "Thank you"
    };

    for (const auto& text : inputs) {
        // Convert to bytes
        std::vector<uint8_t> bytes(text.begin(), text.end());

        // Process input
        auto result = engine.ProcessInput(bytes, dpan::DataModality::TEXT);

        // Record activation
        if (!result.activated_patterns.empty()) {
            assoc_system.RecordPatternActivation(result.activated_patterns[0]);
        }

        std::cout << "Processed: " << text << std::endl;
    }

    // 4. Form associations
    // Note: Get the database pointer from engine
    auto db_ptr = engine.GetDatabase();  // Assuming this method exists
    size_t formed = assoc_system.FormNewAssociations(*db_ptr);
    std::cout << "Formed " << formed << " associations" << std::endl;

    // 5. Get statistics
    auto stats = engine.GetStatistics();
    std::cout << "Total patterns: " << stats.total_patterns << std::endl;

    return 0;
}
```

**Compile and link:**
```bash
g++ -std=c++17 -I/path/to/dpan/src -I/path/to/dpan/include \
    my_program.cpp \
    -L/path/to/dpan/build/src/core -ldpan_core \
    -L/path/to/dpan/build/src/association -ldpan_association \
    -lsqlite3 -lpthread \
    -o my_program
```

### Advanced Library Usage

**1. Pattern Discovery and Matching:**

```cpp
#include "core/pattern_engine.hpp"

// Configure with custom thresholds
dpan::PatternEngine::Config config;
config.matching_config.similarity_threshold = 0.70f;  // Higher = stricter matching
config.matching_config.strong_match_threshold = 0.85f;
config.extraction_config.feature_dimension = 128;     // Larger = more detail

dpan::PatternEngine engine(config);

// Process input
std::vector<uint8_t> input = /* ... */;
auto result = engine.ProcessInput(input, dpan::DataModality::TEXT);

// Check what happened
if (!result.created_patterns.empty()) {
    std::cout << "Created " << result.created_patterns.size()
              << " new patterns" << std::endl;
}

if (!result.activated_patterns.empty()) {
    std::cout << "Matched existing pattern: "
              << result.activated_patterns[0].ToString() << std::endl;
}
```

**2. Similarity Search:**

```cpp
// Find similar patterns
std::vector<uint8_t> query = /* ... */;
auto query_data = dpan::PatternData::FromBytes(query, dpan::DataModality::TEXT);

auto results = engine.FindSimilarPatterns(query_data, 10, 0.5f);

std::cout << "Top 10 similar patterns:" << std::endl;
for (const auto& result : results) {
    auto pattern = engine.GetPattern(result.pattern_id);
    if (pattern) {
        std::cout << "  Similarity: " << result.similarity_score
                  << " Confidence: " << pattern->GetConfidence() << std::endl;
    }
}
```

**3. Association Learning with Context:**

```cpp
#include "association/association_learning_system.hpp"

dpan::AssociationLearningSystem system;

// Record activations with context
dpan::ContextVector context;
context.Set("topic", 0.8f);
context.Set("formality", 0.6f);

for (const auto& pattern_id : activated_patterns) {
    system.RecordPatternActivation(pattern_id, context);
}

// Form associations
size_t formed = system.FormNewAssociations(pattern_database);

// Predict with context
auto predictions = system.PredictWithConfidence(current_pattern, 5, &context);

for (const auto& [pattern_id, confidence] : predictions) {
    std::cout << "Predicted: " << pattern_id.ToString()
              << " Confidence: " << confidence << std::endl;
}
```

**4. Persistent Sessions:**

```cpp
#include "storage/persistent_backend.hpp"

// Create persistent storage
dpan::PersistentBackend::Config db_config;
db_config.db_path = "my_session.db";
db_config.enable_wal = true;
auto db = std::make_shared<dpan::PersistentBackend>(db_config);

// Use with pattern engine
dpan::PatternEngine::Config engine_config;
engine_config.database_type = "persistent";
engine_config.database_path = "my_session.db";
dpan::PatternEngine engine(engine_config);

// ... learn patterns ...

// Save associations separately
dpan::AssociationLearningSystem system;
system.Save("associations.dat");

// Later, restore session
dpan::PatternEngine engine2(engine_config);  // Loads patterns automatically
dpan::AssociationLearningSystem system2;
system2.Load("associations.dat");
```

---

## Using the CLI Interface

The DPAN CLI provides an interactive interface for training and testing. See [DPAN_CLI_Guide.md](DPAN_CLI_Guide.md) for complete CLI documentation.

**Quick Reference:**

```bash
# Start CLI
./build/src/cli/dpan_cli

# Optional: Specify config file
./build/src/cli/dpan_cli --config my_config.yaml
```

**Common Commands:**

| Command | Description |
|---------|-------------|
| `<text>` | Learn from and respond to text |
| `/predict <text>` | Show predictions |
| `/learn <file>` | Batch learn from file |
| `/active` | Toggle active learning mode |
| `/stats` | Show statistics |
| `/patterns` | List learned patterns |
| `/save` | Save session |
| `/reset` | Clear all data |
| `/help` | Show help |
| `exit` | Exit (auto-saves) |

**Configuration File (dpan_config.yaml):**

```yaml
# Interface settings
prompt: "dpan> "
colors_enabled: true
verbose: false
session_file: "dpan_session.db"

# Learning settings
active_learning: false
attention_enabled: false

# Pattern engine
pattern_engine:
  similarity_metric: "context"
  feature_dimension: 64
  similarity_threshold: 0.60
  strong_match_threshold: 0.75

# Association learning
association:
  min_co_occurrences: 2
  decay_rate: 0.01
  window_size_seconds: 300

# Attention mechanism
attention:
  num_heads: 4
  temperature: 1.0
  association_weight: 0.6
  attention_weight: 0.4
```

---

## Configuration Guide

### Pattern Engine Configuration

**Key Parameters:**

```cpp
// Feature extraction
config.extraction_config.feature_dimension = 64;      // Feature vector size (32-512)
config.extraction_config.min_pattern_size = 1;        // Min bytes to extract
config.extraction_config.max_pattern_size = 1000;     // Max bytes to extract

// Pattern matching
config.matching_config.similarity_threshold = 0.60f;  // Match threshold (0.0-1.0)
config.matching_config.strong_match_threshold = 0.75f;  // Strong match threshold

// Similarity metric
config.similarity_metric = "context";  // "context", "cosine", "euclidean"

// Performance options
config.enable_auto_refinement = true;  // Automatic pattern optimization
config.enable_indexing = true;         // Fast pattern lookups
```

**Tuning Guidelines:**

| Goal | Adjustment |
|------|-----------|
| More strict matching | Increase `similarity_threshold` |
| More lenient matching | Decrease `similarity_threshold` |
| Higher precision | Increase `feature_dimension` |
| Faster processing | Decrease `feature_dimension` |
| Handle longer inputs | Increase `max_pattern_size` |

### Association Learning Configuration

**Key Parameters:**

```cpp
// Co-occurrence tracking
config.co_occurrence.window_size = std::chrono::seconds(300);  // 5 minutes
config.co_occurrence.min_co_occurrences = 2;  // Min count to form association

// Association formation
config.formation.min_co_occurrence_count = 2;
config.formation.min_co_occurrence_strength = 0.3f;

// Reinforcement
config.reinforcement.learning_rate = 0.1f;          // How fast to adjust (0.0-1.0)
config.reinforcement.positive_reward = 0.1f;        // Boost for correct predictions
config.reinforcement.negative_penalty = 0.05f;      // Penalty for wrong predictions

// Competition
config.competition.competition_strength = 0.5f;     // How aggressive (0.0-1.0)
config.competition.winner_boost = 0.2f;             // Winner advantage

// Maintenance
config.prune_threshold = 0.05f;                     // Remove associations < 0.05
config.enable_auto_maintenance = true;
config.auto_decay_interval = std::chrono::hours(1);
```

**Tuning Guidelines:**

| Goal | Adjustment |
|------|-----------|
| Form associations faster | Decrease `min_co_occurrences` |
| Form only strong associations | Increase `min_co_occurrence_strength` |
| Learn quickly | Increase `learning_rate` |
| Learn conservatively | Decrease `learning_rate` |
| Stronger competition | Increase `competition_strength` |
| Keep more associations | Decrease `prune_threshold` |

### Attention Configuration

**Key Parameters:**

```cpp
config.num_heads = 4;                    // Number of attention heads (1-16)
config.temperature = 1.0f;               // Softmax temperature (0.1-2.0)
config.use_context = true;               // Context-aware attention
config.use_importance = true;            // Pattern importance weighting

config.association_weight = 0.6f;        // Weight for association strength
config.attention_weight = 0.4f;          // Weight for attention score

config.enable_caching = true;            // Cache attention computations
config.cache_size = 1000;                // LRU cache size
```

**Tuning Guidelines:**

| Goal | Adjustment |
|------|-----------|
| Diverse perspectives | Increase `num_heads` |
| Focused attention | Decrease `temperature` |
| Uniform attention | Increase `temperature` |
| Favor associations | Increase `association_weight` |
| Favor attention | Increase `attention_weight` |
| Faster computation | Enable caching, reduce heads |

---

## Training Strategies

### Strategy 1: Interactive Conversational Training

**Goal:** Train DPAN through natural dialogue.

```bash
# Start CLI with active learning
./dpan_cli

dpan> /active
Active learning mode: ON

dpan> Hello
→ [Learning...]

dpan> Hi there!
→ [Learning...]

dpan> Hello
→ Hi there! [confidence: 0.75]
```

**Best for:** Small-scale learning, testing, demonstrations

### Strategy 2: Batch File Training

**Goal:** Learn from large text corpora.

**Create training file (conversation.txt):**
```
Hello
Hi there!
How are you?
I'm doing great, thanks!
What's your name?
My name is Alice
Nice to meet you
```

**Load in CLI:**
```bash
dpan> /learn conversation.txt
Learning from file: conversation.txt
✓ Learned from 7 lines in 45 ms
  Patterns created: 7
```

**Best for:** Large-scale training, reproducible experiments

### Strategy 3: Programmatic Training Loop

**Goal:** Custom training pipeline with preprocessing.

```cpp
// Load training data
auto corpus = LoadCorpus("training_data.txt");

// Preprocess
for (auto& text : corpus) {
    text = Normalize(text);  // Lowercase, trim, etc.
}

// Train in batches
dpan::PatternEngine engine(/* config */);
dpan::AssociationLearningSystem system(/* config */);

std::vector<dpan::PatternID> batch;
for (size_t i = 0; i < corpus.size(); ++i) {
    auto bytes = TextToBytes(corpus[i]);
    auto result = engine.ProcessInput(bytes, dpan::DataModality::TEXT);

    if (!result.activated_patterns.empty()) {
        batch.push_back(result.activated_patterns[0]);
    }

    // Process batch every 100 inputs
    if (i % 100 == 0 && !batch.empty()) {
        system.RecordPatternActivations(batch);
        batch.clear();

        // Form associations every 1000 inputs
        if (i % 1000 == 0) {
            system.FormNewAssociations(engine.GetDatabase());
        }
    }
}

// Final association formation
system.FormNewAssociations(engine.GetDatabase());

// Evaluate
auto stats = system.GetStatistics();
std::cout << "Training complete:" << std::endl;
std::cout << "  Patterns: " << engine.GetStatistics().total_patterns << std::endl;
std::cout << "  Associations: " << stats.total_associations << std::endl;
std::cout << "  Avg strength: " << stats.average_strength << std::endl;
```

**Best for:** Production systems, custom workflows

### Strategy 4: Reinforcement-Based Training

**Goal:** Improve prediction quality through feedback.

```cpp
dpan::PatternID prev_pattern;

while (HasMoreInputs()) {
    auto input = GetNextInput();
    auto result = engine.ProcessInput(input, dpan::DataModality::TEXT);

    if (!result.activated_patterns.empty()) {
        auto current = result.activated_patterns[0];

        // If we had a previous pattern, check our prediction
        if (prev_pattern.IsValid()) {
            auto predictions = system.Predict(prev_pattern, 1);

            if (!predictions.empty()) {
                bool correct = (predictions[0] == current);
                system.Reinforce(predictions[0], current, correct);

                if (correct) {
                    std::cout << "✓ Correct prediction" << std::endl;
                } else {
                    std::cout << "✗ Wrong prediction" << std::endl;
                }
            }
        }

        system.RecordPatternActivation(current);
        prev_pattern = current;
    }
}
```

**Best for:** Online learning, adaptive systems

### Monitoring Training Progress

```cpp
// Periodic statistics check
void MonitorProgress(dpan::PatternEngine& engine,
                     dpan::AssociationLearningSystem& system) {
    auto engine_stats = engine.GetStatistics();
    auto assoc_stats = system.GetStatistics();

    std::cout << "=== Training Progress ===" << std::endl;
    std::cout << "Patterns: " << engine_stats.total_patterns << std::endl;
    std::cout << "Associations: " << assoc_stats.total_associations << std::endl;
    std::cout << "Avg confidence: " << engine_stats.avg_confidence << std::endl;
    std::cout << "Avg strength: " << assoc_stats.average_strength << std::endl;

    // Check for good learning
    if (assoc_stats.average_strength > 0.65f) {
        std::cout << "✓ Strong associations forming" << std::endl;
    }
    if (assoc_stats.total_associations > engine_stats.total_patterns * 2) {
        std::cout << "✓ Rich connection network" << std::endl;
    }
}
```

---

## Advanced Features

### Multi-Head Attention

**Purpose:** Combine multiple attention perspectives for better predictions.

```cpp
#include "learning/multi_head_attention.hpp"

// Configure multi-head attention
dpan::AttentionConfig config;
config.num_heads = 4;  // Semantic, temporal, structural, association heads
config.temperature = 1.0f;

auto attention = std::make_unique<dpan::MultiHeadAttention>(config);
attention->SetPatternDatabase(db);

// Use with association system
assoc_system.SetAttentionMechanism(attention.get());

// Predictions now use multi-head attention
auto predictions = assoc_system.PredictWithAttention(pattern, 5, context);
```

**Attention Heads:**
- **Semantic**: Content-based similarity
- **Temporal**: Recency-based scoring
- **Structural**: Pattern structure alignment
- **Association**: Association-strength weighting

### Pattern Importance Scoring

```cpp
#include "learning/pattern_importance.hpp"

dpan::PatternImportanceCalculator importance_calc;

// Compute importance for a pattern
float importance = importance_calc.ComputeImportance(pattern_id, db, assoc_matrix);

// Importance considers:
// - Access frequency
// - Confidence score
// - Association richness (number and strength of connections)
// - Prediction success rate
// - Recency of use

// Use for prioritization
std::vector<dpan::PatternID> all_patterns = engine.GetAllPatternIDs();
std::sort(all_patterns.begin(), all_patterns.end(),
    [&](dpan::PatternID a, dpan::PatternID b) {
        return importance_calc.ComputeImportance(a, db, assoc_matrix) >
               importance_calc.ComputeImportance(b, db, assoc_matrix);
    });
```

### Memory Tiering

**Purpose:** Manage large pattern sets with tiered storage.

```cpp
#include "memory/tiered_storage.hpp"

dpan::TieredStorage::Config tier_config;
tier_config.active_capacity = 10000;     // Hot cache
tier_config.warm_capacity = 50000;       // Warm cache
tier_config.cold_capacity = 500000;      // Cold storage
tier_config.archive_threshold = 0.01f;   // Archive utility < 0.01

dpan::TieredStorage tiered_storage(tier_config, db);

// Patterns automatically move between tiers based on utility
// Active tier: Frequently accessed, high confidence
// Warm tier: Occasionally used
// Cold tier: Rarely used but kept
// Archive: Very low utility, stored on disk
```

### Context Tracking

**Purpose:** Maintain dynamic context based on conversation history.

```cpp
#include "cli/context_tracker.hpp"

dpan::ContextTracker tracker;

// Update context based on activations
for (const auto& pattern_id : activated_patterns) {
    auto pattern = engine.GetPattern(pattern_id);
    if (pattern && pattern->HasMetadata("topic")) {
        std::string topic = pattern->GetMetadata("topic");
        tracker.IncrementTopic(topic, 1.0f);
    }
}

// Context decays over time
tracker.ApplyDecay(std::chrono::seconds(30));

// Get current context vector
dpan::ContextVector context = tracker.GetContextVector();

// Use for context-aware predictions
auto predictions = system.PredictWithConfidence(pattern, 5, &context);
```

---

## Performance Tuning

### Optimization Checklist

**1. Enable Indexing**
```cpp
config.enable_indexing = true;
```
- Enables spatial, temporal, and similarity indices
- 10-100x faster pattern search
- Small memory overhead

**2. Use Batch Operations**
```cpp
// Good: Batch operations
engine.GetPatternsBatch(ids);
system.RecordPatternActivations(patterns);

// Avoid: Individual operations in loop
for (auto id : ids) {
    engine.GetPattern(id);  // Slow!
}
```

**3. Configure Cache Sizes**
```cpp
// Pattern similarity cache
config.cache_size_kb = 10240;  // 10MB

// Attention cache
attention_config.enable_caching = true;
attention_config.cache_size = 1000;
```

**4. Tune Maintenance Frequency**
```cpp
// Don't run maintenance after every operation
if (operation_count % 1000 == 0) {
    system.PerformMaintenance();
}
```

**5. Limit Association Capacity**
```cpp
config.association_capacity = 1000000;  // 1M associations max
config.prune_threshold = 0.05f;         // Prune aggressively
```

### Memory Management

**Monitor Memory Usage:**
```cpp
auto stats = engine.GetStatistics();
std::cout << "Storage size: " << stats.storage_stats.total_size_bytes << " bytes" << std::endl;

auto assoc_stats = system.GetStatistics();
std::cout << "Associations: " << assoc_stats.total_associations << std::endl;
```

**Reduce Memory Footprint:**
```cpp
// 1. Prune weak associations
system.PruneWeakAssociations(0.1f);  // More aggressive

// 2. Compact database
engine.Compact();

// 3. Use smaller feature dimension
config.extraction_config.feature_dimension = 32;  // vs 64 or 128

// 4. Limit pattern size
config.extraction_config.max_pattern_size = 500;  // vs 1000

// 5. Use tiered storage for large datasets
```

### Benchmarking

```bash
# Build benchmarks
cmake .. -DENABLE_BENCHMARKS=ON
make benchmarks

# Run benchmarks
./benchmarks/pattern_engine_bench
./benchmarks/association_bench
./benchmarks/attention_bench
```

**Expected Performance (typical hardware):**
- Pattern creation: ~100k patterns/sec
- Pattern matching: ~200k matches/sec
- Association formation: ~50k associations/sec
- Prediction: ~1M predictions/sec (without attention)
- Prediction: ~100k predictions/sec (with attention)
- Attention computation: ~50k attention scores/sec

---

## Troubleshooting

### Common Issues

**Issue: Patterns not matching when they should**

**Symptoms:**
- Creating duplicate patterns for similar inputs
- `activated_patterns` always empty

**Solutions:**
```cpp
// 1. Lower similarity threshold
config.matching_config.similarity_threshold = 0.50f;  // More lenient

// 2. Check feature dimension
config.extraction_config.feature_dimension = 64;  // Higher = more precision

// 3. Verify similarity metric
config.similarity_metric = "context";  // Try different metrics
```

**Issue: Associations not forming**

**Symptoms:**
- `FormNewAssociations()` returns 0
- No predictions available

**Solutions:**
```cpp
// 1. Lower co-occurrence requirements
config.co_occurrence.min_co_occurrences = 1;  // Form faster

// 2. Increase time window
config.co_occurrence.window_size = std::chrono::seconds(600);  // 10 minutes

// 3. Ensure patterns are activated
system.RecordPatternActivation(pattern_id);  // Don't forget this!

// 4. Check if maintenance is pruning too aggressively
config.prune_threshold = 0.01f;  // Keep more associations
```

**Issue: Poor prediction quality**

**Symptoms:**
- Predictions seem random
- Low confidence scores

**Solutions:**
```cpp
// 1. Need more training data
// - Associations strengthen with reinforcement
// - Need diverse examples

// 2. Enable attention mechanisms
system.SetAttentionMechanism(&attention);

// 3. Use context-aware predictions
dpan::ContextVector context;
// ... set context ...
auto predictions = system.PredictWithConfidence(pattern, 5, &context);

// 4. Check association strengths
auto stats = system.GetStatistics();
if (stats.average_strength < 0.5f) {
    // Associations are weak, need more training or reinforcement
}
```

**Issue: High memory usage**

**Solutions:**
```cpp
// 1. Enable automatic pruning
config.enable_auto_maintenance = true;
config.prune_threshold = 0.1f;  // More aggressive

// 2. Limit capacities
config.association_capacity = 500000;
config.max_activation_history = 5000;

// 3. Compact regularly
engine.Compact();

// 4. Use persistent storage and smaller cache
config.database_type = "persistent";
config.cache_size_kb = 5120;  // 5MB vs 10MB
```

**Issue: Database corruption**

**Symptoms:**
- SQLite errors on startup
- Cannot load session

**Solutions:**
```bash
# 1. Check database integrity
sqlite3 dpan_session.db "PRAGMA integrity_check;"

# 2. Restore from backup if available
cp dpan_session.db.backup dpan_session.db

# 3. Reset and start fresh
rm dpan_session.db*
./dpan_cli  # Creates new database
```

### Debug Mode

**Enable verbose output:**
```bash
# CLI
dpan> /verbose
Verbose mode: ON

# Library
engine.SetVerbose(true);
system.SetVerbose(true);
```

**Build with debug symbols:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
make
```

**Check for memory leaks:**
```bash
valgrind --leak-check=full ./dpan_cli
```

---

## Best Practices

### Code Organization

**1. Separate Configuration**
```cpp
// config.hpp
struct AppConfig {
    dpan::PatternEngine::Config engine;
    dpan::AssociationLearningSystem::Config associations;
    dpan::AttentionConfig attention;

    static AppConfig LoadFromFile(const std::string& path);
    void SaveToFile(const std::string& path) const;
};

// main.cpp
auto config = AppConfig::LoadFromFile("config.yaml");
dpan::PatternEngine engine(config.engine);
dpan::AssociationLearningSystem system(config.associations);
```

**2. Encapsulate DPAN Components**
```cpp
class DPANManager {
public:
    DPANManager(const Config& config);

    void ProcessInput(const std::string& text);
    std::vector<std::string> Predict(const std::string& text, size_t k);
    void SaveSession(const std::string& path);
    void LoadSession(const std::string& path);

private:
    std::unique_ptr<dpan::PatternEngine> engine_;
    std::unique_ptr<dpan::AssociationLearningSystem> system_;
    std::unique_ptr<dpan::AttentionMechanism> attention_;
};
```

### Error Handling

```cpp
// Check optional returns
auto pattern = engine.GetPattern(id);
if (!pattern) {
    std::cerr << "Pattern not found: " << id.ToString() << std::endl;
    return;
}

// Check boolean returns
if (!engine.UpdatePattern(id, new_data)) {
    std::cerr << "Failed to update pattern" << std::endl;
}

// Catch exceptions from storage
try {
    auto db = std::make_shared<dpan::PersistentBackend>(config);
} catch (const std::exception& e) {
    std::cerr << "Database error: " << e.what() << std::endl;
    return;
}
```

### Testing

```cpp
// Unit test example
TEST(PatternEngineTest, ProcessInput) {
    dpan::PatternEngine::Config config;
    dpan::PatternEngine engine(config);

    std::string text = "test input";
    std::vector<uint8_t> bytes(text.begin(), text.end());

    auto result = engine.ProcessInput(bytes, dpan::DataModality::TEXT);

    ASSERT_FALSE(result.activated_patterns.empty() &&
                 result.created_patterns.empty());
}
```

### Documentation

```cpp
/// Process user input and update patterns/associations
///
/// @param text User input text
/// @return Predicted response text (if available)
///
/// This method:
/// 1. Converts text to pattern
/// 2. Records activation
/// 3. Forms associations if needed
/// 4. Generates prediction
std::optional<std::string> ProcessUserInput(const std::string& text);
```

---

## Examples and Tutorials

### Tutorial 1: Building a Simple Chatbot

```cpp
#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"
#include <iostream>
#include <string>

class SimpleChatbot {
public:
    SimpleChatbot() {
        // Configure components
        dpan::PatternEngine::Config engine_config;
        engine_config.database_type = "memory";
        engine_ = std::make_unique<dpan::PatternEngine>(engine_config);

        dpan::AssociationLearningSystem::Config assoc_config;
        assoc_config.enable_auto_maintenance = true;
        system_ = std::make_unique<dpan::AssociationLearningSystem>(assoc_config);
    }

    void Train(const std::vector<std::string>& conversation) {
        dpan::PatternID prev;

        for (const auto& text : conversation) {
            auto id = ProcessText(text);
            if (id.IsValid() && prev.IsValid()) {
                // Association will form between prev and current
            }
            prev = id;
        }

        // Form associations
        system_->FormNewAssociations(*engine_->GetDatabase());
        std::cout << "Training complete" << std::endl;
    }

    std::string Respond(const std::string& input) {
        auto input_id = ProcessText(input);
        if (!input_id.IsValid()) {
            return "I don't understand.";
        }

        // Predict response
        auto predictions = system_->PredictWithConfidence(input_id, 1);
        if (predictions.empty()) {
            return "I'm still learning...";
        }

        // Get predicted pattern
        auto response_pattern = engine_->GetPattern(predictions[0].first);
        if (!response_pattern) {
            return "Error retrieving response.";
        }

        // Convert back to text (simplified)
        auto data = response_pattern->GetData().GetRawData();
        return std::string(data.begin(), data.end());
    }

private:
    dpan::PatternID ProcessText(const std::string& text) {
        std::vector<uint8_t> bytes(text.begin(), text.end());
        auto result = engine_->ProcessInput(bytes, dpan::DataModality::TEXT);

        if (!result.activated_patterns.empty()) {
            system_->RecordPatternActivation(result.activated_patterns[0]);
            return result.activated_patterns[0];
        }
        return dpan::PatternID();
    }

    std::unique_ptr<dpan::PatternEngine> engine_;
    std::unique_ptr<dpan::AssociationLearningSystem> system_;
};

int main() {
    SimpleChatbot bot;

    // Train
    std::vector<std::string> training = {
        "Hello", "Hi there!",
        "How are you?", "I'm fine, thanks!",
        "Goodbye", "See you later!"
    };
    bot.Train(training);

    // Chat
    std::cout << "Chatbot ready. Type 'quit' to exit." << std::endl;
    while (true) {
        std::cout << "You: ";
        std::string input;
        std::getline(std::cin, input);

        if (input == "quit") break;

        std::string response = bot.Respond(input);
        std::cout << "Bot: " << response << std::endl;
    }

    return 0;
}
```

### Tutorial 2: Sequence Prediction

```cpp
// Predict next element in numeric sequences

#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"

class SequencePredictor {
public:
    SequencePredictor() {
        // Setup...
    }

    void TrainSequence(const std::vector<int>& sequence) {
        dpan::PatternID prev;

        for (int value : sequence) {
            // Convert number to bytes
            std::vector<uint8_t> bytes(sizeof(int));
            std::memcpy(bytes.data(), &value, sizeof(int));

            auto result = engine_->ProcessInput(bytes, dpan::DataModality::NUMERIC);
            if (!result.activated_patterns.empty()) {
                auto current = result.activated_patterns[0];
                system_->RecordPatternActivation(current);

                if (prev.IsValid()) {
                    // Reinforce if we predicted correctly
                    auto predictions = system_->Predict(prev, 1);
                    if (!predictions.empty()) {
                        bool correct = (predictions[0] == current);
                        system_->Reinforce(predictions[0], current, correct);
                    }
                }

                prev = current;
            }
        }

        system_->FormNewAssociations(*engine_->GetDatabase());
    }

    std::optional<int> PredictNext(int current) {
        // Convert to pattern
        std::vector<uint8_t> bytes(sizeof(int));
        std::memcpy(bytes.data(), &current, sizeof(int));

        auto result = engine_->ProcessInput(bytes, dpan::DataModality::NUMERIC);
        if (result.activated_patterns.empty()) {
            return std::nullopt;
        }

        // Predict
        auto predictions = system_->Predict(result.activated_patterns[0], 1);
        if (predictions.empty()) {
            return std::nullopt;
        }

        // Convert back to int
        auto pattern = engine_->GetPattern(predictions[0]);
        if (!pattern) return std::nullopt;

        auto data = pattern->GetData().GetRawData();
        int value;
        std::memcpy(&value, data.data(), sizeof(int));
        return value;
    }

private:
    std::unique_ptr<dpan::PatternEngine> engine_;
    std::unique_ptr<dpan::AssociationLearningSystem> system_;
};

int main() {
    SequencePredictor predictor;

    // Train on sequences
    std::vector<int> fibonacci = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55};
    predictor.TrainSequence(fibonacci);

    // Predict next
    auto next = predictor.PredictNext(55);
    if (next) {
        std::cout << "Predicted next: " << *next << std::endl;
    }

    return 0;
}
```

---

## Conclusion

DPAN provides a powerful framework for building self-organizing neural networks that learn from unlabeled data. This guide covered:

- **Getting started** with building and basic usage
- **Core concepts** of patterns, associations, and learning
- **Library usage** for programmatic control
- **CLI interface** for interactive training
- **Configuration** for tuning behavior
- **Advanced features** like attention and tiering
- **Performance optimization** strategies
- **Troubleshooting** common issues
- **Best practices** for production use

### Next Steps

1. **Explore Examples**: See `examples/` directory for more code samples
2. **Read API Reference**: [DPAN_API_Reference.md](DPAN_API_Reference.md)
3. **Read CLI Guide**: [DPAN_CLI_Guide.md](DPAN_CLI_Guide.md)
4. **Join Community**: Contribute on GitHub, report issues, share feedback

### Resources

- **GitHub**: https://github.com/yourusername/dpan
- **Documentation**: https://dpan.readthedocs.io
- **Issues**: https://github.com/yourusername/dpan/issues
- **Discussions**: https://github.com/yourusername/dpan/discussions

---

**Happy learning with DPAN!**
