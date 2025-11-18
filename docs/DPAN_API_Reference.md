# DPAN API Reference

**Version:** 1.0
**Date:** 2025-11-18

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
  - [PatternID](#patternid)
  - [Timestamp](#timestamp)
  - [ContextVector](#contextvector)
  - [PatternType](#patterntype)
  - [AssociationType](#associationtype)
  - [DataModality](#datamodality)
- [Core Classes](#core-classes)
  - [PatternData](#patterndata)
  - [FeatureVector](#featurevector)
  - [PatternNode](#patternnode)
  - [AssociationEdge](#associationedge)
- [Pattern Engine](#pattern-engine)
  - [Configuration](#patternengine-configuration)
  - [Pattern Discovery](#pattern-discovery)
  - [Pattern Retrieval](#pattern-retrieval)
  - [Pattern Search](#pattern-search)
  - [Pattern Management](#pattern-management)
  - [Statistics](#patternengine-statistics)
  - [Maintenance](#patternengine-maintenance)
- [Association Learning System](#association-learning-system)
  - [Configuration](#associationlearningsystem-configuration)
  - [Recording Activations](#recording-activations)
  - [Association Formation](#association-formation)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Maintenance Operations](#maintenance-operations)
  - [Prediction](#prediction)
  - [Statistics](#associationlearningsystem-statistics)
- [Attention Mechanism](#attention-mechanism)
  - [Configuration](#attention-configuration)
  - [Computing Attention](#computing-attention)
  - [Applying Attention](#applying-attention)
- [Storage Backends](#storage-backends)
  - [PatternDatabase Interface](#patterndatabase-interface)
  - [MemoryBackend](#memorybackend)
  - [PersistentBackend](#persistentbackend)
- [Complete Examples](#complete-examples)

---

## Overview

The DPAN (Dynamic Pattern Association Network) API provides a comprehensive interface for building self-organizing neural networks that learn patterns and associations from raw input without labeled data.

**Key Features:**
- Pattern discovery and extraction from raw bytes
- Similarity-based pattern matching
- Association learning with multiple learning strategies
- Attention mechanisms for context-aware prediction
- Persistent storage with SQLite backend
- Thread-safe operations throughout
- Comprehensive statistics and monitoring

**Namespace:** All DPAN classes and functions are in the `dpan` namespace.

**Threading:** All public APIs are thread-safe unless explicitly noted otherwise.

---

## Core Types

### PatternID

**Purpose:** Unique 64-bit identifier for patterns.

**Definition:**
```cpp
class PatternID {
public:
    using ValueType = uint64_t;

    PatternID();                          // Invalid ID
    explicit PatternID(ValueType value);  // From value

    static PatternID Generate();          // Thread-safe unique ID

    bool IsValid() const;
    ValueType value() const;
    std::string ToString() const;

    // Comparison operators
    bool operator==(const PatternID& other) const;
    bool operator!=(const PatternID& other) const;
    bool operator<(const PatternID& other) const;

    // Hash support for std::unordered_map
    struct Hash {
        size_t operator()(const PatternID& id) const;
    };
};
```

**Usage:**
```cpp
// Generate new unique ID
PatternID id = PatternID::Generate();

// Check validity
if (id.IsValid()) {
    std::cout << "Pattern ID: " << id.ToString() << std::endl;
}

// Use in containers
std::unordered_map<PatternID, PatternNode, PatternID::Hash> pattern_map;
std::set<PatternID> pattern_set;
```

---

### Timestamp

**Purpose:** Microsecond-precision time tracking.

**Definition:**
```cpp
class Timestamp {
public:
    using ClockType = std::chrono::steady_clock;
    using TimePoint = ClockType::time_point;
    using Duration = std::chrono::microseconds;

    static Timestamp Now();
    static Timestamp FromMicros(int64_t micros);

    Timestamp();  // Zero timestamp

    int64_t ToMicros() const;
    std::string ToString() const;

    // Arithmetic operators
    Duration operator-(const Timestamp& other) const;
    Timestamp operator+(Duration duration) const;
    Timestamp operator-(Duration duration) const;

    // Comparison operators
    bool operator<(const Timestamp& other) const;
    bool operator>(const Timestamp& other) const;
    bool operator==(const Timestamp& other) const;
    bool operator!=(const Timestamp& other) const;
};
```

**Usage:**
```cpp
// Get current time
Timestamp now = Timestamp::Now();

// Measure elapsed time
Timestamp start = Timestamp::Now();
// ... do work ...
Timestamp end = Timestamp::Now();
auto elapsed = end - start;
std::cout << "Elapsed: " << elapsed.count() << " microseconds" << std::endl;

// Create from microseconds
Timestamp t = Timestamp::FromMicros(1234567890);
```

---

### ContextVector

**Purpose:** Sparse multi-dimensional context representation.

**Definition:**
```cpp
class ContextVector {
public:
    using DimensionType = std::string;
    using ValueType = float;

    ContextVector() = default;

    void Set(const DimensionType& dimension, ValueType value);
    ValueType Get(const DimensionType& dimension) const;  // 0.0 if absent
    bool Has(const DimensionType& dimension) const;
    void Remove(const DimensionType& dimension);
    void Clear();

    size_t Size() const;
    bool IsEmpty() const;
    std::vector<DimensionType> GetDimensions() const;

    // Similarity metrics
    float CosineSimilarity(const ContextVector& other) const;
    float EuclideanDistance(const ContextVector& other) const;
    float DotProduct(const ContextVector& other) const;
    float Norm() const;

    ContextVector Normalized() const;

    // Operators
    ContextVector operator+(const ContextVector& other) const;
    ContextVector operator*(float scalar) const;
    bool operator==(const ContextVector& other) const;

    std::string ToString() const;
};
```

**Usage:**
```cpp
// Create context
ContextVector context;
context.Set("topic", 0.8f);
context.Set("formality", 0.6f);
context.Set("sentiment", 0.3f);

// Query context
if (context.Has("topic")) {
    float topic_value = context.Get("topic");
}

// Compute similarity
ContextVector other;
other.Set("topic", 0.9f);
float similarity = context.CosineSimilarity(other);

// Vector operations
ContextVector combined = context + other;
ContextVector normalized = context.Normalized();
```

---

### PatternType

**Purpose:** Classification of pattern complexity.

**Definition:**
```cpp
enum class PatternType : uint8_t {
    ATOMIC = 0,      // Indivisible, basic pattern
    COMPOSITE = 1,   // Composed of multiple sub-patterns
    META = 2,        // Pattern of patterns (highest abstraction)
};

const char* ToString(PatternType type);
PatternType ParsePatternType(const std::string& str);
```

**Usage:**
```cpp
PatternType type = PatternType::ATOMIC;
std::cout << ToString(type) << std::endl;  // "ATOMIC"

PatternType parsed = ParsePatternType("COMPOSITE");
```

---

### AssociationType

**Purpose:** Type of relationship between patterns.

**Definition:**
```cpp
enum class AssociationType : uint8_t {
    CAUSAL = 0,         // A typically precedes B
    CATEGORICAL = 1,    // A and B belong to same category
    SPATIAL = 2,        // A and B appear in similar spatial config
    FUNCTIONAL = 3,     // A and B serve similar purposes
    COMPOSITIONAL = 4,  // A contains B or vice versa
};

const char* ToString(AssociationType type);
AssociationType ParseAssociationType(const std::string& str);
```

**Usage:**
```cpp
AssociationType type = AssociationType::CAUSAL;
std::cout << ToString(type) << std::endl;  // "CAUSAL"
```

---

### DataModality

**Purpose:** Type of data a pattern represents.

**Definition:**
```cpp
enum class DataModality : uint8_t {
    UNKNOWN = 0,
    NUMERIC = 1,      // Numerical vector data
    IMAGE = 2,        // Image/visual data
    AUDIO = 3,        // Audio/sound data
    TEXT = 4,         // Text/language data
    COMPOSITE = 5,    // Mix of multiple modalities
};

const char* ToString(DataModality modality);
```

---

## Core Classes

### PatternData

**Purpose:** Stores compressed pattern data with feature extraction.

**Definition:**
```cpp
class PatternData {
public:
    static constexpr size_t kMaxRawDataSize = 10 * 1024 * 1024;  // 10MB

    PatternData() = default;
    explicit PatternData(DataModality modality);

    // Factory methods
    static PatternData FromBytes(const std::vector<uint8_t>& data,
                                  DataModality modality);
    static PatternData FromFeatures(const FeatureVector& features,
                                     DataModality modality);

    // Accessors
    DataModality GetModality() const;
    FeatureVector GetFeatures() const;
    std::vector<uint8_t> GetRawData() const;  // May decompress

    // Size information
    size_t GetCompressedSize() const;
    size_t GetOriginalSize() const;
    float GetCompressionRatio() const;

    bool IsEmpty() const;
    std::string ToString() const;

    bool operator==(const PatternData& other) const;
    bool operator!=(const PatternData& other) const;
};
```

**Usage:**
```cpp
// Create from raw bytes
std::vector<uint8_t> raw_data = {/* ... */};
PatternData data = PatternData::FromBytes(raw_data, DataModality::TEXT);

// Get features
FeatureVector features = data.GetFeatures();

// Check compression
std::cout << "Original: " << data.GetOriginalSize() << " bytes" << std::endl;
std::cout << "Compressed: " << data.GetCompressedSize() << " bytes" << std::endl;
std::cout << "Ratio: " << data.GetCompressionRatio() << std::endl;
```

---

### FeatureVector

**Purpose:** Numerical vector representation for similarity computation.

**Definition:**
```cpp
class FeatureVector {
public:
    using ValueType = float;
    using StorageType = std::vector<ValueType>;

    FeatureVector() = default;
    explicit FeatureVector(size_t dimension);
    explicit FeatureVector(const StorageType& data);

    size_t Dimension() const;

    ValueType operator[](size_t index) const;
    ValueType& operator[](size_t index);

    const StorageType& Data() const;
    StorageType& Data();

    // Vector operations
    float Norm() const;
    FeatureVector Normalized() const;
    float DotProduct(const FeatureVector& other) const;
    float EuclideanDistance(const FeatureVector& other) const;
    float CosineSimilarity(const FeatureVector& other) const;

    FeatureVector operator+(const FeatureVector& other) const;
    FeatureVector operator-(const FeatureVector& other) const;
    FeatureVector operator*(float scalar) const;

    std::string ToString(size_t max_elements = 10) const;
};
```

**Usage:**
```cpp
// Create feature vector
FeatureVector v1(64);  // 64-dimensional
for (size_t i = 0; i < v1.Dimension(); ++i) {
    v1[i] = static_cast<float>(i) / 64.0f;
}

// Compute similarity
FeatureVector v2(64);
// ... fill v2 ...
float similarity = v1.CosineSimilarity(v2);
float distance = v1.EuclideanDistance(v2);

// Vector operations
FeatureVector sum = v1 + v2;
FeatureVector scaled = v1 * 2.0f;
FeatureVector normalized = v1.Normalized();
```

---

### PatternNode

**Purpose:** Complete pattern representation with metadata.

**Definition:**
```cpp
class PatternNode {
public:
    PatternNode() = default;

    // Accessors
    PatternID GetID() const;
    void SetID(PatternID id);

    PatternType GetType() const;
    void SetType(PatternType type);

    const PatternData& GetData() const;
    void SetData(const PatternData& data);

    float GetConfidence() const;
    void SetConfidence(float confidence);

    Timestamp GetCreationTime() const;
    Timestamp GetLastAccessTime() const;
    void UpdateAccessTime();

    uint64_t GetAccessCount() const;
    void IncrementAccessCount();

    // Sub-patterns (for composite patterns)
    const std::vector<PatternID>& GetSubPatterns() const;
    void AddSubPattern(PatternID id);
    void SetSubPatterns(const std::vector<PatternID>& ids);

    // Metadata
    void SetMetadata(const std::string& key, const std::string& value);
    std::string GetMetadata(const std::string& key) const;
    bool HasMetadata(const std::string& key) const;

    bool IsValid() const;
};
```

**Usage:**
```cpp
// Create pattern node
PatternNode node;
node.SetID(PatternID::Generate());
node.SetType(PatternType::ATOMIC);
node.SetConfidence(0.75f);

// Set data
auto data = PatternData::FromBytes(raw_bytes, DataModality::TEXT);
node.SetData(data);

// Track access
node.UpdateAccessTime();
node.IncrementAccessCount();

// Metadata
node.SetMetadata("source", "user_input");
node.SetMetadata("language", "en");
```

---

### AssociationEdge

**Purpose:** Represents a directed association between two patterns.

**Definition:**
```cpp
class AssociationEdge {
public:
    AssociationEdge() = default;
    AssociationEdge(PatternID source, PatternID target,
                     float strength, AssociationType type);

    PatternID GetSource() const;
    PatternID GetTarget() const;
    float GetStrength() const;
    AssociationType GetType() const;

    void SetStrength(float strength);
    void SetType(AssociationType type);

    Timestamp GetCreationTime() const;
    Timestamp GetLastActivation() const;
    void UpdateActivationTime();

    uint64_t GetActivationCount() const;
    void IncrementActivationCount();

    bool IsValid() const;
};
```

---

## Pattern Engine

### PatternEngine Configuration

**Definition:**
```cpp
struct PatternEngine::Config {
    // Database configuration
    std::string database_path;
    std::string database_type{"memory"};  // "memory" or "persistent"

    // Component configurations
    PatternExtractor::Config extraction_config;
    PatternMatcher::Config matching_config;

    // Similarity metric selection
    std::string similarity_metric{"context"};  // "context", "cosine", "euclidean"

    // Engine options
    bool enable_auto_refinement{true};
    bool enable_indexing{true};
};
```

**Usage:**
```cpp
PatternEngine::Config config;
config.database_type = "persistent";
config.database_path = "patterns.db";
config.similarity_metric = "context";
config.enable_auto_refinement = true;
config.enable_indexing = true;

// Set extraction configuration
config.extraction_config.feature_dimension = 64;
config.extraction_config.min_pattern_size = 1;
config.extraction_config.max_pattern_size = 1000;

// Set matching configuration
config.matching_config.similarity_threshold = 0.60f;
config.matching_config.strong_match_threshold = 0.75f;

PatternEngine engine(config);
```

---

### Pattern Discovery

**Methods:**

#### ProcessInput
Process raw input end-to-end (extract, match, create).

```cpp
struct ProcessResult {
    std::vector<PatternID> activated_patterns;
    std::vector<PatternID> created_patterns;
    std::vector<PatternID> updated_patterns;
    float processing_time_ms;
};

ProcessResult ProcessInput(const std::vector<uint8_t>& raw_input,
                            DataModality modality);
```

**Usage:**
```cpp
std::string text = "Hello, world!";
std::vector<uint8_t> bytes(text.begin(), text.end());

auto result = engine.ProcessInput(bytes, DataModality::TEXT);

std::cout << "Activated: " << result.activated_patterns.size() << std::endl;
std::cout << "Created: " << result.created_patterns.size() << std::endl;
std::cout << "Time: " << result.processing_time_ms << " ms" << std::endl;
```

#### DiscoverPatterns
Extract patterns from raw input.

```cpp
std::vector<PatternID> DiscoverPatterns(const std::vector<uint8_t>& raw_input,
                                         DataModality modality);
```

**Usage:**
```cpp
auto pattern_ids = engine.DiscoverPatterns(bytes, DataModality::TEXT);
for (auto id : pattern_ids) {
    auto pattern = engine.GetPattern(id);
    // Process pattern...
}
```

---

### Pattern Retrieval

#### GetPattern
Retrieve a single pattern by ID.

```cpp
std::optional<PatternNode> GetPattern(PatternID id) const;
```

**Usage:**
```cpp
auto maybe_pattern = engine.GetPattern(pattern_id);
if (maybe_pattern) {
    const PatternNode& pattern = *maybe_pattern;
    std::cout << "Confidence: " << pattern.GetConfidence() << std::endl;
}
```

#### GetPatternsBatch
Retrieve multiple patterns efficiently.

```cpp
std::vector<PatternNode> GetPatternsBatch(const std::vector<PatternID>& ids) const;
```

**Usage:**
```cpp
std::vector<PatternID> ids = {id1, id2, id3};
auto patterns = engine.GetPatternsBatch(ids);
// Returns only patterns that were found
```

#### GetAllPatternIDs
Get all pattern IDs in the database.

```cpp
std::vector<PatternID> GetAllPatternIDs() const;
```

---

### Pattern Search

#### FindSimilarPatterns
Find k most similar patterns to a query.

```cpp
struct SearchResult {
    PatternID pattern_id;
    float similarity_score;
};

std::vector<SearchResult> FindSimilarPatterns(const PatternData& query,
                                               size_t k = 10,
                                               float threshold = 0.0f) const;
```

**Usage:**
```cpp
PatternData query = PatternData::FromBytes(query_bytes, DataModality::TEXT);
auto results = engine.FindSimilarPatterns(query, 5, 0.5f);

for (const auto& result : results) {
    std::cout << "Pattern: " << result.pattern_id.ToString()
              << " Similarity: " << result.similarity_score << std::endl;
}
```

#### FindSimilarPatternsById
Find patterns similar to an existing pattern.

```cpp
std::vector<SearchResult> FindSimilarPatternsById(PatternID query_id,
                                                   size_t k = 10,
                                                   float threshold = 0.0f) const;
```

---

### Pattern Management

#### CreatePattern
Create a new atomic pattern.

```cpp
PatternID CreatePattern(const PatternData& data, float confidence = 0.5f);
```

**Usage:**
```cpp
auto data = PatternData::FromBytes(bytes, DataModality::TEXT);
PatternID id = engine.CreatePattern(data, 0.75f);
```

#### CreateCompositePattern
Create a composite pattern from sub-patterns.

```cpp
PatternID CreateCompositePattern(const std::vector<PatternID>& sub_patterns,
                                  const PatternData& data);
```

**Usage:**
```cpp
std::vector<PatternID> sub_patterns = {id1, id2, id3};
auto composite_data = PatternData::FromBytes(composite_bytes, DataModality::TEXT);
PatternID composite_id = engine.CreateCompositePattern(sub_patterns, composite_data);
```

#### UpdatePattern
Update an existing pattern's data.

```cpp
bool UpdatePattern(PatternID id, const PatternData& new_data);
```

#### DeletePattern
Delete a pattern.

```cpp
bool DeletePattern(PatternID id);
```

---

### PatternEngine Statistics

```cpp
struct Statistics {
    size_t total_patterns{0};
    size_t atomic_patterns{0};
    size_t composite_patterns{0};
    size_t meta_patterns{0};
    float avg_confidence{0.0f};
    float avg_pattern_size_bytes{0.0f};
    StorageStats storage_stats;
};

Statistics GetStatistics() const;
```

**Usage:**
```cpp
auto stats = engine.GetStatistics();
std::cout << "Total patterns: " << stats.total_patterns << std::endl;
std::cout << "Average confidence: " << stats.avg_confidence << std::endl;
```

---

### PatternEngine Maintenance

#### Compact
Compact the database to reclaim space.

```cpp
void Compact();
```

#### Flush
Flush pending writes to disk.

```cpp
void Flush();
```

#### RunMaintenance
Run all maintenance tasks (refinement, pruning, etc.).

```cpp
void RunMaintenance();
```

---

## Association Learning System

### AssociationLearningSystem Configuration

**Definition:**
```cpp
struct AssociationLearningSystem::Config {
    // Sub-component configurations
    CoOccurrenceTracker::Config co_occurrence;
    AssociationFormationRules::Config formation;
    ReinforcementManager::Config reinforcement;
    CompetitiveLearner::Config competition;
    StrengthNormalizer::Config normalization;

    // System parameters
    size_t association_capacity{1000000};
    Timestamp::Duration activation_window{std::chrono::seconds(10)};
    size_t max_activation_history{10000};

    // Auto-maintenance intervals
    Timestamp::Duration auto_decay_interval{std::chrono::hours(1)};
    Timestamp::Duration auto_competition_interval{std::chrono::minutes(30)};
    Timestamp::Duration auto_normalization_interval{std::chrono::minutes(30)};

    float prune_threshold{0.05f};
    bool enable_auto_maintenance{true};
};
```

**Common Configuration Example:**
```cpp
AssociationLearningSystem::Config config;

// Co-occurrence tracking
config.co_occurrence.window_size = std::chrono::seconds(300);  // 5 minutes
config.co_occurrence.min_co_occurrences = 2;

// Formation rules
config.formation.min_co_occurrence_count = 2;
config.formation.min_co_occurrence_strength = 0.3f;

// Enable automatic maintenance
config.enable_auto_maintenance = true;
config.auto_decay_interval = std::chrono::hours(1);
config.prune_threshold = 0.05f;

AssociationLearningSystem system(config);
```

---

### Recording Activations

#### RecordPatternActivation
Record a single pattern activation.

```cpp
void RecordPatternActivation(PatternID pattern,
                              const ContextVector& context = ContextVector());
```

**Usage:**
```cpp
// Simple activation
system.RecordPatternActivation(pattern_id);

// With context
ContextVector context;
context.Set("topic", 0.8f);
system.RecordPatternActivation(pattern_id, context);
```

#### RecordPatternActivations
Record multiple activations (batch operation).

```cpp
void RecordPatternActivations(const std::vector<PatternID>& patterns,
                               const ContextVector& context = ContextVector());
```

**Usage:**
```cpp
std::vector<PatternID> activated = {id1, id2, id3};
system.RecordPatternActivations(activated);
```

---

### Association Formation

#### FormNewAssociations
Analyze co-occurrences and create associations.

```cpp
size_t FormNewAssociations(const PatternDatabase& pattern_db);
```

**Usage:**
```cpp
// After recording many activations, form associations
size_t count = system.FormNewAssociations(pattern_database);
std::cout << "Formed " << count << " new associations" << std::endl;
```

#### FormAssociationsForPattern
Form associations for a specific pattern.

```cpp
size_t FormAssociationsForPattern(PatternID pattern,
                                   const PatternDatabase& pattern_db);
```

---

### Reinforcement Learning

#### Reinforce
Apply reinforcement based on prediction accuracy.

```cpp
void Reinforce(PatternID predicted, PatternID actual, bool correct);
```

**Usage:**
```cpp
// Pattern was correctly predicted
system.Reinforce(predicted_id, actual_id, true);

// Pattern was incorrectly predicted
system.Reinforce(predicted_id, actual_id, false);
```

#### ReinforceBatch
Batch reinforcement for efficiency.

```cpp
void ReinforceBatch(
    const std::vector<std::tuple<PatternID, PatternID, bool>>& outcomes
);
```

**Usage:**
```cpp
std::vector<std::tuple<PatternID, PatternID, bool>> outcomes;
outcomes.emplace_back(pred1, actual1, true);
outcomes.emplace_back(pred2, actual2, false);
system.ReinforceBatch(outcomes);
```

---

### Maintenance Operations

#### ApplyDecay
Apply time-based decay to associations.

```cpp
void ApplyDecay(Timestamp::Duration elapsed);
```

**Usage:**
```cpp
auto elapsed = std::chrono::hours(1);
system.ApplyDecay(elapsed);
```

#### ApplyCompetition
Apply competitive learning (winner-take-all).

```cpp
size_t ApplyCompetition();
```

**Usage:**
```cpp
size_t patterns_affected = system.ApplyCompetition();
```

#### ApplyNormalization
Normalize association strengths.

```cpp
size_t ApplyNormalization();
```

#### PruneWeakAssociations
Remove associations below threshold.

```cpp
size_t PruneWeakAssociations(float min_strength = 0.0f);
```

**Usage:**
```cpp
// Use configured threshold
size_t pruned = system.PruneWeakAssociations();

// Custom threshold
size_t pruned = system.PruneWeakAssociations(0.1f);
```

#### PerformMaintenance
Run all maintenance operations.

```cpp
struct MaintenanceStats {
    size_t competitions_applied;
    size_t normalizations_applied;
    size_t associations_pruned;
    Timestamp::Duration decay_applied;
};

MaintenanceStats PerformMaintenance();
```

**Usage:**
```cpp
auto stats = system.PerformMaintenance();
std::cout << "Pruned: " << stats.associations_pruned << " associations" << std::endl;
```

---

### Prediction

#### Predict
Predict next patterns based on current pattern.

```cpp
std::vector<PatternID> Predict(PatternID pattern,
                                size_t k = 5,
                                const ContextVector* context = nullptr) const;
```

**Usage:**
```cpp
// Simple prediction
auto predictions = system.Predict(current_pattern, 5);

// Context-aware prediction
ContextVector context;
context.Set("topic", 0.8f);
auto predictions = system.Predict(current_pattern, 5, &context);
```

#### PredictWithConfidence
Predict with confidence scores.

```cpp
std::vector<std::pair<PatternID, float>> PredictWithConfidence(
    PatternID pattern,
    size_t k = 5,
    const ContextVector* context = nullptr
) const;
```

**Usage:**
```cpp
auto predictions = system.PredictWithConfidence(current_pattern, 5);
for (const auto& [pattern_id, confidence] : predictions) {
    std::cout << "Pattern: " << pattern_id.ToString()
              << " Confidence: " << confidence << std::endl;
}
```

#### PredictWithAttention
Predict using attention-weighted scoring.

```cpp
std::vector<std::pair<PatternID, float>> PredictWithAttention(
    PatternID source,
    size_t k = 5,
    const ContextVector& context = ContextVector()
) const;
```

**Usage:**
```cpp
// Requires SetAttentionMechanism() to be called first
system.SetAttentionMechanism(&attention_mechanism);

ContextVector context;
auto predictions = system.PredictWithAttention(current_pattern, 5, context);
```

#### PropagateActivation
Propagate activation through the network.

```cpp
std::vector<AssociationMatrix::ActivationResult> PropagateActivation(
    PatternID source,
    float initial_activation = 1.0f,
    size_t max_hops = 3,
    float min_activation = 0.01f,
    const ContextVector* context = nullptr
) const;
```

**Usage:**
```cpp
auto results = system.PropagateActivation(source_pattern, 1.0f, 3, 0.01f);
for (const auto& result : results) {
    std::cout << "Reached: " << result.pattern_id.ToString()
              << " Activation: " << result.activation_level << std::endl;
}
```

---

### AssociationLearningSystem Statistics

```cpp
struct Statistics {
    size_t total_associations;
    size_t active_associations;
    float average_strength;
    float min_strength;
    float max_strength;
    size_t patterns_with_associations;
    float average_associations_per_pattern;
    size_t total_co_occurrences;
    size_t activation_history_size;
    Timestamp last_decay;
    Timestamp last_competition;
    Timestamp last_normalization;
    Timestamp last_pruning;
    size_t formations_count;
    size_t reinforcements_count;
    size_t predictions_count;
};

Statistics GetStatistics() const;
size_t GetAssociationCount() const;
float GetAverageStrength() const;
void PrintStatistics(std::ostream& out) const;
```

**Usage:**
```cpp
auto stats = system.GetStatistics();
std::cout << "Total associations: " << stats.total_associations << std::endl;
std::cout << "Average strength: " << stats.average_strength << std::endl;

// Print comprehensive statistics
system.PrintStatistics(std::cout);
```

---

## Attention Mechanism

### Attention Configuration

```cpp
struct AttentionConfig {
    size_t num_heads = 4;
    float temperature = 1.0f;
    bool use_context = true;
    bool use_importance = true;
    std::string attention_type = "dot_product";  // "dot_product", "additive", "multiplicative"

    float association_weight = 0.6f;
    float attention_weight = 0.4f;

    bool enable_caching = true;
    size_t cache_size = 1000;
    bool debug_logging = false;

    bool Validate() const;
};
```

**Usage:**
```cpp
AttentionConfig config;
config.num_heads = 4;
config.temperature = 1.0f;
config.use_context = true;
config.use_importance = true;
config.association_weight = 0.6f;
config.attention_weight = 0.4f;
config.enable_caching = true;

auto attention = std::make_unique<BasicAttention>(config);
```

---

### Computing Attention

#### ComputeAttention
Compute normalized attention weights.

```cpp
std::map<PatternID, float> ComputeAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context
);
```

**Usage:**
```cpp
std::vector<PatternID> candidates = {id1, id2, id3};
ContextVector context;
context.Set("topic", 0.8f);

auto weights = attention->ComputeAttention(query_pattern, candidates, context);
for (const auto& [pattern_id, weight] : weights) {
    std::cout << "Pattern: " << pattern_id.ToString()
              << " Weight: " << weight << std::endl;
}
```

#### ComputeDetailedAttention
Get detailed attention scores with component breakdown.

```cpp
struct AttentionScore {
    PatternID pattern_id;
    float weight;
    float raw_score;

    struct Components {
        float semantic_similarity = 0.0f;
        float context_similarity = 0.0f;
        float importance_score = 0.0f;
        float temporal_score = 0.0f;
        float structural_score = 0.0f;
    } components;
};

std::vector<AttentionScore> ComputeDetailedAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context
);
```

**Usage:**
```cpp
auto scores = attention->ComputeDetailedAttention(query, candidates, context);
for (const auto& score : scores) {
    std::cout << "Pattern: " << score.pattern_id.ToString() << std::endl;
    std::cout << "  Weight: " << score.weight << std::endl;
    std::cout << "  Semantic: " << score.components.semantic_similarity << std::endl;
    std::cout << "  Context: " << score.components.context_similarity << std::endl;
    std::cout << "  Importance: " << score.components.importance_score << std::endl;
}
```

---

### Applying Attention

#### ApplyAttention
Combine attention with association strengths.

```cpp
std::vector<std::pair<PatternID, float>> ApplyAttention(
    PatternID query,
    const std::vector<PatternID>& predictions,
    const ContextVector& context
);
```

**Usage:**
```cpp
// Get initial predictions from associations
auto predictions = system.Predict(current_pattern, 10);

// Apply attention weighting
auto weighted = attention->ApplyAttention(current_pattern, predictions, context);

// Results are re-ranked by combined score
for (const auto& [pattern_id, score] : weighted) {
    std::cout << "Pattern: " << pattern_id.ToString()
              << " Score: " << score << std::endl;
}
```

---

## Storage Backends

### PatternDatabase Interface

Abstract interface for pattern storage.

```cpp
class PatternDatabase {
public:
    virtual ~PatternDatabase() = default;

    // Single operations
    virtual bool Store(const PatternNode& node) = 0;
    virtual std::optional<PatternNode> Retrieve(PatternID id) = 0;
    virtual bool Update(const PatternNode& node) = 0;
    virtual bool Delete(PatternID id) = 0;
    virtual bool Exists(PatternID id) const = 0;

    // Batch operations
    virtual size_t StoreBatch(const std::vector<PatternNode>& nodes) = 0;
    virtual std::vector<PatternNode> RetrieveBatch(const std::vector<PatternID>& ids) = 0;
    virtual size_t DeleteBatch(const std::vector<PatternID>& ids) = 0;

    // Queries
    virtual std::vector<PatternID> FindByType(PatternType type,
                                               const QueryOptions& options) = 0;
    virtual std::vector<PatternID> FindByTimeRange(Timestamp start,
                                                    Timestamp end,
                                                    const QueryOptions& options) = 0;
    virtual std::vector<PatternID> FindAll(const QueryOptions& options) = 0;

    // Statistics
    virtual size_t Count() const = 0;
    virtual StorageStats GetStats() const = 0;

    // Maintenance
    virtual void Flush() = 0;
    virtual void Compact() = 0;
    virtual void Clear() = 0;

    // Snapshots
    virtual bool CreateSnapshot(const std::string& path) = 0;
    virtual bool RestoreSnapshot(const std::string& path) = 0;
};
```

---

### MemoryBackend

In-memory storage backend (fast, volatile).

**Usage:**
```cpp
#include "storage/memory_backend.hpp"

MemoryBackend::Config config;
config.initial_capacity = 10000;
auto db = std::make_shared<MemoryBackend>(config);

// Store pattern
PatternNode node;
// ... configure node ...
db->Store(node);

// Retrieve
auto retrieved = db->Retrieve(pattern_id);
```

---

### PersistentBackend

SQLite-based persistent storage.

**Configuration:**
```cpp
struct PersistentBackend::Config {
    std::string db_path;
    bool enable_wal{true};
    size_t cache_size_kb{10240};       // 10MB
    size_t page_size{4096};            // 4KB
    bool enable_auto_vacuum{true};
    std::string synchronous{"NORMAL"}; // FULL, NORMAL, or OFF
};
```

**Usage:**
```cpp
#include "storage/persistent_backend.hpp"

PersistentBackend::Config config;
config.db_path = "patterns.db";
config.enable_wal = true;
config.cache_size_kb = 10240;  // 10MB cache

auto db = std::make_shared<PersistentBackend>(config);

// All operations are persistent
db->Store(node);
db->Flush();  // Ensure written to disk
db->Compact();  // Reclaim space
```

---

## Complete Examples

### Example 1: Basic Pattern Learning

```cpp
#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"

// Configure pattern engine
PatternEngine::Config engine_config;
engine_config.database_type = "memory";
engine_config.similarity_metric = "context";
PatternEngine engine(engine_config);

// Configure association system
AssociationLearningSystem::Config assoc_config;
assoc_config.enable_auto_maintenance = true;
AssociationLearningSystem assoc_system(assoc_config);

// Process inputs
std::vector<std::string> inputs = {
    "Hello",
    "How are you?",
    "I am fine",
    "Thank you"
};

PatternID prev_id;
for (const auto& text : inputs) {
    std::vector<uint8_t> bytes(text.begin(), text.end());

    // Process input
    auto result = engine.ProcessInput(bytes, DataModality::TEXT);

    if (!result.activated_patterns.empty()) {
        PatternID current_id = result.activated_patterns[0];

        // Record activation
        assoc_system.RecordPatternActivation(current_id);

        // If we had a previous pattern, we can form associations
        if (prev_id.IsValid()) {
            // Association will be formed between prev_id and current_id
        }

        prev_id = current_id;
    }
}

// Form associations after collecting activations
auto db = engine.GetDatabase();
size_t formed = assoc_system.FormNewAssociations(*db);
std::cout << "Formed " << formed << " associations" << std::endl;
```

---

### Example 2: Prediction with Attention

```cpp
#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"
#include "learning/basic_attention.hpp"

// Set up components
PatternEngine engine(/* config */);
AssociationLearningSystem assoc_system(/* config */);

// Configure attention
AttentionConfig attention_config;
attention_config.num_heads = 4;
attention_config.association_weight = 0.6f;
attention_config.attention_weight = 0.4f;

auto attention = std::make_unique<BasicAttention>(attention_config);
attention->SetPatternDatabase(engine.GetDatabase().get());

// Integrate attention with association system
assoc_system.SetAttentionMechanism(attention.get());

// ... learn patterns and associations ...

// Make prediction with attention
std::string query_text = "Hello";
std::vector<uint8_t> query_bytes(query_text.begin(), query_text.end());
auto result = engine.ProcessInput(query_bytes, DataModality::TEXT);

if (!result.activated_patterns.empty()) {
    PatternID query_id = result.activated_patterns[0];

    ContextVector context;
    context.Set("formality", 0.7f);

    // Predict with attention-weighted scoring
    auto predictions = assoc_system.PredictWithAttention(query_id, 5, context);

    std::cout << "Predictions:" << std::endl;
    for (const auto& [pattern_id, score] : predictions) {
        auto pattern = engine.GetPattern(pattern_id);
        if (pattern) {
            std::cout << "  Score: " << score << std::endl;
        }
    }
}
```

---

### Example 3: Persistent Session Management

```cpp
#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"
#include "storage/persistent_backend.hpp"

// Create persistent storage
PersistentBackend::Config db_config;
db_config.db_path = "session.db";
db_config.enable_wal = true;
auto db = std::make_shared<PersistentBackend>(db_config);

// Pattern engine with persistent storage
PatternEngine::Config engine_config;
engine_config.database_type = "persistent";
engine_config.database_path = "session.db";
PatternEngine engine(engine_config);

// Association system
AssociationLearningSystem assoc_system;

// Train session
// ... process inputs, record activations, form associations ...

// Save association system state
assoc_system.Save("session_associations.dat");

// Flush pattern database
db->Flush();
db->Compact();

std::cout << "Session saved!" << std::endl;

// ---- Later, in a new session ----

// Load pattern database (automatically loaded)
PatternEngine engine2(engine_config);

// Load association system
AssociationLearningSystem assoc_system2;
assoc_system2.Load("session_associations.dat");

// Continue from where we left off
auto stats = assoc_system2.GetStatistics();
std::cout << "Loaded " << stats.total_associations << " associations" << std::endl;
```

---

### Example 4: Batch Processing and Maintenance

```cpp
#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"

PatternEngine engine(/* config */);
AssociationLearningSystem assoc_system(/* config */);

// Batch process many inputs
std::vector<std::string> corpus = LoadTextCorpus("data.txt");

std::vector<PatternID> all_patterns;
for (size_t i = 0; i < corpus.size(); ++i) {
    std::vector<uint8_t> bytes(corpus[i].begin(), corpus[i].end());
    auto result = engine.ProcessInput(bytes, DataModality::TEXT);

    if (!result.activated_patterns.empty()) {
        all_patterns.push_back(result.activated_patterns[0]);
    }

    // Record activation batch every 100 inputs
    if (i % 100 == 0 && !all_patterns.empty()) {
        assoc_system.RecordPatternActivations(all_patterns);
        all_patterns.clear();

        // Form associations periodically
        if (i % 1000 == 0) {
            auto db = engine.GetDatabase();
            size_t formed = assoc_system.FormNewAssociations(*db);
            std::cout << "Formed " << formed << " associations at input " << i << std::endl;
        }
    }
}

// Final association formation
auto db = engine.GetDatabase();
assoc_system.FormNewAssociations(*db);

// Perform maintenance
auto maint_stats = assoc_system.PerformMaintenance();
std::cout << "Maintenance complete:" << std::endl;
std::cout << "  Pruned: " << maint_stats.associations_pruned << std::endl;
std::cout << "  Competitions: " << maint_stats.competitions_applied << std::endl;

// Get final statistics
auto stats = assoc_system.GetStatistics();
std::cout << "Final statistics:" << std::endl;
std::cout << "  Patterns: " << engine.GetStatistics().total_patterns << std::endl;
std::cout << "  Associations: " << stats.total_associations << std::endl;
std::cout << "  Average strength: " << stats.average_strength << std::endl;
```

---

### Example 5: Real-time Learning with Reinforcement

```cpp
#include "core/pattern_engine.hpp"
#include "association/association_learning_system.hpp"

PatternEngine engine(/* config */);
AssociationLearningSystem assoc_system(/* config */);

PatternID prev_pattern;
while (true) {
    // Get user input
    std::string input = GetUserInput();
    if (input == "quit") break;

    std::vector<uint8_t> bytes(input.begin(), input.end());
    auto result = engine.ProcessInput(bytes, DataModality::TEXT);

    if (!result.activated_patterns.empty()) {
        PatternID current_pattern = result.activated_patterns[0];

        // Record activation
        assoc_system.RecordPatternActivation(current_pattern);

        // If we made a prediction last time, reinforce based on correctness
        if (prev_pattern.IsValid()) {
            // Get what we predicted
            auto predictions = assoc_system.Predict(prev_pattern, 1);

            if (!predictions.empty()) {
                bool correct = (predictions[0] == current_pattern);
                assoc_system.Reinforce(predictions[0], current_pattern, correct);

                if (correct) {
                    std::cout << "[Correct prediction!]" << std::endl;
                }
            }
        }

        // Make prediction for next input
        auto next_predictions = assoc_system.PredictWithConfidence(current_pattern, 3);
        if (!next_predictions.empty()) {
            std::cout << "Expecting next: ";
            for (const auto& [pid, conf] : next_predictions) {
                std::cout << " [" << conf << "]";
            }
            std::cout << std::endl;
        }

        prev_pattern = current_pattern;
    }

    // Periodic maintenance (every 100 inputs)
    static int input_count = 0;
    if (++input_count % 100 == 0) {
        assoc_system.PerformMaintenance();
    }
}
```

---

## Best Practices

### Performance Optimization

1. **Use Batch Operations**: Always prefer batch operations for better performance.
   ```cpp
   // Good
   db->StoreBatch(patterns);

   // Avoid
   for (const auto& pattern : patterns) {
       db->Store(pattern);
   }
   ```

2. **Enable Caching**: Use caching for similarity search and attention.
   ```cpp
   config.enable_caching = true;
   config.cache_size = 10000;
   ```

3. **Periodic Maintenance**: Run maintenance periodically, not after every operation.
   ```cpp
   if (operation_count % 1000 == 0) {
       assoc_system.PerformMaintenance();
   }
   ```

### Memory Management

1. **Set Capacity Limits**: Configure reasonable capacity limits.
   ```cpp
   config.association_capacity = 1000000;
   config.max_activation_history = 10000;
   ```

2. **Prune Regularly**: Remove weak associations.
   ```cpp
   assoc_system.PruneWeakAssociations(0.05f);
   ```

3. **Compact Storage**: Reclaim space periodically.
   ```cpp
   db->Compact();
   ```

### Thread Safety

All public APIs are thread-safe. For concurrent access:

```cpp
// Multiple threads can safely call:
std::thread t1([&]() { engine.ProcessInput(data1, modality); });
std::thread t2([&]() { engine.ProcessInput(data2, modality); });
t1.join();
t2.join();
```

### Error Handling

Most methods return `std::optional` or boolean for error indication:

```cpp
auto pattern = engine.GetPattern(id);
if (!pattern) {
    std::cerr << "Pattern not found: " << id.ToString() << std::endl;
    return;
}

if (!db->Store(node)) {
    std::cerr << "Failed to store pattern" << std::endl;
}
```

---

## Version History

**1.0** (2025-11-18)
- Initial API reference
- Complete coverage of PatternEngine, AssociationLearningSystem, and AttentionMechanism
- Storage backend documentation
- Complete examples

---

**End of API Reference**
