// File: src/association/association_edge.cpp
#include "association/association_edge.hpp"
#include "core/binary_io.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace dpan {

// ============================================================================
// Constructor
// ============================================================================

AssociationEdge::AssociationEdge(
    PatternID source,
    PatternID target,
    AssociationType type,
    float initial_strength
) : source_(source),
    target_(target),
    type_(type),
    strength_(std::clamp(initial_strength, 0.0f, 1.0f)),
    creation_time_(Timestamp::Now())
{
    last_reinforcement_.store(creation_time_.ToMicros(), std::memory_order_relaxed);
}

// Move constructor
AssociationEdge::AssociationEdge(AssociationEdge&& other) noexcept
    : source_(other.source_),
      target_(other.target_),
      type_(other.type_),
      strength_(other.strength_.load(std::memory_order_relaxed)),
      co_occurrence_count_(other.co_occurrence_count_.load(std::memory_order_relaxed)),
      temporal_correlation_(other.temporal_correlation_.load(std::memory_order_relaxed)),
      decay_rate_(other.decay_rate_.load(std::memory_order_relaxed)),
      last_reinforcement_(other.last_reinforcement_.load(std::memory_order_relaxed)),
      context_profile_(std::move(other.context_profile_)),
      creation_time_(other.creation_time_)
{
    // Note: mutex is not moveable, so we get a new one (default constructed)
}

// Move assignment operator
AssociationEdge& AssociationEdge::operator=(AssociationEdge&& other) noexcept {
    if (this != &other) {
        source_ = other.source_;
        target_ = other.target_;
        type_ = other.type_;
        strength_.store(other.strength_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        co_occurrence_count_.store(other.co_occurrence_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        temporal_correlation_.store(other.temporal_correlation_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        decay_rate_.store(other.decay_rate_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        last_reinforcement_.store(other.last_reinforcement_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        context_profile_ = std::move(other.context_profile_);
        creation_time_ = other.creation_time_;
        // Note: mutex is not moveable
    }
    return *this;
}

// ============================================================================
// Strength Management
// ============================================================================

void AssociationEdge::SetStrength(float strength) {
    // Clamp to valid range [0, 1]
    strength = std::clamp(strength, 0.0f, 1.0f);
    strength_.store(strength, std::memory_order_relaxed);
}

void AssociationEdge::AdjustStrength(float delta) {
    float current = strength_.load(std::memory_order_relaxed);
    float new_strength = std::clamp(current + delta, 0.0f, 1.0f);
    strength_.store(new_strength, std::memory_order_relaxed);
}

// ============================================================================
// Co-occurrence Tracking
// ============================================================================

void AssociationEdge::IncrementCoOccurrence(uint32_t count) {
    co_occurrence_count_.fetch_add(count, std::memory_order_relaxed);
}

// ============================================================================
// Temporal Correlation
// ============================================================================

void AssociationEdge::SetTemporalCorrelation(float correlation) {
    correlation = std::clamp(correlation, -1.0f, 1.0f);
    temporal_correlation_.store(correlation, std::memory_order_relaxed);
}

void AssociationEdge::UpdateTemporalCorrelation(float new_observation, float learning_rate) {
    float current = temporal_correlation_.load(std::memory_order_relaxed);
    // Exponential moving average
    float updated = current + learning_rate * (new_observation - current);
    updated = std::clamp(updated, -1.0f, 1.0f);
    temporal_correlation_.store(updated, std::memory_order_relaxed);
}

// ============================================================================
// Decay Management
// ============================================================================

void AssociationEdge::SetDecayRate(float rate) {
    decay_rate_.store(std::max(0.0f, rate), std::memory_order_relaxed);
}

Timestamp AssociationEdge::GetLastReinforcement() const {
    uint64_t micros = last_reinforcement_.load(std::memory_order_relaxed);
    return Timestamp::FromMicros(micros);
}

void AssociationEdge::RecordReinforcement() {
    Timestamp now = Timestamp::Now();
    last_reinforcement_.store(now.ToMicros(), std::memory_order_relaxed);
}

void AssociationEdge::ApplyDecay(Timestamp::Duration elapsed_time) {
    // Exponential decay: s(t) = s(0) * exp(-d * t)
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count();
    float decay_factor = std::exp(-decay_rate_.load(std::memory_order_relaxed) * seconds);

    float current = strength_.load(std::memory_order_relaxed);
    float decayed = current * decay_factor;
    strength_.store(std::max(0.0f, decayed), std::memory_order_relaxed);
}

// ============================================================================
// Context Profile
// ============================================================================

ContextVector AssociationEdge::GetContextProfile() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return context_profile_;
}

void AssociationEdge::SetContextProfile(const ContextVector& context) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    context_profile_ = context;
}

void AssociationEdge::UpdateContextProfile(const ContextVector& observed_context, float learning_rate) {
    std::lock_guard<std::mutex> lock(context_mutex_);

    // Update each dimension using exponential moving average
    for (const auto& dim : observed_context.GetDimensions()) {
        float current = context_profile_.Get(dim);
        float observed = observed_context.Get(dim);
        float updated = current + learning_rate * (observed - current);
        context_profile_.Set(dim, updated);
    }
}

float AssociationEdge::GetContextualStrength(const ContextVector& current_context) const {
    std::lock_guard<std::mutex> lock(context_mutex_);

    if (context_profile_.IsEmpty()) {
        // No context profile yet, use base strength
        return strength_.load(std::memory_order_relaxed);
    }

    // Compute context similarity (cosine similarity)
    float context_match = context_profile_.CosineSimilarity(current_context);

    // Modulate strength by context match
    // If contexts match well (similarity near 1), use full strength
    // If contexts don't match (similarity near 0), reduce strength
    float base_strength = strength_.load(std::memory_order_relaxed);
    float context_factor = 0.5f + 0.5f * context_match;  // Maps [-1,1] to [0,1]

    return base_strength * context_factor;
}

// ============================================================================
// Age and Statistics
// ============================================================================

Timestamp::Duration AssociationEdge::GetAge() const {
    return Timestamp::Now() - creation_time_;
}

bool AssociationEdge::IsActive(Timestamp::Duration max_idle_time) const {
    Timestamp last_reinforcement = GetLastReinforcement();
    Timestamp::Duration idle_time = Timestamp::Now() - last_reinforcement;
    return idle_time <= max_idle_time;
}

// ============================================================================
// Serialization
// ============================================================================

void AssociationEdge::Serialize(std::ostream& out) const {
    // Serialize core data
    source_.Serialize(out);
    target_.Serialize(out);

    io::Write<uint8_t>(out, static_cast<uint8_t>(type_));
    io::Write<float>(out, strength_.load(std::memory_order_relaxed));
    io::Write<uint32_t>(out, co_occurrence_count_.load(std::memory_order_relaxed));
    io::Write<float>(out, temporal_correlation_.load(std::memory_order_relaxed));
    io::Write<float>(out, decay_rate_.load(std::memory_order_relaxed));
    io::Write<uint64_t>(out, last_reinforcement_.load(std::memory_order_relaxed));

    creation_time_.Serialize(out);

    // Serialize context (protected by lock during serialization)
    std::lock_guard<std::mutex> lock(context_mutex_);
    context_profile_.Serialize(out);
}

std::unique_ptr<AssociationEdge> AssociationEdge::Deserialize(std::istream& in) {
    auto edge = std::make_unique<AssociationEdge>();

    edge->source_ = PatternID::Deserialize(in);
    edge->target_ = PatternID::Deserialize(in);

    edge->type_ = static_cast<AssociationType>(io::Read<uint8_t>(in));
    edge->strength_.store(io::Read<float>(in), std::memory_order_relaxed);
    edge->co_occurrence_count_.store(io::Read<uint32_t>(in), std::memory_order_relaxed);
    edge->temporal_correlation_.store(io::Read<float>(in), std::memory_order_relaxed);
    edge->decay_rate_.store(io::Read<float>(in), std::memory_order_relaxed);
    edge->last_reinforcement_.store(static_cast<uint64_t>(
        Timestamp::FromPersistedMicros(static_cast<int64_t>(io::Read<uint64_t>(in))).ToMicros()),
        std::memory_order_relaxed);

    edge->creation_time_ = Timestamp::Deserialize(in);

    edge->context_profile_ = ContextVector::Deserialize(in);

    return edge;
}

// ============================================================================
// Utility
// ============================================================================

std::string AssociationEdge::ToString() const {
    std::ostringstream oss;
    oss << "AssociationEdge{";
    oss << "src=" << source_.ToString();
    oss << ", tgt=" << target_.ToString();
    oss << ", type=" << dpan::ToString(type_);
    oss << ", strength=" << strength_.load(std::memory_order_relaxed);
    oss << ", co_occ=" << co_occurrence_count_.load(std::memory_order_relaxed);
    oss << ", temp_corr=" << temporal_correlation_.load(std::memory_order_relaxed);
    oss << ", age=" << GetAge().count() / 1000000 << "s";
    oss << "}";
    return oss.str();
}

size_t AssociationEdge::EstimateMemoryUsage() const {
    size_t base_size = sizeof(*this);

    std::lock_guard<std::mutex> lock(context_mutex_);
    // Rough estimate: string key + float value + overhead
    size_t context_size = context_profile_.Size() * (sizeof(std::string) + sizeof(float) + 32);

    return base_size + context_size;
}

std::unique_ptr<AssociationEdge> AssociationEdge::Clone() const {
    // Create new edge with same source, target, type
    auto cloned = std::make_unique<AssociationEdge>(source_, target_, type_);

    // Copy atomic values
    cloned->strength_.store(strength_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    cloned->co_occurrence_count_.store(co_occurrence_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    cloned->temporal_correlation_.store(temporal_correlation_.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // Copy non-atomic values
    cloned->decay_rate_.store(decay_rate_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    cloned->last_reinforcement_.store(last_reinforcement_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    cloned->creation_time_ = creation_time_;

    // Copy context profile (protected by lock)
    std::lock_guard<std::mutex> lock(context_mutex_);
    cloned->context_profile_ = context_profile_;

    return cloned;
}

void AssociationEdge::CopyStateFrom(const AssociationEdge& other) {
    if (this == &other) {
        return;
    }

    strength_.store(other.strength_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    co_occurrence_count_.store(other.co_occurrence_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    temporal_correlation_.store(other.temporal_correlation_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    decay_rate_.store(other.decay_rate_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    last_reinforcement_.store(other.last_reinforcement_.load(std::memory_order_relaxed), std::memory_order_relaxed);

    ContextVector context = other.GetContextProfile();
    std::lock_guard<std::mutex> lock(context_mutex_);
    context_profile_ = std::move(context);
}

// ============================================================================
// Comparison Operators
// ============================================================================

bool AssociationEdge::operator==(const AssociationEdge& other) const {
    return source_ == other.source_ &&
           target_ == other.target_ &&
           type_ == other.type_;
}

bool AssociationEdge::operator<(const AssociationEdge& other) const {
    // Sort by strength (descending)
    return strength_.load(std::memory_order_relaxed) >
           other.strength_.load(std::memory_order_relaxed);
}

} // namespace dpan
