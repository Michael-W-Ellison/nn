// File: src/agent/pattern_memory.cpp
#include "agent/pattern_memory.hpp"

#include "core/binary_io.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace dpan {
namespace agent {

namespace {

AssociationLearningSystem::Config TransitionConfig() {
    AssociationLearningSystem::Config config;
    // Strengths are managed by RecordTransition; no time-based maintenance
    config.enable_auto_maintenance = false;
    return config;
}

constexpr float kMinTransitionStrength = 0.01f;
constexpr float kNeutralActionStrength = 0.5f;

/// Estimated bytes per association: the edge plus its lookup and index entries
constexpr size_t kEdgeBytes = sizeof(AssociationEdge) + 160;

/// Recency is measured in steps, passed to the calculator as hours
/// (one step = one hour), with the decay set for the requested half-life
UtilityCalculator::Config UtilityConfig(size_t half_life_steps) {
    UtilityCalculator::Config config;
    config.recency_decay = std::log(2.0f) / static_cast<float>(std::max<size_t>(1, half_life_steps));
    return config;
}

}  // namespace

PatternMemory::PatternMemory(size_t input_size, size_t class_outputs, std::vector<size_t> value_sizes,
                             size_t observed_size, const Config& config)
    : config_(config),
      input_size_(input_size),
      class_outputs_(class_outputs),
      value_sizes_(std::move(value_sizes)),
      observed_size_(observed_size),
      projection_(input_size * config.dimension),
      class_serial_(class_outputs, 0),
      value_serial_(value_sizes_.size(), 0),
      associations_(TransitionConfig()),
      utility_(UtilityConfig(config.recency_half_life)) {
    if (config.dimension == 0 || config.capacity == 0) {
        throw std::invalid_argument("PatternMemory: dimension and capacity must be > 0");
    }
    // Gaussian random projection (approximately preserves distances)
    std::mt19937 rng(config.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f / std::sqrt(static_cast<float>(config.dimension)));
    for (float& w : projection_) {
        w = normal(rng);
    }
}

std::vector<float> PatternMemory::Project(const std::vector<float>& fused) const {
    if (fused.size() != input_size_) {
        throw std::invalid_argument("PatternMemory::Project: wrong input size");
    }
    std::vector<float> point(config_.dimension, 0.0f);
    for (size_t i = 0; i < input_size_; ++i) {
        float x = fused[i];
        if (x == 0.0f) continue;
        const float* row = &projection_[i * config_.dimension];
        for (size_t j = 0; j < config_.dimension; ++j) {
            point[j] += x * row[j];
        }
    }
    return point;
}

float PatternMemory::Distance(const std::vector<float>& a, const std::vector<float>& b) {
    return std::sqrt(SquaredDistance(a.data(), b.data(), a.size(), std::numeric_limits<float>::infinity()));
}

float PatternMemory::SquaredDistance(const float* a, const float* b, size_t n, float bound) {
    // Blocks of 16 vectorize; stop once the sum exceeds `bound`
    float sum = 0.0f;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float block = 0.0f;
        for (size_t j = i; j < i + 16; ++j) {
            float d = a[j] - b[j];
            block += d * d;
        }
        sum += block;
        if (sum > bound) return sum;
    }
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

PatternMemory::Prototype* PatternMemory::Find(PatternID id) {
    auto it = slots_.find(id);
    return it == slots_.end() ? nullptr : &prototypes_[it->second];
}

namespace {

/// Insert (slot, distance) into a list of the k nearest, nearest first
void Keep(std::vector<std::pair<size_t, float>>& best, size_t k, size_t slot, float distance) {
    if (best.size() == k && distance >= best.back().second) return;
    auto pos = std::upper_bound(best.begin(), best.end(), distance,
                                [](float value, const auto& entry) { return value < entry.second; });
    best.insert(pos, {slot, distance});
    if (best.size() > k) best.pop_back();
}

}  // namespace

template <typename UsableA, typename UsableB>
std::pair<std::vector<std::pair<size_t, float>>, std::vector<std::pair<size_t, float>>>
PatternMemory::NearestTwo(const std::vector<float>& point, size_t k, UsableA usable_a, UsableB usable_b) const {
    std::vector<std::pair<size_t, float>> a, b;
    a.reserve(k + 1);
    b.reserve(k + 1);
    const float infinity = std::numeric_limits<float>::infinity();
    for (size_t slot = 0; slot < prototypes_.size(); ++slot) {
        const Prototype& p = prototypes_[slot];
        bool use_a = usable_a(p), use_b = usable_b(p);
        if (!use_a && !use_b) continue;
        // Only distances that can enter a list matter
        float bound_a = use_a && a.size() == k ? a.back().second : (use_a ? infinity : 0.0f);
        float bound_b = use_b && b.size() == k ? b.back().second : (use_b ? infinity : 0.0f);
        float bound = std::max(bound_a, bound_b);
        float squared = SquaredDistance(point.data(), p.center.data(), point.size(),
                                        std::isinf(bound) ? infinity : bound * bound);
        float distance = std::sqrt(squared);
        if (distance >= bound) continue;
        if (use_a) Keep(a, k, slot, distance);
        if (use_b) Keep(b, k, slot, distance);
    }
    return {std::move(a), std::move(b)};
}

template <typename Usable>
std::vector<std::pair<size_t, float>> PatternMemory::Nearest(const std::vector<float>& point, size_t k,
                                                             Usable usable) const {
    return NearestTwo(point, k, usable, [](const Prototype&) { return false; }).first;
}

float PatternMemory::NearestDistance(const std::vector<float>& point) const {
    auto nearest = Nearest(point, 1, [](const Prototype&) { return true; });
    return nearest.empty() ? std::numeric_limits<float>::infinity() : nearest[0].second;
}

PatternMemory::Placement PatternMemory::Insert(const std::vector<float>& point, uint64_t step,
                                               const std::vector<float>& observed) {
    // Every insertion keeps room for itself and the associations that follow:
    // first free retired associations, then prune patterns
    if (OverBudget() && writes_since_compact_ > 0) {
        associations_.Compact();
        writes_since_compact_ = 0;
    }
    if (OverBudget()) Prune(step);
    auto nearest = Nearest(point, 1, [](const Prototype&) { return true; });
    float distance = nearest.empty() ? std::numeric_limits<float>::infinity() : nearest[0].second;

    bool merge = has_typical_distance_ && !nearest.empty() &&
                 distance < config_.merge_ratio * typical_distance_;
    if (std::isfinite(distance)) {
        typical_distance_ = has_typical_distance_ ? 0.99f * typical_distance_ + 0.01f * distance : distance;
        has_typical_distance_ = true;
    }

    if (merge) {
        Prototype& p = prototypes_[nearest[0].first];
        p.count++;
        float rate = 1.0f / static_cast<float>(std::min<uint32_t>(p.count, 20));
        for (size_t i = 0; i < point.size(); ++i) {
            p.center[i] += rate * (point[i] - p.center[i]);
        }
        for (size_t i = 0; i < observed.size() && i < p.observed.size(); ++i) {
            if (std::isnan(observed[i])) continue;
            p.observed[i] = std::isnan(p.observed[i]) ? observed[i]
                                                      : p.observed[i] + rate * (observed[i] - p.observed[i]);
        }
        p.last_used = step;
        return {p.id, distance, false};
    }

    if (prototypes_.size() >= config_.capacity) {
        Prune(step);
    }

    Prototype p;
    p.id = PatternID::Generate();
    p.center = point;
    p.count = 1;
    p.last_used = step;
    p.class_weights.resize(class_outputs_);
    p.class_counts.assign(class_outputs_, 0);
    p.class_serials.assign(class_outputs_, 0);
    for (size_t size : value_sizes_) {
        p.value_means.emplace_back(size, 0.0f);
    }
    p.value_counts.assign(value_sizes_.size(), 0);
    p.value_serials.assign(value_sizes_.size(), 0);
    p.observed = observed;
    p.observed.resize(observed_size_, std::numeric_limits<float>::quiet_NaN());
    slots_.emplace(p.id, prototypes_.size());
    prototype_bytes_ += PrototypeBytes(p);
    prototypes_.push_back(std::move(p));
    return {prototypes_.back().id, distance, true};
}

void PatternMemory::Remove(size_t slot) {
    PatternID id = prototypes_[slot].id;
    prototype_bytes_ -= PrototypeBytes(prototypes_[slot]);
    size_t writes = 0;
    for (bool outgoing : {true, false}) {
        std::vector<std::pair<PatternID, PatternID>> edges;
        for (const auto* edge : associations_.GetAssociations(id, outgoing)) {
            edges.emplace_back(edge->GetSource(), edge->GetTarget());
        }
        for (const auto& [source, target] : edges) {
            associations_.RemoveAssociation(source, target);
        }
        writes += edges.size();
    }
    AfterAssociationWrites(writes);
    slots_.erase(id);
    if (slot + 1 != prototypes_.size()) {
        prototypes_[slot] = std::move(prototypes_.back());
        slots_[prototypes_[slot].id] = slot;
    }
    prototypes_.pop_back();
}

void PatternMemory::AddLabel(PatternID id, size_t class_output, size_t class_index) {
    Prototype* p = Find(id);
    if (!p) return;
    auto& weights = p->class_weights.at(class_output);
    if (weights.size() <= class_index) {
        prototype_bytes_ += (class_index + 1 - weights.size()) * sizeof(float);
        weights.resize(class_index + 1, 0.0f);
    }
    uint32_t n = ++p->class_counts[class_output];
    p->class_serials[class_output] = ++class_serial_[class_output];
    float rate = std::max(config_.label_rate, 1.0f / static_cast<float>(n));
    for (float& w : weights) w *= 1.0f - rate;
    weights[class_index] += rate;
}

void PatternMemory::AddNegative(PatternID id, size_t class_output, size_t class_index) {
    Prototype* p = Find(id);
    if (!p) return;
    auto& weights = p->class_weights.at(class_output);
    if (class_index < weights.size()) {
        weights[class_index] *= 1.0f - config_.label_rate;
    }
}

void PatternMemory::AddValue(PatternID id, size_t value_output, const std::vector<float>& values) {
    Prototype* p = Find(id);
    if (!p) return;
    auto& mean = p->value_means.at(value_output);
    uint32_t n = ++p->value_counts[value_output];
    p->value_serials[value_output] = ++value_serial_[value_output];
    float rate = std::max(config_.label_rate, 1.0f / static_cast<float>(n));
    for (size_t i = 0; i < mean.size() && i < values.size(); ++i) {
        if (!std::isnan(values[i])) mean[i] += rate * (values[i] - mean[i]);
    }
}

std::optional<std::vector<float>> PatternMemory::ClassFromNeighbors(
    const std::vector<std::pair<size_t, float>>& neighbors, size_t class_output, size_t class_count) const {
    if (neighbors.empty() || class_count == 0) return std::nullopt;
    float epsilon = 0.1f * typical_distance_ + 1e-6f;
    std::vector<float> votes(class_count, 0.0f);
    double total = 0.0;
    for (const auto& [slot, distance] : neighbors) {
        float weight = 1.0f / (distance + epsilon);
        const auto& weights = prototypes_[slot].class_weights[class_output];
        for (size_t c = 0; c < weights.size() && c < class_count; ++c) {
            votes[c] += weight * weights[c];
            total += weight * weights[c];
        }
    }
    if (total <= 0.0) return std::nullopt;
    for (float& v : votes) v = static_cast<float>(v / total);
    return votes;
}

std::optional<std::vector<float>> PatternMemory::ValueFromNeighbors(
    const std::vector<std::pair<size_t, float>>& neighbors, size_t value_output) const {
    if (neighbors.empty()) return std::nullopt;
    float epsilon = 0.1f * typical_distance_ + 1e-6f;
    std::vector<float> mean(value_sizes_[value_output], 0.0f);
    double total = 0.0;
    for (const auto& [slot, distance] : neighbors) {
        float weight = 1.0f / (distance + epsilon);
        const auto& values = prototypes_[slot].value_means[value_output];
        for (size_t i = 0; i < mean.size(); ++i) mean[i] += weight * values[i];
        total += weight;
    }
    for (float& v : mean) v = static_cast<float>(v / total);
    return mean;
}

PatternMemory::Votes PatternMemory::ClassVotes(const std::vector<float>& point, size_t class_output,
                                               size_t class_count, size_t k, size_t horizon) const {
    uint64_t newest = class_serial_[class_output];
    uint64_t oldest = horizon > 0 && newest > horizon ? newest - horizon : 0;
    auto [all, recent] = NearestTwo(
        point, k, [&](const Prototype& p) { return p.class_counts[class_output] > 0; },
        [&](const Prototype& p) { return p.class_counts[class_output] > 0 && p.class_serials[class_output] > oldest; });
    return {ClassFromNeighbors(recent, class_output, class_count),
            ClassFromNeighbors(all, class_output, class_count)};
}

PatternMemory::Votes PatternMemory::ValueVotes(const std::vector<float>& point, size_t value_output, size_t k,
                                               size_t horizon) const {
    uint64_t newest = value_serial_[value_output];
    uint64_t oldest = horizon > 0 && newest > horizon ? newest - horizon : 0;
    auto [all, recent] = NearestTwo(
        point, k, [&](const Prototype& p) { return p.value_counts[value_output] > 0; },
        [&](const Prototype& p) { return p.value_counts[value_output] > 0 && p.value_serials[value_output] > oldest; });
    return {ValueFromNeighbors(recent, value_output), ValueFromNeighbors(all, value_output)};
}

std::optional<std::vector<float>> PatternMemory::ClassVote(const std::vector<float>& point,
                                                           size_t class_output, size_t class_count,
                                                           size_t k, size_t horizon) const {
    Votes votes = ClassVotes(point, class_output, class_count, k, horizon);
    return horizon > 0 ? votes.short_term : votes.long_term;
}

std::optional<std::vector<float>> PatternMemory::ValueVote(const std::vector<float>& point,
                                                           size_t value_output, size_t k,
                                                           size_t horizon) const {
    Votes votes = ValueVotes(point, value_output, k, horizon);
    return horizon > 0 ? votes.short_term : votes.long_term;
}

void PatternMemory::RecordTransition(PatternID from, PatternID to) {
    if (!Contains(from) || !Contains(to)) return;
    const float rate = config_.transition_rate;
    std::vector<std::pair<PatternID, float>> edges;
    for (const auto* edge : associations_.GetAssociations(from, true)) {
        if (Contains(edge->GetTarget())) {  // skip state -> action edges
            edges.emplace_back(edge->GetTarget(), edge->GetStrength());
        }
    }
    bool found = false;
    for (auto [target, strength] : edges) {
        if (target == to) {
            found = true;
            strength += rate * (1.0f - strength);
        } else {
            strength *= 1.0f - rate;
        }
        if (strength < kMinTransitionStrength) {
            associations_.RemoveAssociation(from, target);
        } else {
            associations_.SetAssociationStrength(from, target, strength);
        }
    }
    if (!found) {
        associations_.AddAssociation(from, to, AssociationType::CAUSAL, rate);
    }
    AfterAssociationWrites(edges.size());
}

std::optional<std::vector<float>> PatternMemory::PredictNextObserved(PatternID current) const {
    // The 3 strongest transitions to stored patterns; ties by position in
    // the store, so the result does not depend on the order or the values
    // of pattern IDs (which differ between a run and its restored copy)
    std::vector<std::pair<size_t, float>> ranked;  // slot, strength
    for (const auto* edge : associations_.GetAssociations(current, true)) {
        auto it = slots_.find(edge->GetTarget());
        if (it != slots_.end()) ranked.emplace_back(it->second, edge->GetStrength());
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    if (ranked.size() > 3) ranked.resize(3);
    std::vector<double> sum(observed_size_, 0.0), weight(observed_size_, 0.0);
    for (const auto& [slot, strength] : ranked) {
        const auto& observed = prototypes_[slot].observed;
        for (size_t i = 0; i < observed_size_; ++i) {
            if (std::isnan(observed[i])) continue;
            sum[i] += strength * observed[i];
            weight[i] += strength;
        }
    }
    std::vector<float> result(observed_size_, std::numeric_limits<float>::quiet_NaN());
    bool any = false;
    for (size_t i = 0; i < observed_size_; ++i) {
        if (weight[i] > 0.0) {
            result[i] = static_cast<float>(sum[i] / weight[i]);
            any = true;
        }
    }
    if (!any) return std::nullopt;
    return result;
}

void PatternMemory::ReinforceAction(PatternID state, PatternID action, float reward) {
    if (!Contains(state) || reward == 0.0f || std::isnan(reward)) return;
    const auto* edge = associations_.GetAssociationMatrix().GetAssociation(state, action);
    float strength = edge ? edge->GetStrength() : kNeutralActionStrength;
    const float rate = config_.transition_rate;
    strength = reward > 0.0f ? strength + rate * (1.0f - strength) : strength * (1.0f - rate);
    if (edge) {
        associations_.SetAssociationStrength(state, action, strength);
        AfterAssociationWrites(1);
    } else {
        associations_.AddAssociation(state, action, AssociationType::FUNCTIONAL, strength);
    }
}

std::vector<float> PatternMemory::ActionStrengths(PatternID state, const std::vector<PatternID>& actions) const {
    std::vector<float> strengths(actions.size(), kNeutralActionStrength);
    for (size_t a = 0; a < actions.size(); ++a) {
        if (const auto* edge = associations_.GetAssociationMatrix().GetAssociation(state, actions[a])) {
            strengths[a] = edge->GetStrength();
        }
    }
    return strengths;
}

namespace {

template <typename T>
void WriteVector(std::ostream& out, const std::vector<T>& values) {
    io::WriteSize(out, values.size());
    for (const T& v : values) io::Write<T>(out, v);
}

template <typename T>
std::vector<T> ReadVector(std::istream& in) {
    std::vector<T> values(io::ReadSize(in, 1u << 30));
    for (T& v : values) v = io::Read<T>(in);
    return values;
}

}  // namespace

void PatternMemory::SaveState(std::ostream& out) const {
    io::WriteSize(out, input_size_);
    io::WriteSize(out, config_.dimension);
    io::WriteSize(out, config_.memory_budget_bytes);
    WriteVector(out, projection_);
    WriteVector(out, class_serial_);
    WriteVector(out, value_serial_);
    io::Write<float>(out, typical_distance_);
    io::Write<uint8_t>(out, has_typical_distance_ ? 1 : 0);
    io::WriteSize(out, prototype_bytes_);
    io::WriteSize(out, writes_since_compact_);
    io::Write<uint64_t>(out, pruned_);

    io::WriteSize(out, prototypes_.size());
    for (const auto& p : prototypes_) {
        io::Write<uint64_t>(out, p.id.value());
        WriteVector(out, p.center);
        io::Write<uint32_t>(out, p.count);
        io::Write<uint64_t>(out, p.last_used);
        io::WriteSize(out, p.class_weights.size());
        for (const auto& w : p.class_weights) WriteVector(out, w);
        WriteVector(out, p.class_counts);
        WriteVector(out, p.class_serials);
        io::WriteSize(out, p.value_means.size());
        for (const auto& m : p.value_means) WriteVector(out, m);
        WriteVector(out, p.value_counts);
        WriteVector(out, p.value_serials);
        WriteVector(out, p.observed);
    }
    // Associations, per source pattern in the store's order
    for (const auto& p : prototypes_) {
        auto edges = associations_.GetAssociations(p.id, true);
        io::WriteSize(out, edges.size());
        for (const auto* edge : edges) {
            io::Write<uint64_t>(out, edge->GetTarget().value());
            io::Write<uint8_t>(out, static_cast<uint8_t>(edge->GetType()));
            io::Write<float>(out, edge->GetStrength());
        }
    }
}

void PatternMemory::LoadState(std::istream& in) {
    if (io::ReadSize(in, 1u << 30) != input_size_ || io::ReadSize(in, 1u << 20) != config_.dimension) {
        throw std::runtime_error("PatternMemory::LoadState: saved state has other sizes");
    }
    config_.memory_budget_bytes = io::ReadSize(in);
    projection_ = ReadVector<float>(in);
    class_serial_ = ReadVector<uint64_t>(in);
    value_serial_ = ReadVector<uint64_t>(in);
    typical_distance_ = io::Read<float>(in);
    has_typical_distance_ = io::Read<uint8_t>(in) != 0;
    prototype_bytes_ = io::ReadSize(in);
    writes_since_compact_ = io::ReadSize(in);
    pruned_ = io::Read<uint64_t>(in);
    if (projection_.size() != input_size_ * config_.dimension) {
        throw std::runtime_error("PatternMemory::LoadState: bad projection size");
    }

    prototypes_.clear();
    slots_.clear();
    prototypes_.resize(io::ReadSize(in, 1u << 30));
    for (size_t slot = 0; slot < prototypes_.size(); ++slot) {
        Prototype& p = prototypes_[slot];
        p.id = PatternID(io::Read<uint64_t>(in));
        p.center = ReadVector<float>(in);
        p.count = io::Read<uint32_t>(in);
        p.last_used = io::Read<uint64_t>(in);
        p.class_weights.resize(io::ReadSize(in, 1u << 20));
        for (auto& w : p.class_weights) w = ReadVector<float>(in);
        p.class_counts = ReadVector<uint32_t>(in);
        p.class_serials = ReadVector<uint64_t>(in);
        p.value_means.resize(io::ReadSize(in, 1u << 20));
        for (auto& m : p.value_means) m = ReadVector<float>(in);
        p.value_counts = ReadVector<uint32_t>(in);
        p.value_serials = ReadVector<uint64_t>(in);
        p.observed = ReadVector<float>(in);
        if (p.center.size() != config_.dimension) throw std::runtime_error("PatternMemory::LoadState: bad pattern");
        slots_.emplace(p.id, slot);
    }
    associations_.GetAssociationMatrix().Clear();
    for (const auto& p : prototypes_) {
        size_t edges = io::ReadSize(in, 1u << 30);
        for (size_t e = 0; e < edges; ++e) {
            PatternID target(io::Read<uint64_t>(in));
            auto type = io::Read<uint8_t>(in);
            float strength = io::Read<float>(in);
            if (type > static_cast<uint8_t>(AssociationType::COMPOSITIONAL)) {
                throw std::runtime_error("PatternMemory::LoadState: bad association type");
            }
            associations_.AddAssociation(p.id, target, static_cast<AssociationType>(type), strength);
        }
    }
}

PatternID::ValueType PatternMemory::MaxPatternId() const {
    PatternID::ValueType max = 0;
    for (const auto& p : prototypes_) max = std::max(max, p.id.value());
    return max;
}

size_t PatternMemory::PrototypeBytes(const Prototype& p) const {
    size_t bytes = sizeof(Prototype) + 64;  // + slot map entry
    bytes += (p.center.size() + p.observed.size()) * sizeof(float);
    for (const auto& w : p.class_weights) bytes += sizeof(w) + w.size() * sizeof(float);
    for (const auto& m : p.value_means) bytes += sizeof(m) + m.size() * sizeof(float);
    bytes += (p.class_counts.size() + p.value_counts.size()) * sizeof(uint32_t);
    bytes += (p.class_serials.size() + p.value_serials.size()) * sizeof(uint64_t);
    return bytes;
}

size_t PatternMemory::MemoryBytes() const {
    size_t edges = associations_.GetAssociationCount() + writes_since_compact_;
    return prototype_bytes_ + projection_.size() * sizeof(float) + edges * kEdgeBytes;
}

bool PatternMemory::OverBudget() const {
    if (config_.memory_budget_bytes == 0) return false;
    // Leave room for the pattern about to be added and a few associations
    size_t headroom = (prototypes_.empty() ? 0 : prototype_bytes_ / prototypes_.size()) + 16 * kEdgeBytes;
    return MemoryBytes() + headroom > config_.memory_budget_bytes;
}

void PatternMemory::AfterAssociationWrites(size_t writes) {
    // Updated and removed edges are retired, not freed, until Compact();
    // compacting once retired edges outnumber live ones bounds the waste
    writes_since_compact_ += writes;
    size_t limit = std::max<size_t>(10000, associations_.GetAssociationCount());
    if (config_.memory_budget_bytes > 0) {
        limit = std::min(limit, std::max<size_t>(64, config_.memory_budget_bytes / (10 * kEdgeBytes)));
    }
    if (writes_since_compact_ > limit) {
        associations_.Compact();
        writes_since_compact_ = 0;
    }
}

float PatternMemory::Utility(const Prototype& p, uint64_t step) const {
    uint64_t accesses = p.count;
    for (uint32_t n : p.class_counts) accesses += n;
    for (uint32_t n : p.value_counts) accesses += n;

    // Summed in sorted order: independent of the association store's order
    std::vector<float> strengths;
    for (const auto* edge : associations_.GetAssociations(p.id, true)) strengths.push_back(edge->GetStrength());
    std::sort(strengths.begin(), strengths.end());
    float strength = 0.0f;
    for (float s : strengths) strength += s;
    if (!strengths.empty()) strength /= static_cast<float>(strengths.size());

    // Confidence: how consistently the pattern's labels agree (0.5 unlabelled)
    float confidence = 0.5f;
    for (size_t o = 0; o < p.class_weights.size(); ++o) {
        if (p.class_counts[o] == 0 || p.class_weights[o].empty()) continue;
        float total = 0.0f, top = 0.0f;
        for (float w : p.class_weights[o]) {
            total += w;
            top = std::max(top, w);
        }
        if (total > 0.0f) confidence = top / total;
    }
    uint64_t age = step > p.last_used ? step - p.last_used : 0;
    return utility_.CalculateUtility(accesses, std::chrono::hours(age), strength, confidence);
}

float PatternMemory::Utility(PatternID id, uint64_t step) const {
    auto it = slots_.find(id);
    return it == slots_.end() ? 0.0f : Utility(prototypes_[it->second], step);
}

void PatternMemory::Prune(uint64_t step) {
    // Remove the lowest-utility share at once (amortizes scoring)
    size_t count = std::max<size_t>(1, static_cast<size_t>(config_.prune_fraction * prototypes_.size()));
    std::vector<std::pair<float, PatternID>> scored;
    scored.reserve(prototypes_.size());
    for (const auto& p : prototypes_) scored.emplace_back(Utility(p, step), p.id);
    count = std::min(count, scored.size());
    std::nth_element(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(count - 1), scored.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t i = 0; i < count; ++i) {
        auto it = slots_.find(scored[i].second);
        if (it != slots_.end()) Remove(it->second);
    }
    pruned_ += count;
    // Still over the byte budget (e.g. many associations): keep pruning
    if (OverBudget() && prototypes_.size() > 1) Prune(step);
}

}  // namespace agent
}  // namespace dpan
