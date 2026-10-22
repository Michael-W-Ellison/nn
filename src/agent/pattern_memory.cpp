// File: src/agent/pattern_memory.cpp
#include "agent/pattern_memory.hpp"

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
      associations_(TransitionConfig()) {
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
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return static_cast<float>(std::sqrt(sum));
}

PatternMemory::Prototype* PatternMemory::Find(PatternID id) {
    auto it = slots_.find(id);
    return it == slots_.end() ? nullptr : &prototypes_[it->second];
}

template <typename Usable>
std::vector<std::pair<size_t, float>> PatternMemory::Nearest(const std::vector<float>& point, size_t k,
                                                             Usable usable) const {
    std::vector<std::pair<size_t, float>> best;
    best.reserve(k + 1);
    for (size_t slot = 0; slot < prototypes_.size(); ++slot) {
        if (!usable(prototypes_[slot])) continue;
        float d = Distance(point, prototypes_[slot].center);
        if (best.size() == k && d >= best.back().second) continue;
        auto pos = std::upper_bound(best.begin(), best.end(), d,
                                    [](float value, const auto& entry) { return value < entry.second; });
        best.insert(pos, {slot, d});
        if (best.size() > k) best.pop_back();
    }
    return best;
}

float PatternMemory::NearestDistance(const std::vector<float>& point) const {
    auto nearest = Nearest(point, 1, [](const Prototype&) { return true; });
    return nearest.empty() ? std::numeric_limits<float>::infinity() : nearest[0].second;
}

PatternMemory::Placement PatternMemory::Insert(const std::vector<float>& point, uint64_t step,
                                               const std::vector<float>& observed) {
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
        auto lru = std::min_element(prototypes_.begin(), prototypes_.end(),
                                    [](const Prototype& a, const Prototype& b) { return a.last_used < b.last_used; });
        Remove(static_cast<size_t>(lru - prototypes_.begin()));
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
    prototypes_.push_back(std::move(p));
    return {prototypes_.back().id, distance, true};
}

void PatternMemory::Remove(size_t slot) {
    PatternID id = prototypes_[slot].id;
    for (bool outgoing : {true, false}) {
        std::vector<std::pair<PatternID, PatternID>> edges;
        for (const auto* edge : associations_.GetAssociations(id, outgoing)) {
            edges.emplace_back(edge->GetSource(), edge->GetTarget());
        }
        for (const auto& [source, target] : edges) {
            associations_.RemoveAssociation(source, target);
        }
    }
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
    if (weights.size() <= class_index) weights.resize(class_index + 1, 0.0f);
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

std::optional<std::vector<float>> PatternMemory::ClassVote(const std::vector<float>& point,
                                                           size_t class_output, size_t class_count,
                                                           size_t k, size_t horizon) const {
    uint64_t newest = class_serial_[class_output];
    uint64_t oldest = horizon > 0 && newest > horizon ? newest - horizon : 0;
    auto neighbors = Nearest(point, k, [&](const Prototype& p) {
        return p.class_counts[class_output] > 0 && p.class_serials[class_output] > oldest;
    });
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

std::optional<std::vector<float>> PatternMemory::ValueVote(const std::vector<float>& point,
                                                           size_t value_output, size_t k,
                                                           size_t horizon) const {
    uint64_t newest = value_serial_[value_output];
    uint64_t oldest = horizon > 0 && newest > horizon ? newest - horizon : 0;
    auto neighbors = Nearest(point, k, [&](const Prototype& p) {
        return p.value_counts[value_output] > 0 && p.value_serials[value_output] > oldest;
    });
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
}

std::optional<std::vector<float>> PatternMemory::PredictNextObserved(PatternID current) const {
    // The 3 strongest transitions to stored patterns
    std::vector<std::pair<PatternID, float>> ranked;
    for (const auto* edge : associations_.GetAssociations(current, true)) {
        if (Contains(edge->GetTarget())) ranked.emplace_back(edge->GetTarget(), edge->GetStrength());
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    if (ranked.size() > 3) ranked.resize(3);
    std::vector<double> sum(observed_size_, 0.0), weight(observed_size_, 0.0);
    for (const auto& [id, strength] : ranked) {
        auto it = slots_.find(id);
        if (it == slots_.end()) continue;
        const auto& observed = prototypes_[it->second].observed;
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

}  // namespace agent
}  // namespace dpan
