// File: src/agent/replay_buffer.hpp
//
// Trainable samples and the replay buffer (docs/AGENT_PLAN.md, section 4).
// When the buffer is full, a new sample replaces one of a few randomly drawn
// candidates: the one from the most frequent class (keeping classes
// balanced), oldest first among equals. Random replacement keeps recent
// samples alongside a random sample of older ones.
#pragma once

#include "agent/network.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace dpan {
namespace agent {

struct Sample {
    uint64_t step{0};

    /// Fused vector, stored sparsely
    std::vector<uint32_t> indices;
    std::vector<float> values;

    /// One entry per network head (raw units for regressors; the agent
    /// standardizes them when it trains)
    std::vector<HeadTarget> targets;

    /// Trains the anomaly autoencoder
    bool reconstruct{false};

    /// Q-learning: the next state of the episode (sparse fused vector), or
    /// terminal if the episode ended after this step
    std::vector<uint32_t> next_indices;
    std::vector<float> next_values;
    bool has_next{false};
    bool terminal{false};

    /// The next step's sample in the same episode, for multi-step returns
    /// (weak, so a stream without episodes does not keep old samples alive)
    std::weak_ptr<Sample> next_sample;

    /// Class used for balancing (-1 = none)
    int balance_class{-1};

    bool in_replay{false};

    void SetFused(const std::vector<float>& fused);
    void Fill(float* row, size_t size) const;
    void SetNext(const Sample& next);
    void FillNext(float* row, size_t size) const;
};

using SamplePtr = std::shared_ptr<Sample>;

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity, uint32_t seed);

    /// Add a sample (no-op if it is already in the buffer)
    void Add(const SamplePtr& sample);

    /// Up to `count` distinct random samples
    std::vector<SamplePtr> Draw(size_t count);

    /// Call after changing a stored sample's balance_class
    void Reclassify(const SamplePtr& sample, int old_class);

    size_t Size() const { return samples_.size(); }
    size_t Capacity() const { return capacity_; }

private:
    void Evict();

    size_t capacity_;
    std::mt19937 rng_;
    std::vector<SamplePtr> samples_;
    std::unordered_map<int, size_t> class_counts_;
};

}  // namespace agent
}  // namespace dpan
