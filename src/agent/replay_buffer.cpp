// File: src/agent/replay_buffer.cpp
#include "agent/replay_buffer.hpp"

#include <algorithm>
#include <stdexcept>

namespace dpan {
namespace agent {

void Sample::SetFused(const std::vector<float>& fused) {
    indices.clear();
    values.clear();
    for (size_t i = 0; i < fused.size(); ++i) {
        if (fused[i] != 0.0f) {
            indices.push_back(static_cast<uint32_t>(i));
            values.push_back(fused[i]);
        }
    }
}

void Sample::Fill(float* row, size_t size) const {
    std::fill(row, row + size, 0.0f);
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < size) row[indices[i]] = values[i];
    }
}

void Sample::SetNext(const Sample& next) {
    next_indices = next.indices;
    next_values = next.values;
    has_next = true;
}

void Sample::FillNext(float* row, size_t size) const {
    std::fill(row, row + size, 0.0f);
    for (size_t i = 0; i < next_indices.size(); ++i) {
        if (next_indices[i] < size) row[next_indices[i]] = next_values[i];
    }
}

size_t Sample::MemoryBytes() const {
    size_t bytes = sizeof(Sample) + 32;
    bytes += (indices.capacity() + next_indices.capacity()) * sizeof(uint32_t);
    bytes += (values.capacity() + next_values.capacity()) * sizeof(float);
    bytes += targets.capacity() * sizeof(HeadTarget);
    for (const auto& target : targets) bytes += target.values.capacity() * sizeof(float);
    return bytes;
}

ReplayBuffer::ReplayBuffer(size_t capacity, uint32_t seed, size_t memory_budget_bytes)
    : capacity_(capacity), budget_(memory_budget_bytes), rng_(seed) {
    if (capacity == 0) {
        throw std::invalid_argument("ReplayBuffer: capacity must be > 0");
    }
}

void ReplayBuffer::Add(const SamplePtr& sample) {
    size_t bytes = sample->MemoryBytes();
    if (sample->in_replay) {
        bytes_ = bytes_ - sample->stored_bytes + bytes;
        sample->stored_bytes = bytes;
        EnforceBudget(0);
        return;
    }
    if (samples_.size() >= capacity_) {
        Evict();
    }
    EnforceBudget(bytes);
    sample->in_replay = true;
    sample->stored_bytes = bytes;
    bytes_ += bytes;
    samples_.push_back(sample);
    class_counts_[sample->balance_class]++;
}

void ReplayBuffer::SetMemoryBudget(size_t bytes) {
    budget_ = bytes;
    EnforceBudget(0);
}

void ReplayBuffer::EnforceBudget(size_t incoming) {
    while (budget_ > 0 && !samples_.empty() && bytes_ + incoming > budget_) {
        Evict();
    }
}

void ReplayBuffer::Reclassify(const SamplePtr& sample, int old_class) {
    if (!sample->in_replay || old_class == sample->balance_class) return;
    class_counts_[old_class]--;
    class_counts_[sample->balance_class]++;
}

void ReplayBuffer::Evict() {
    constexpr size_t kCandidates = 8;
    std::uniform_int_distribution<size_t> pick(0, samples_.size() - 1);
    size_t victim = pick(rng_);
    for (size_t i = 1; i < kCandidates; ++i) {
        size_t candidate = pick(rng_);
        const auto& a = samples_[candidate];
        const auto& b = samples_[victim];
        size_t count_a = class_counts_[a->balance_class];
        size_t count_b = class_counts_[b->balance_class];
        if (count_a > count_b || (count_a == count_b && a->step < b->step)) {
            victim = candidate;
        }
    }
    Erase(victim);
}

void ReplayBuffer::Erase(size_t index) {
    auto& removed = samples_[index];
    removed->in_replay = false;
    bytes_ -= removed->stored_bytes;
    class_counts_[removed->balance_class]--;
    removed = std::move(samples_.back());
    samples_.pop_back();
}

std::vector<SamplePtr> ReplayBuffer::Draw(size_t count) {
    std::vector<SamplePtr> drawn;
    if (samples_.empty()) return drawn;
    count = std::min(count, samples_.size());
    // Rejection sampling: cheap when count is much smaller than the buffer
    std::vector<size_t> chosen;
    std::uniform_int_distribution<size_t> pick(0, samples_.size() - 1);
    while (chosen.size() < count) {
        size_t index = pick(rng_);
        if (std::find(chosen.begin(), chosen.end(), index) == chosen.end()) {
            chosen.push_back(index);
            drawn.push_back(samples_[index]);
        }
    }
    return drawn;
}

}  // namespace agent
}  // namespace dpan
