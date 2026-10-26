// File: src/agent/replay_buffer.cpp
#include "agent/replay_buffer.hpp"

#include "core/binary_io.hpp"

#include <sstream>

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
    // Sizes, not capacities: the estimate depends only on the sample's
    // contents, so a restored buffer makes the same eviction decisions
    bytes += (indices.size() + next_indices.size()) * sizeof(uint32_t);
    bytes += (values.size() + next_values.size()) * sizeof(float);
    bytes += targets.size() * sizeof(HeadTarget);
    for (const auto& target : targets) bytes += target.values.size() * sizeof(float);
    bytes += continuous.size() * sizeof(ContinuousTarget);
    for (const auto& target : continuous) bytes += target.action.size() * sizeof(float);
    return bytes;
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

void Sample::Save(std::ostream& out) const {
    io::Write<uint64_t>(out, step);
    WriteVector(out, indices);
    WriteVector(out, values);
    io::WriteSize(out, targets.size());
    for (const auto& t : targets) {
        io::Write<int32_t>(out, t.label);
        io::Write<int32_t>(out, t.negative);
        WriteVector(out, t.values);
        io::Write<int32_t>(out, t.action);
        io::Write<float>(out, t.reward);
        io::Write<uint8_t>(out, t.exploratory ? 1 : 0);
    }
    io::WriteSize(out, continuous.size());
    for (const auto& t : continuous) {
        WriteVector(out, t.action);
        io::Write<float>(out, t.reward);
    }
    io::Write<uint8_t>(out, reconstruct ? 1 : 0);
    WriteVector(out, next_indices);
    WriteVector(out, next_values);
    io::Write<uint8_t>(out, has_next ? 1 : 0);
    io::Write<uint8_t>(out, terminal ? 1 : 0);
    io::Write<int32_t>(out, balance_class);
    io::Write<uint64_t>(out, key);
    io::Write<uint8_t>(out, held_out ? 1 : 0);
    io::WriteSize(out, stored_bytes);
}

std::shared_ptr<Sample> Sample::Load(std::istream& in) {
    auto sample = std::make_shared<Sample>();
    sample->step = io::Read<uint64_t>(in);
    sample->indices = ReadVector<uint32_t>(in);
    sample->values = ReadVector<float>(in);
    sample->targets.resize(io::ReadSize(in, 1u << 16));
    for (auto& t : sample->targets) {
        t.label = io::Read<int32_t>(in);
        t.negative = io::Read<int32_t>(in);
        t.values = ReadVector<float>(in);
        t.action = io::Read<int32_t>(in);
        t.reward = io::Read<float>(in);
        t.exploratory = io::Read<uint8_t>(in) != 0;
    }
    sample->continuous.resize(io::ReadSize(in, 1u << 16));
    for (auto& t : sample->continuous) {
        t.action = ReadVector<float>(in);
        t.reward = io::Read<float>(in);
    }
    sample->reconstruct = io::Read<uint8_t>(in) != 0;
    sample->next_indices = ReadVector<uint32_t>(in);
    sample->next_values = ReadVector<float>(in);
    sample->has_next = io::Read<uint8_t>(in) != 0;
    sample->terminal = io::Read<uint8_t>(in) != 0;
    sample->balance_class = io::Read<int32_t>(in);
    sample->key = io::Read<uint64_t>(in);
    sample->held_out = io::Read<uint8_t>(in) != 0;
    sample->stored_bytes = io::ReadSize(in);
    if (sample->indices.size() != sample->values.size() || sample->next_indices.size() != sample->next_values.size()) {
        throw std::runtime_error("Sample::Load: malformed sample");
    }
    return sample;
}

void ReplayBuffer::SaveState(std::ostream& out, const std::function<uint64_t(const Sample*)>& id) const {
    io::WriteSize(out, budget_);
    io::WriteSize(out, bytes_);
    std::ostringstream rng;
    rng << rng_;
    io::WriteString(out, rng.str());
    io::WriteSize(out, samples_.size());
    for (const auto& sample : samples_) io::Write<uint64_t>(out, id(sample.get()));
}

void ReplayBuffer::LoadState(std::istream& in, const std::function<std::shared_ptr<Sample>(uint64_t)>& sample) {
    budget_ = io::ReadSize(in);
    bytes_ = io::ReadSize(in);
    std::istringstream rng(io::ReadString(in, 1u << 20));
    rng >> rng_;
    if (!rng) throw std::runtime_error("ReplayBuffer::LoadState: bad random state");
    samples_.clear();
    class_counts_.clear();
    size_t count = io::ReadSize(in, 1u << 30);
    for (size_t i = 0; i < count; ++i) {
        auto stored = sample(io::Read<uint64_t>(in));
        stored->in_replay = true;
        class_counts_[stored->balance_class]++;
        samples_.push_back(std::move(stored));
    }
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
