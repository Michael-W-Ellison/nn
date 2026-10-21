// File: src/core/clock.cpp
#include "core/clock.hpp"

#include <chrono>
#include <mutex>
#include <vector>

namespace dpan {

namespace {

struct ClockRegistry {
    std::mutex mutex;
    // Every installed clock is kept alive so raw pointers handed out by
    // CurrentClock() never dangle; clocks are installed rarely.
    std::vector<std::shared_ptr<const Clock>> installed;
    std::shared_ptr<const Clock> current;
    std::atomic<const Clock*> fast{nullptr};
};

ClockRegistry& Registry() {
    static ClockRegistry registry;
    return registry;
}

const Clock& DefaultClock() {
    static const SystemClock clock;
    return clock;
}

}  // namespace

Timestamp SystemClock::Now() const {
    auto since_epoch = std::chrono::system_clock::now().time_since_epoch();
    return Timestamp::FromMicros(
        std::chrono::duration_cast<std::chrono::microseconds>(since_epoch).count());
}

ManualClock::ManualClock(Timestamp start) : micros_(start.ToMicros()) {}

Timestamp ManualClock::Now() const {
    return Timestamp::FromMicros(micros_.load(std::memory_order_acquire));
}

void ManualClock::Set(Timestamp time) {
    micros_.store(time.ToMicros(), std::memory_order_release);
}

void ManualClock::Advance(Timestamp::Duration duration) {
    micros_.fetch_add(duration.count(), std::memory_order_acq_rel);
}

const Clock& CurrentClock() {
    const Clock* clock = Registry().fast.load(std::memory_order_acquire);
    return clock ? *clock : DefaultClock();
}

void SetClock(std::shared_ptr<const Clock> clock) {
    ClockRegistry& registry = Registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (clock) {
        registry.installed.push_back(clock);
    }
    registry.current = clock;
    registry.fast.store(clock.get(), std::memory_order_release);
}

ScopedClock::ScopedClock(std::shared_ptr<const Clock> clock) {
    {
        std::lock_guard<std::mutex> lock(Registry().mutex);
        previous_ = Registry().current;
    }
    SetClock(std::move(clock));
}

ScopedClock::~ScopedClock() {
    SetClock(previous_);
}

}  // namespace dpan
