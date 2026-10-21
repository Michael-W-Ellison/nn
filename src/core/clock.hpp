// File: src/core/clock.hpp
//
// Process clock used by Timestamp::Now().
//
// Every time-dependent decision in DPAN (decay, recency, co-occurrence
// windows, maintenance intervals, staleness) reads Timestamp::Now(), which
// asks the current process clock. By default that is the system clock.
// Tests, simulations and replays install a ManualClock to control time and
// make runs reproducible:
//
//   auto clock = std::make_shared<ManualClock>(Timestamp::FromMicros(0));
//   ScopedClock scoped(clock);         // Timestamp::Now() now reads `clock`
//   clock->Advance(std::chrono::seconds(5));
//
// The clock is process-wide: every component and thread sees the same time.
// Installing or replacing it is thread-safe. Elapsed-time measurements for
// performance statistics and the sleep intervals of background threads use
// real time and are not affected.
#pragma once

#include "core/types.hpp"

#include <atomic>
#include <memory>

namespace dpan {

class Clock {
public:
    virtual ~Clock() = default;

    /// Current time; must be thread-safe
    virtual Timestamp Now() const = 0;
};

/// Wall-clock time (std::chrono::system_clock), microseconds since the Unix epoch
class SystemClock : public Clock {
public:
    Timestamp Now() const override;
};

/// Clock that only moves when told to (thread-safe)
class ManualClock : public Clock {
public:
    explicit ManualClock(Timestamp start = Timestamp::FromMicros(0));

    Timestamp Now() const override;

    void Set(Timestamp time);
    void Advance(Timestamp::Duration duration);

private:
    std::atomic<int64_t> micros_;
};

/// Clock currently used by Timestamp::Now()
const Clock& CurrentClock();

/// Install `clock` as the process clock (nullptr restores the system clock).
/// Clocks stay alive until the process exits, so callers that already read
/// the previous clock keep a valid reference.
void SetClock(std::shared_ptr<const Clock> clock);

/// Installs a clock for the lifetime of this object and restores the
/// previous one afterwards
class ScopedClock {
public:
    explicit ScopedClock(std::shared_ptr<const Clock> clock);
    ~ScopedClock();
    ScopedClock(const ScopedClock&) = delete;
    ScopedClock& operator=(const ScopedClock&) = delete;

private:
    std::shared_ptr<const Clock> previous_;
};

}  // namespace dpan
