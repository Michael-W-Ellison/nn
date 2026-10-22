// File: tests/core/clock_test.cpp
//
// Process clock: ManualClock/SystemClock/ScopedClock behaviour, and
// reproducibility of time-dependent learning when driven by a fake clock.
#include "core/clock.hpp"
#include "association/association_learning_system.hpp"
#include "association/categorical_learner.hpp"
#include "memory/memory_manager.hpp"
#include "storage/memory_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <thread>

namespace dpan {
namespace {

TEST(ClockTest, SystemClockIsWallClockTime) {
    // 2020-01-01 in microseconds since the Unix epoch
    constexpr int64_t kYear2020 = 1577836800LL * 1000000LL;
    EXPECT_GT(SystemClock().Now().ToMicros(), kYear2020);
    EXPECT_GT(Timestamp::Now().ToMicros(), kYear2020);
}

TEST(ClockTest, ManualClockMovesOnlyWhenTold) {
    ManualClock clock(Timestamp::FromMicros(1000));
    EXPECT_EQ(1000, clock.Now().ToMicros());
    clock.Advance(std::chrono::milliseconds(5));
    EXPECT_EQ(6000, clock.Now().ToMicros());
    clock.Set(Timestamp::FromMicros(42));
    EXPECT_EQ(42, clock.Now().ToMicros());
}

TEST(ClockTest, ScopedClockInstallsAndRestores) {
    auto outer = std::make_shared<ManualClock>(Timestamp::FromMicros(100));
    auto inner = std::make_shared<ManualClock>(Timestamp::FromMicros(200));
    {
        ScopedClock scoped_outer(outer);
        EXPECT_EQ(100, Timestamp::Now().ToMicros());
        {
            ScopedClock scoped_inner(inner);
            EXPECT_EQ(200, Timestamp::Now().ToMicros());
        }
        EXPECT_EQ(100, Timestamp::Now().ToMicros());
    }
    // Back to the system clock
    EXPECT_GT(Timestamp::Now().ToMicros(), 1000000LL);
}

TEST(ClockTest, VisibleFromOtherThreads) {
    auto clock = std::make_shared<ManualClock>(Timestamp::FromMicros(777));
    ScopedClock scoped(clock);
    int64_t seen = 0;
    std::thread reader([&seen]() { seen = Timestamp::Now().ToMicros(); });
    reader.join();
    EXPECT_EQ(777, seen);
}

TEST(ClockTest, TimestampSerializationUsesUnixEpoch) {
    Timestamp t = Timestamp::FromMicros(1700000000123456LL);
    std::stringstream ss;
    t.Serialize(ss);
    EXPECT_EQ(t, Timestamp::Deserialize(ss));
    EXPECT_EQ(std::chrono::seconds(3), Timestamp::FromMicros(3000000) - Timestamp());
}

// ============================================================================
// Reproducibility under a fake clock
// ============================================================================

/// Values describing the learned state, in a fixed order
std::vector<double> RunTimeDependentScenario() {
    auto clock = std::make_shared<ManualClock>(Timestamp::FromMicros(1'000'000'000'000LL));
    ScopedClock scoped(clock);

    // Fixed IDs so hash-based containers iterate identically in each run
    std::vector<PatternID> ids;
    for (uint64_t i = 0; i < 6; ++i) {
        ids.push_back(PatternID(1'000'000 + i));
    }

    AssociationLearningSystem::Config config;
    config.co_occurrence.window_size = std::chrono::seconds(2);
    config.formation.min_co_occurrences = 3;
    config.auto_decay_interval = std::chrono::seconds(30);
    config.auto_competition_interval = std::chrono::seconds(45);
    config.auto_normalization_interval = std::chrono::seconds(60);
    AssociationLearningSystem system(config);
    MemoryBackend db(MemoryBackend::Config{});

    // Sequences in simulated time: no sleeping needed
    for (int episode = 0; episode < 20; ++episode) {
        for (size_t i = 0; i < ids.size(); ++i) {
            ContextVector context;
            context.Set(episode % 2 == 0 ? "even" : "odd", 1.0f);
            system.RecordPatternActivation(ids[i], context);
            clock->Advance(std::chrono::milliseconds(i % 2 == 0 ? 500 : 3000));
        }
        clock->Advance(std::chrono::seconds(10));  // Auto maintenance runs as time passes
        if (episode % 5 == 4) {
            system.FormNewAssociations(db);
        }
    }
    system.PerformMaintenance();

    std::vector<double> state;
    auto keys = system.GetAssociationMatrix().GetAllAssociationKeys();
    std::sort(keys.begin(), keys.end());
    for (const auto& [source, target] : keys) {
        const auto* edge = system.GetAssociationMatrix().GetAssociation(source, target);
        state.push_back(static_cast<double>(source.value()));
        state.push_back(static_cast<double>(target.value()));
        state.push_back(edge->GetStrength());
        state.push_back(static_cast<double>(edge->GetCoOccurrenceCount()));
    }
    ContextVector even;
    even.Set("even", 1.0f);
    for (const auto& [target, confidence] : system.PredictWithConfidence(ids[0], 3, &even)) {
        state.push_back(static_cast<double>(target.value()));
        state.push_back(confidence);
    }
    auto stats = system.GetStatistics();
    state.push_back(static_cast<double>(stats.last_decay.ToMicros()));
    state.push_back(static_cast<double>(stats.last_competition.ToMicros()));

    // Categorical clustering
    CategoricalLearner categorical;
    for (size_t i = 0; i < ids.size(); ++i) {
        categorical.AddPattern(ids[i], FeatureVector(std::vector<float>{
            static_cast<float>(i % 3), static_cast<float>(i) * 0.5f}));
    }
    categorical.ComputeClusters(2);
    for (const auto& id : ids) {
        state.push_back(static_cast<double>(categorical.GetClusterID(id).value_or(999)));
    }
    return state;
}

TEST(ClockTest, FakeClockRunsAreReproducible) {
    auto first = RunTimeDependentScenario();
    auto second = RunTimeDependentScenario();
    // The scenario must have learned associations (4 values each) besides
    // the 2 maintenance timestamps and 6 cluster IDs
    ASSERT_GT(first.size(), 8u + 4u * 3u);
    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(first[i], second[i]) << "at index " << i;
    }
}

TEST(ClockTest, SimulatedTimeDrivesWindowsAndDecay) {
    auto clock = std::make_shared<ManualClock>(Timestamp::FromMicros(0));
    ScopedClock scoped(clock);

    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::seconds(10);
    CoOccurrenceTracker tracker(config);
    PatternID a(5'000'001), b(5'000'002), c(5'000'003);

    tracker.RecordActivation(a);
    clock->Advance(std::chrono::seconds(5));
    tracker.RecordActivation(b);                  // 5 s after a: co-occur
    clock->Advance(std::chrono::seconds(30));
    tracker.RecordActivation(c);                  // 30 s after b: outside window
    EXPECT_EQ(1u, tracker.GetCoOccurrenceCount(a, b));
    EXPECT_EQ(0u, tracker.GetCoOccurrenceCount(b, c));

    // Edge age follows the clock
    AssociationEdge edge(a, b, AssociationType::CAUSAL, 0.8f);
    clock->Advance(std::chrono::hours(2));
    EXPECT_EQ(std::chrono::hours(2), Timestamp::Now() - edge.GetLastReinforcement());
}

}  // namespace
}  // namespace dpan
