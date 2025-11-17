// File: tests/memory/adaptive_thresholds_test.cpp
#include "memory/adaptive_thresholds.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>

using namespace dpan;

// ============================================================================
// Config Validation Tests (6 tests)
// ============================================================================

TEST(AdaptiveThresholdsTest, ValidConfig) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.5f;
    config.pressure_factor = 1.5f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    config.target_memory_bytes = 4ULL * 1024 * 1024 * 1024;  // 4GB
    config.pressure_update_interval = 30.0f;
    config.target_eviction_rate = 0.3f;
    config.smoothing_factor = 0.2f;

    EXPECT_TRUE(config.IsValid());

    // Should not throw
    EXPECT_NO_THROW(AdaptiveThresholdManager manager(config));
}

TEST(AdaptiveThresholdsTest, InvalidMinMaxThresholds) {
    AdaptiveThresholdManager::Config config;

    // min > max
    config.min_threshold = 0.8f;
    config.max_threshold = 0.2f;
    EXPECT_FALSE(config.IsValid());
    EXPECT_THROW(AdaptiveThresholdManager manager(config), std::invalid_argument);

    // min out of range
    config.min_threshold = -0.1f;
    config.max_threshold = 0.9f;
    EXPECT_FALSE(config.IsValid());

    // max out of range
    config.min_threshold = 0.1f;
    config.max_threshold = 1.5f;
    EXPECT_FALSE(config.IsValid());
}

TEST(AdaptiveThresholdsTest, InvalidBaseline) {
    AdaptiveThresholdManager::Config config;

    // Baseline below min
    config.baseline_threshold = 0.05f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    EXPECT_FALSE(config.IsValid());

    // Baseline above max
    config.baseline_threshold = 0.95f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    EXPECT_FALSE(config.IsValid());
}

TEST(AdaptiveThresholdsTest, InvalidPressureFactor) {
    AdaptiveThresholdManager::Config config;

    // Negative pressure factor
    config.pressure_factor = -1.0f;
    EXPECT_FALSE(config.IsValid());

    // Zero pressure factor
    config.pressure_factor = 0.0f;
    EXPECT_FALSE(config.IsValid());
}

TEST(AdaptiveThresholdsTest, InvalidTargetMemory) {
    AdaptiveThresholdManager::Config config;

    // Zero target memory
    config.target_memory_bytes = 0;
    EXPECT_FALSE(config.IsValid());
}

TEST(AdaptiveThresholdsTest, InvalidEvictionRate) {
    AdaptiveThresholdManager::Config config;

    // Negative eviction rate
    config.target_eviction_rate = -0.1f;
    EXPECT_FALSE(config.IsValid());

    // Eviction rate > 1.0
    config.target_eviction_rate = 1.5f;
    EXPECT_FALSE(config.IsValid());
}

// ============================================================================
// Memory Pressure Tests (5 tests)
// ============================================================================

TEST(AdaptiveThresholdsTest, PressureAtTarget) {
    AdaptiveThresholdManager::Config config;
    config.target_memory_bytes = 1000;
    AdaptiveThresholdManager manager(config);

    // At target: P = 0
    float pressure = manager.ComputeMemoryPressure(1000);
    EXPECT_FLOAT_EQ(pressure, 0.0f);
}

TEST(AdaptiveThresholdsTest, PressureUnderUtilized) {
    AdaptiveThresholdManager::Config config;
    config.target_memory_bytes = 1000;
    AdaptiveThresholdManager manager(config);

    // Under-utilized: P < 0
    // P = (500 - 1000) / 1000 = -0.5
    float pressure = manager.ComputeMemoryPressure(500);
    EXPECT_FLOAT_EQ(pressure, -0.5f);

    // Minimum under-utilization
    // P = (0 - 1000) / 1000 = -1.0
    pressure = manager.ComputeMemoryPressure(0);
    EXPECT_FLOAT_EQ(pressure, -1.0f);
}

TEST(AdaptiveThresholdsTest, PressureOverUtilized) {
    AdaptiveThresholdManager::Config config;
    config.target_memory_bytes = 1000;
    AdaptiveThresholdManager manager(config);

    // Over-utilized: P > 0
    // P = (1500 - 1000) / 1000 = 0.5
    float pressure = manager.ComputeMemoryPressure(1500);
    EXPECT_FLOAT_EQ(pressure, 0.5f);

    // Double the target
    // P = (2000 - 1000) / 1000 = 1.0
    pressure = manager.ComputeMemoryPressure(2000);
    EXPECT_FLOAT_EQ(pressure, 1.0f);
}

TEST(AdaptiveThresholdsTest, PressureExtreme) {
    AdaptiveThresholdManager::Config config;
    config.target_memory_bytes = 1000;
    AdaptiveThresholdManager manager(config);

    // Very high pressure
    // P = (5000 - 1000) / 1000 = 4.0
    float pressure = manager.ComputeMemoryPressure(5000);
    EXPECT_FLOAT_EQ(pressure, 4.0f);
}

TEST(AdaptiveThresholdsTest, PressureFormula) {
    AdaptiveThresholdManager::Config config;
    config.target_memory_bytes = 8ULL * 1024 * 1024 * 1024;  // 8GB
    AdaptiveThresholdManager manager(config);

    size_t target = config.target_memory_bytes;

    // Test formula: P = (M_used - M_target) / M_target

    // 10GB used (25% over)
    size_t used_10gb = 10ULL * 1024 * 1024 * 1024;
    float expected_pressure = static_cast<float>(used_10gb - target) / static_cast<float>(target);
    float actual_pressure = manager.ComputeMemoryPressure(used_10gb);
    EXPECT_FLOAT_EQ(actual_pressure, expected_pressure);
    EXPECT_NEAR(actual_pressure, 0.25f, 0.001f);

    // 6GB used (25% under)
    size_t used_6gb = 6ULL * 1024 * 1024 * 1024;
    expected_pressure = static_cast<float>(used_6gb) / static_cast<float>(target) - 1.0f;
    actual_pressure = manager.ComputeMemoryPressure(used_6gb);
    EXPECT_FLOAT_EQ(actual_pressure, expected_pressure);
    EXPECT_NEAR(actual_pressure, -0.25f, 0.001f);
}

// ============================================================================
// Threshold Adjustment Tests (7 tests)
// ============================================================================

TEST(AdaptiveThresholdsTest, ThresholdIncreasesWithPressure) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.pressure_factor = 2.0f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;  // Very small interval
    config.smoothing_factor = 1.0f;  // No smoothing for testing

    AdaptiveThresholdManager manager(config);

    // Initial threshold
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.3f);

    // Wait to ensure time passes beyond update interval
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Update with high memory usage (pressure = 0.5)
    // T = 0.3 * (1 + 2.0 * 0.5) = 0.3 * 2.0 = 0.6
    manager.UpdateThreshold(1500);
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.6f);
}

TEST(AdaptiveThresholdsTest, ThresholdDecreasesWithNegativePressure) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.5f;
    config.pressure_factor = 1.0f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;  // Very small interval
    config.smoothing_factor = 1.0f;  // No smoothing

    AdaptiveThresholdManager manager(config);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Update with low memory usage (pressure = -0.5)
    // T = 0.5 * (1 + 1.0 * (-0.5)) = 0.5 * 0.5 = 0.25
    manager.UpdateThreshold(500);
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.25f);
}

TEST(AdaptiveThresholdsTest, ThresholdClampedToMax) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.5f;
    config.pressure_factor = 5.0f;  // High sensitivity
    config.min_threshold = 0.1f;
    config.max_threshold = 0.8f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 1.0f;

    AdaptiveThresholdManager manager(config);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Very high pressure would give T = 0.5 * (1 + 5.0 * 2.0) = 5.5
    // But should be clamped to 0.8
    manager.UpdateThreshold(3000);
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.8f);
}

TEST(AdaptiveThresholdsTest, ThresholdClampedToMin) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.5f;
    config.pressure_factor = 5.0f;
    config.min_threshold = 0.2f;
    config.max_threshold = 0.9f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 1.0f;

    AdaptiveThresholdManager manager(config);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Very low pressure would give T = 0.5 * (1 + 5.0 * (-0.9)) = -1.75
    // But should be clamped to 0.2
    manager.UpdateThreshold(100);
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.2f);
}

TEST(AdaptiveThresholdsTest, ThresholdSmoothingWorks) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.pressure_factor = 2.0f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 0.3f;  // EMA smoothing

    AdaptiveThresholdManager manager(config);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Update with pressure = 0.5
    // New threshold (before smoothing) = 0.3 * (1 + 2.0 * 0.5) = 0.6
    // Smoothed: T = 0.3 * 0.6 + 0.7 * 0.3 = 0.18 + 0.21 = 0.39
    manager.UpdateThreshold(1500);
    EXPECT_NEAR(manager.GetCurrentThreshold(), 0.39f, 0.001f);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Second update with same pressure
    // New threshold = 0.6
    // Smoothed: T = 0.3 * 0.6 + 0.7 * 0.39 = 0.18 + 0.273 = 0.453
    manager.UpdateThreshold(1500);
    EXPECT_NEAR(manager.GetCurrentThreshold(), 0.453f, 0.001f);
}

TEST(AdaptiveThresholdsTest, ThresholdUpdateInterval) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.pressure_factor = 2.0f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.1f;  // 100ms interval
    config.smoothing_factor = 1.0f;

    AdaptiveThresholdManager manager(config);

    // First update should work after waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    manager.UpdateThreshold(1500);
    float first_threshold = manager.GetCurrentThreshold();
    EXPECT_GT(first_threshold, 0.3f);

    // Second update immediately after should not change threshold
    manager.UpdateThreshold(500);  // Different memory value
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), first_threshold);

    // Wait for interval to pass
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Now update should work
    manager.UpdateThreshold(500);
    EXPECT_LT(manager.GetCurrentThreshold(), first_threshold);
}

TEST(AdaptiveThresholdsTest, ThresholdResetToBaseline) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.4f;
    config.pressure_factor = 2.0f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 1.0f;

    AdaptiveThresholdManager manager(config);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Adjust threshold
    manager.UpdateThreshold(1500);
    EXPECT_NE(manager.GetCurrentThreshold(), 0.4f);

    // Reset should restore baseline
    manager.Reset();
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.4f);

    auto stats = manager.GetStats();
    EXPECT_FLOAT_EQ(stats.memory_pressure, 0.0f);
    EXPECT_EQ(stats.current_memory_bytes, 0);
    EXPECT_EQ(stats.pattern_count, 0);
}

// ============================================================================
// Percentile-Based Tests (5 tests)
// ============================================================================

TEST(AdaptiveThresholdsTest, PercentileThresholdBasic) {
    AdaptiveThresholdManager::Config config;
    config.target_eviction_rate = 0.2f;  // 20th percentile
    AdaptiveThresholdManager manager(config);

    // Utilities: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    std::vector<float> utilities = {0.5f, 0.1f, 0.9f, 0.3f, 0.7f,
                                     0.2f, 0.8f, 0.4f, 1.0f, 0.6f};

    // 20th percentile of 10 items = index 2 (when sorted)
    // Sorted: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    // Index 2 = 0.3
    float threshold = manager.ComputePercentileThreshold(utilities);
    EXPECT_FLOAT_EQ(threshold, 0.3f);
}

TEST(AdaptiveThresholdsTest, PercentileThresholdEmptyUtilities) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.35f;
    config.target_eviction_rate = 0.2f;
    AdaptiveThresholdManager manager(config);

    std::vector<float> empty_utilities;

    // Should return baseline when empty
    float threshold = manager.ComputePercentileThreshold(empty_utilities);
    EXPECT_FLOAT_EQ(threshold, 0.35f);
}

TEST(AdaptiveThresholdsTest, PercentileThreshold20Percent) {
    AdaptiveThresholdManager::Config config;
    config.target_eviction_rate = 0.2f;  // 20%
    AdaptiveThresholdManager manager(config);

    // 100 utilities from 0.01 to 1.00
    std::vector<float> utilities;
    for (int i = 1; i <= 100; ++i) {
        utilities.push_back(i / 100.0f);
    }

    // 20% of 100 = index 20
    // utilities[20] = 0.21
    float threshold = manager.ComputePercentileThreshold(utilities);
    EXPECT_FLOAT_EQ(threshold, 0.21f);
}

TEST(AdaptiveThresholdsTest, PercentileThreshold50Percent) {
    AdaptiveThresholdManager::Config config;
    config.target_eviction_rate = 0.5f;  // Median
    AdaptiveThresholdManager manager(config);

    std::vector<float> utilities = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                                     0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    // 50% of 10 = index 5
    // Sorted utilities[5] = 0.6
    float threshold = manager.ComputePercentileThreshold(utilities);
    EXPECT_FLOAT_EQ(threshold, 0.6f);
}

TEST(AdaptiveThresholdsTest, PercentileUpdateMode) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.use_percentile = true;
    config.target_eviction_rate = 0.25f;
    config.smoothing_factor = 1.0f;  // No smoothing
    config.min_threshold = 0.0f;
    config.max_threshold = 1.0f;
    AdaptiveThresholdManager manager(config);

    // Initial threshold is baseline
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.3f);

    // Update with utilities
    std::vector<float> utilities = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                                     0.6f, 0.7f, 0.8f};

    // 25% of 8 = index 2
    // Sorted utilities[2] = 0.3
    manager.UpdateThresholdFromUtilities(utilities);
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.3f);

    // Different utilities
    utilities = {0.2f, 0.4f, 0.6f, 0.8f};
    // 25% of 4 = index 1
    // Sorted utilities[1] = 0.4
    manager.UpdateThresholdFromUtilities(utilities);
    EXPECT_FLOAT_EQ(manager.GetCurrentThreshold(), 0.4f);
}

// ============================================================================
// Statistics Tests (3 tests)
// ============================================================================

TEST(AdaptiveThresholdsTest, GetStatsReturnsCorrectValues) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.35f;
    config.target_memory_bytes = 5000;
    AdaptiveThresholdManager manager(config);

    auto stats = manager.GetStats();

    EXPECT_FLOAT_EQ(stats.current_threshold, 0.35f);
    EXPECT_FLOAT_EQ(stats.baseline_threshold, 0.35f);
    EXPECT_FLOAT_EQ(stats.memory_pressure, 0.0f);
    EXPECT_EQ(stats.target_memory_bytes, 5000);
    EXPECT_EQ(stats.current_memory_bytes, 0);
    EXPECT_EQ(stats.pattern_count, 0);
}

TEST(AdaptiveThresholdsTest, StatisticsTrackUpdates) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.pressure_factor = 2.0f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 1.0f;
    AdaptiveThresholdManager manager(config);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Update with memory and pattern count
    manager.UpdateThreshold(1500, 100);

    auto stats = manager.GetStats();

    // Threshold: 0.3 * (1 + 2.0 * 0.5) = 0.6
    EXPECT_FLOAT_EQ(stats.current_threshold, 0.6f);
    EXPECT_FLOAT_EQ(stats.memory_pressure, 0.5f);
    EXPECT_EQ(stats.current_memory_bytes, 1500);
    EXPECT_EQ(stats.pattern_count, 100);
    EXPECT_FLOAT_EQ(stats.baseline_threshold, 0.3f);
    EXPECT_EQ(stats.target_memory_bytes, 1000);
}

TEST(AdaptiveThresholdsTest, StatsAfterReset) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.4f;
    config.target_memory_bytes = 2000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 1.0f;
    AdaptiveThresholdManager manager(config);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Adjust state
    manager.UpdateThreshold(3000, 50);

    auto stats_before = manager.GetStats();
    EXPECT_GT(stats_before.current_threshold, 0.4f);
    EXPECT_EQ(stats_before.pattern_count, 50);

    // Reset
    manager.Reset();

    auto stats_after = manager.GetStats();
    EXPECT_FLOAT_EQ(stats_after.current_threshold, 0.4f);
    EXPECT_FLOAT_EQ(stats_after.memory_pressure, 0.0f);
    EXPECT_EQ(stats_after.current_memory_bytes, 0);
    EXPECT_EQ(stats_after.pattern_count, 0);
    EXPECT_FLOAT_EQ(stats_after.baseline_threshold, 0.4f);
}

// ============================================================================
// Integration Tests (4 tests)
// ============================================================================

TEST(AdaptiveThresholdsTest, PressureBasedAdaptationScenario) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.4f;
    config.pressure_factor = 1.0f;
    config.min_threshold = 0.2f;
    config.max_threshold = 0.8f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 1.0f;  // No smoothing for clear demonstration
    AdaptiveThresholdManager manager(config);

    // Scenario: Memory usage gradually increases from baseline to over-capacity

    // Start at target (P = 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.UpdateThreshold(1000, 100);
    float threshold_100pct = manager.GetCurrentThreshold();
    EXPECT_FLOAT_EQ(threshold_100pct, 0.4f);  // At baseline

    // Move to 110% of target (P = 0.1)
    // T = 0.4 * (1 + 1.0 * 0.1) = 0.44
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.UpdateThreshold(1100, 110);
    float threshold_110pct = manager.GetCurrentThreshold();
    EXPECT_FLOAT_EQ(threshold_110pct, 0.44f);
    EXPECT_GT(threshold_110pct, threshold_100pct);

    // Move to 125% of target (P = 0.25)
    // T = 0.4 * (1 + 1.0 * 0.25) = 0.5
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.UpdateThreshold(1250, 125);
    float threshold_125pct = manager.GetCurrentThreshold();
    EXPECT_FLOAT_EQ(threshold_125pct, 0.5f);
    EXPECT_GT(threshold_125pct, threshold_110pct);

    // Over target at 150% (P = 0.5)
    // T = 0.4 * (1 + 1.0 * 0.5) = 0.6
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.UpdateThreshold(1500, 150);
    float threshold_150pct = manager.GetCurrentThreshold();
    EXPECT_FLOAT_EQ(threshold_150pct, 0.6f);
    EXPECT_GT(threshold_150pct, threshold_125pct);

    // All thresholds should be within bounds
    EXPECT_GE(threshold_100pct, config.min_threshold);
    EXPECT_LE(threshold_150pct, config.max_threshold);
}

TEST(AdaptiveThresholdsTest, PercentileBasedAdaptationScenario) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.5f;
    config.use_percentile = true;
    config.target_eviction_rate = 0.3f;  // Target 30% eviction
    config.smoothing_factor = 0.4f;
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    AdaptiveThresholdManager manager(config);

    // Scenario: Utility distribution changes over time

    // High utility patterns
    std::vector<float> high_utilities;
    for (int i = 50; i <= 100; ++i) {
        high_utilities.push_back(i / 100.0f);
    }
    manager.UpdateThresholdFromUtilities(high_utilities);
    float threshold_high = manager.GetCurrentThreshold();
    // 30% of 51 items = index 15, which is 0.65
    EXPECT_GT(threshold_high, 0.5f);

    // Mixed utility patterns
    std::vector<float> mixed_utilities;
    for (int i = 1; i <= 100; ++i) {
        mixed_utilities.push_back(i / 100.0f);
    }
    manager.UpdateThresholdFromUtilities(mixed_utilities);
    float threshold_mixed = manager.GetCurrentThreshold();
    // Should be lower than high-utility scenario
    EXPECT_LT(threshold_mixed, threshold_high);

    // Low utility patterns
    std::vector<float> low_utilities;
    for (int i = 1; i <= 50; ++i) {
        low_utilities.push_back(i / 100.0f);
    }
    manager.UpdateThresholdFromUtilities(low_utilities);
    float threshold_low = manager.GetCurrentThreshold();
    EXPECT_LT(threshold_low, threshold_mixed);
}

TEST(AdaptiveThresholdsTest, MemoryGrowthScenario) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.pressure_factor = 2.0f;
    config.min_threshold = 0.15f;
    config.max_threshold = 0.75f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 1.0f;  // No smoothing for predictability
    AdaptiveThresholdManager manager(config);

    // Simulate memory growth from under-capacity to over-capacity
    std::vector<size_t> memory_progression = {400, 600, 800, 1000, 1200, 1400, 1600};
    std::vector<float> thresholds;

    for (size_t memory : memory_progression) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        manager.UpdateThreshold(memory);
        thresholds.push_back(manager.GetCurrentThreshold());
    }

    // Thresholds should monotonically increase
    for (size_t i = 1; i < thresholds.size(); ++i) {
        EXPECT_GE(thresholds[i], thresholds[i-1])
            << "Threshold should increase or stay same as memory grows";
    }

    // First threshold (40% of target, P = -0.6)
    // T = 0.3 * (1 + 2.0 * (-0.6)) = 0.3 * (-0.2) = -0.06 -> clamped to 0.15
    EXPECT_FLOAT_EQ(thresholds[0], 0.15f);

    // Last threshold (160% of target, P = 0.6)
    // T = 0.3 * (1 + 2.0 * 0.6) = 0.3 * 2.2 = 0.66
    EXPECT_FLOAT_EQ(thresholds[6], 0.66f);
}

TEST(AdaptiveThresholdsTest, MemoryShrinkScenario) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.5f;
    config.pressure_factor = 1.0f;
    config.min_threshold = 0.25f;
    config.max_threshold = 0.85f;
    config.target_memory_bytes = 1000;
    config.pressure_update_interval = 0.001f;
    config.smoothing_factor = 1.0f;
    AdaptiveThresholdManager manager(config);

    // Simulate memory shrinking from over-capacity to under-capacity
    std::vector<size_t> memory_progression = {1600, 1400, 1200, 1000, 800, 600, 400};
    std::vector<float> thresholds;

    for (size_t memory : memory_progression) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        manager.UpdateThreshold(memory);
        thresholds.push_back(manager.GetCurrentThreshold());
    }

    // Thresholds should monotonically decrease
    for (size_t i = 1; i < thresholds.size(); ++i) {
        EXPECT_LE(thresholds[i], thresholds[i-1])
            << "Threshold should decrease or stay same as memory shrinks";
    }

    // First threshold (160% of target, P = 0.6)
    // T = 0.5 * (1 + 1.0 * 0.6) = 0.5 * 1.6 = 0.8
    EXPECT_FLOAT_EQ(thresholds[0], 0.8f);

    // Last threshold (40% of target, P = -0.6)
    // T = 0.5 * (1 + 1.0 * (-0.6)) = 0.5 * 0.4 = 0.2 -> clamped to 0.25
    EXPECT_FLOAT_EQ(thresholds[6], 0.25f);

    // Verify stats at end
    auto stats = manager.GetStats();
    EXPECT_FLOAT_EQ(stats.current_threshold, 0.25f);
    EXPECT_FLOAT_EQ(stats.memory_pressure, -0.6f);
}
