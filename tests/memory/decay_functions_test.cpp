// File: tests/memory/decay_functions_test.cpp
#include "memory/decay_functions.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>

using namespace dpan;

// ============================================================================
// ExponentialDecay Tests (7 tests)
// ============================================================================

TEST(ExponentialDecayTest, BasicDecayBehavior) {
    ExponentialDecay decay(0.05f);  // Moderate decay rate

    float initial = 1.0f;
    auto one_hour = std::chrono::hours(1);

    float after_one_hour = decay.ApplyDecay(initial, one_hour);

    // After 1 hour with λ=0.05: s(1) = 1.0 * e^(-0.05 * 1) ≈ 0.951
    EXPECT_LT(after_one_hour, initial);
    EXPECT_GT(after_one_hour, 0.9f);
    EXPECT_NEAR(0.951f, after_one_hour, 0.01f);
}

TEST(ExponentialDecayTest, DecayIncreasesWithTime) {
    ExponentialDecay decay(0.05f);

    float initial = 1.0f;
    auto one_hour = std::chrono::hours(1);
    auto ten_hours = std::chrono::hours(10);
    auto hundred_hours = std::chrono::hours(100);

    float after_1h = decay.ApplyDecay(initial, one_hour);
    float after_10h = decay.ApplyDecay(initial, ten_hours);
    float after_100h = decay.ApplyDecay(initial, hundred_hours);

    // Verify monotonic decay
    EXPECT_LT(after_10h, after_1h);
    EXPECT_LT(after_100h, after_10h);

    // Very old patterns should have very low strength
    EXPECT_LT(after_100h, 0.01f);
}

TEST(ExponentialDecayTest, ZeroDecayConstantNoDecay) {
    ExponentialDecay decay(0.0f);  // No decay

    float initial = 1.0f;
    auto very_long_time = std::chrono::hours(10000);

    float result = decay.ApplyDecay(initial, very_long_time);

    EXPECT_FLOAT_EQ(initial, result);
}

TEST(ExponentialDecayTest, HalfLifeCalculation) {
    ExponentialDecay decay(0.05f);

    float half_life = decay.GetHalfLife();

    // Half-life = ln(2) / λ = ln(2) / 0.05 ≈ 13.86 hours
    EXPECT_NEAR(13.86f, half_life, 0.1f);

    // Verify that strength is ~50% after half-life time
    float initial = 1.0f;
    auto half_life_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::duration<float, std::ratio<3600>>(half_life)
    );

    float after_half_life = decay.ApplyDecay(initial, half_life_duration);
    EXPECT_NEAR(0.5f, after_half_life, 0.01f);
}

TEST(ExponentialDecayTest, NeverExceedsInitialStrength) {
    ExponentialDecay decay(0.05f);

    float initial = 0.8f;
    auto elapsed = std::chrono::hours(5);

    float result = decay.ApplyDecay(initial, elapsed);

    EXPECT_LE(result, initial);
    EXPECT_GE(result, 0.0f);
}

TEST(ExponentialDecayTest, ZeroInitialStrength) {
    ExponentialDecay decay(0.05f);

    float initial = 0.0f;
    auto elapsed = std::chrono::hours(10);

    float result = decay.ApplyDecay(initial, elapsed);

    EXPECT_FLOAT_EQ(0.0f, result);
}

TEST(ExponentialDecayTest, ParameterAccessors) {
    ExponentialDecay decay(0.03f);

    EXPECT_FLOAT_EQ(0.03f, decay.GetDecayConstant());

    decay.SetDecayConstant(0.07f);
    EXPECT_FLOAT_EQ(0.07f, decay.GetDecayConstant());

    // Negative values should be clamped to 0
    decay.SetDecayConstant(-0.5f);
    EXPECT_FLOAT_EQ(0.0f, decay.GetDecayConstant());
}

// ============================================================================
// PowerLawDecay Tests (7 tests)
// ============================================================================

TEST(PowerLawDecayTest, BasicDecayBehavior) {
    PowerLawDecay decay(1.0f, 0.5f);  // τ=1.0, β=0.5

    float initial = 1.0f;
    auto one_hour = std::chrono::hours(1);

    float after_one_hour = decay.ApplyDecay(initial, one_hour);

    // After 1 hour: s(1) = 1.0 / (1 + 1/1)^0.5 = 1.0 / 2^0.5 ≈ 0.707
    EXPECT_LT(after_one_hour, initial);
    EXPECT_NEAR(0.707f, after_one_hour, 0.01f);
}

TEST(PowerLawDecayTest, SlowerDecayThanExponential) {
    // Power-law decay typically slower than exponential for long periods
    PowerLawDecay power_decay(1.0f, 0.5f);
    ExponentialDecay exp_decay(0.05f);

    float initial = 1.0f;
    auto long_time = std::chrono::hours(100);

    float power_result = power_decay.ApplyDecay(initial, long_time);
    float exp_result = exp_decay.ApplyDecay(initial, long_time);

    // Power-law should decay slower over long periods
    EXPECT_GT(power_result, exp_result);
}

TEST(PowerLawDecayTest, ExponentAffectsDecayRate) {
    PowerLawDecay low_exp(1.0f, 0.3f);   // Slower decay
    PowerLawDecay high_exp(1.0f, 0.8f);  // Faster decay

    float initial = 1.0f;
    auto elapsed = std::chrono::hours(10);

    float low_result = low_exp.ApplyDecay(initial, elapsed);
    float high_result = high_exp.ApplyDecay(initial, elapsed);

    // Higher exponent = faster decay
    EXPECT_GT(low_result, high_result);
}

TEST(PowerLawDecayTest, TimeConstantAffectsDecay) {
    PowerLawDecay small_tau(0.5f, 0.5f);   // Faster decay
    PowerLawDecay large_tau(5.0f, 0.5f);   // Slower decay

    float initial = 1.0f;
    auto elapsed = std::chrono::hours(5);

    float small_result = small_tau.ApplyDecay(initial, elapsed);
    float large_result = large_tau.ApplyDecay(initial, elapsed);

    // Larger time constant = slower decay
    EXPECT_GT(large_result, small_result);
}

TEST(PowerLawDecayTest, NeverExceedsInitialStrength) {
    PowerLawDecay decay(1.0f, 0.5f);

    float initial = 0.75f;
    auto elapsed = std::chrono::hours(20);

    float result = decay.ApplyDecay(initial, elapsed);

    EXPECT_LE(result, initial);
    EXPECT_GE(result, 0.0f);
}

TEST(PowerLawDecayTest, ZeroInitialStrength) {
    PowerLawDecay decay(1.0f, 0.5f);

    float initial = 0.0f;
    auto elapsed = std::chrono::hours(10);

    float result = decay.ApplyDecay(initial, elapsed);

    EXPECT_FLOAT_EQ(0.0f, result);
}

TEST(PowerLawDecayTest, ParameterAccessors) {
    PowerLawDecay decay(2.5f, 0.6f);

    EXPECT_FLOAT_EQ(2.5f, decay.GetTimeConstant());
    EXPECT_FLOAT_EQ(0.6f, decay.GetExponent());

    decay.SetTimeConstant(3.0f);
    decay.SetExponent(0.8f);

    EXPECT_FLOAT_EQ(3.0f, decay.GetTimeConstant());
    EXPECT_FLOAT_EQ(0.8f, decay.GetExponent());

    // Invalid values should be clamped
    decay.SetTimeConstant(-1.0f);
    EXPECT_GT(decay.GetTimeConstant(), 0.0f);  // Should be clamped to minimum

    decay.SetExponent(-0.5f);
    EXPECT_GE(decay.GetExponent(), 0.0f);  // Should be clamped to 0
}

// ============================================================================
// StepDecay Tests (8 tests)
// ============================================================================

TEST(StepDecayTest, NoDecayBeforeFirstStep) {
    StepDecay decay(0.9f, std::chrono::hours(24));

    float initial = 1.0f;
    auto half_day = std::chrono::hours(12);  // Less than one step

    float result = decay.ApplyDecay(initial, half_day);

    EXPECT_FLOAT_EQ(initial, result);  // No decay yet
}

TEST(StepDecayTest, SingleStepDecay) {
    StepDecay decay(0.9f, std::chrono::hours(24));

    float initial = 1.0f;
    auto one_day = std::chrono::hours(24);

    float result = decay.ApplyDecay(initial, one_day);

    // After 1 step: s = 1.0 * 0.9^1 = 0.9
    EXPECT_NEAR(0.9f, result, 0.001f);
}

TEST(StepDecayTest, MultipleStepsDecay) {
    StepDecay decay(0.8f, std::chrono::hours(24));

    float initial = 1.0f;
    auto three_days = std::chrono::hours(72);

    float result = decay.ApplyDecay(initial, three_days);

    // After 3 steps: s = 1.0 * 0.8^3 = 0.512
    EXPECT_NEAR(0.512f, result, 0.001f);
}

TEST(StepDecayTest, PartialStepIgnored) {
    StepDecay decay(0.9f, std::chrono::hours(24));

    float initial = 1.0f;
    auto one_and_half_days = std::chrono::hours(36);

    float result = decay.ApplyDecay(initial, one_and_half_days);

    // Only 1 complete step, so: s = 1.0 * 0.9^1 = 0.9
    EXPECT_NEAR(0.9f, result, 0.001f);
}

TEST(StepDecayTest, NoDecayWhenFactorIsOne) {
    StepDecay decay(1.0f, std::chrono::hours(24));  // No decay

    float initial = 1.0f;
    auto many_days = std::chrono::hours(1000);

    float result = decay.ApplyDecay(initial, many_days);

    EXPECT_FLOAT_EQ(initial, result);  // No decay
}

TEST(StepDecayTest, HalfLifeCalculation) {
    StepDecay decay(0.9f, std::chrono::hours(24));

    float half_life_steps = decay.GetHalfLifeSteps();

    // Half-life = log(0.5) / log(0.9) ≈ 6.58 steps
    EXPECT_NEAR(6.58f, half_life_steps, 0.1f);

    // Verify ~50% strength after that many steps
    auto half_life_time = std::chrono::hours(static_cast<long>(half_life_steps * 24));
    float result = decay.ApplyDecay(1.0f, half_life_time);

    EXPECT_NEAR(0.5f, result, 0.1f);
}

TEST(StepDecayTest, NeverExceedsInitialStrength) {
    StepDecay decay(0.85f, std::chrono::hours(24));

    float initial = 0.6f;
    auto elapsed = std::chrono::hours(120);  // 5 days

    float result = decay.ApplyDecay(initial, elapsed);

    EXPECT_LE(result, initial);
    EXPECT_GE(result, 0.0f);
}

TEST(StepDecayTest, ParameterAccessors) {
    StepDecay decay(0.85f, std::chrono::hours(12));

    EXPECT_FLOAT_EQ(0.85f, decay.GetDecayFactor());
    EXPECT_EQ(std::chrono::hours(12), decay.GetStepSize());

    decay.SetDecayFactor(0.75f);
    decay.SetStepSize(std::chrono::hours(48));

    EXPECT_FLOAT_EQ(0.75f, decay.GetDecayFactor());
    EXPECT_EQ(std::chrono::hours(48), decay.GetStepSize());

    // Invalid decay factor should be clamped
    decay.SetDecayFactor(1.5f);
    EXPECT_LE(decay.GetDecayFactor(), 1.0f);

    decay.SetDecayFactor(-0.5f);
    EXPECT_GE(decay.GetDecayFactor(), 0.0f);

    // Invalid step size should be rejected
    auto original_step = decay.GetStepSize();
    decay.SetStepSize(std::chrono::hours(0));
    EXPECT_EQ(original_step, decay.GetStepSize());  // Should not change
}

// ============================================================================
// Interface and General Tests (6 tests)
// ============================================================================

TEST(DecayFunctionTest, GetDecayAmount) {
    ExponentialDecay decay(0.05f);

    float initial = 1.0f;
    auto elapsed = std::chrono::hours(10);

    float decayed = decay.ApplyDecay(initial, elapsed);
    float decay_amount = decay.GetDecayAmount(initial, elapsed);

    EXPECT_NEAR(initial - decayed, decay_amount, 0.001f);
    EXPECT_GE(decay_amount, 0.0f);
}

TEST(DecayFunctionTest, CloneFunctionality) {
    ExponentialDecay original(0.03f);

    auto cloned = original.Clone();

    ASSERT_NE(nullptr, cloned);

    // Cloned function should behave identically
    float initial = 1.0f;
    auto elapsed = std::chrono::hours(5);

    float original_result = original.ApplyDecay(initial, elapsed);
    float cloned_result = cloned->ApplyDecay(initial, elapsed);

    EXPECT_FLOAT_EQ(original_result, cloned_result);
}

TEST(DecayFunctionTest, PowerLawClone) {
    PowerLawDecay original(2.0f, 0.7f);

    auto cloned = original.Clone();
    ASSERT_NE(nullptr, cloned);

    float initial = 1.0f;
    auto elapsed = std::chrono::hours(15);

    EXPECT_FLOAT_EQ(
        original.ApplyDecay(initial, elapsed),
        cloned->ApplyDecay(initial, elapsed)
    );
}

TEST(DecayFunctionTest, StepDecayClone) {
    StepDecay original(0.88f, std::chrono::hours(6));

    auto cloned = original.Clone();
    ASSERT_NE(nullptr, cloned);

    float initial = 1.0f;
    auto elapsed = std::chrono::hours(20);

    EXPECT_FLOAT_EQ(
        original.ApplyDecay(initial, elapsed),
        cloned->ApplyDecay(initial, elapsed)
    );
}

TEST(DecayFunctionTest, FactoryFunction) {
    auto exp_decay = CreateDecayFunction("exponential");
    ASSERT_NE(nullptr, exp_decay);
    EXPECT_STREQ("ExponentialDecay", exp_decay->GetName());

    auto power_decay = CreateDecayFunction("powerlaw");
    ASSERT_NE(nullptr, power_decay);
    EXPECT_STREQ("PowerLawDecay", power_decay->GetName());

    auto step_decay = CreateDecayFunction("step");
    ASSERT_NE(nullptr, step_decay);
    EXPECT_STREQ("StepDecay", step_decay->GetName());

    auto invalid = CreateDecayFunction("invalid_name");
    EXPECT_EQ(nullptr, invalid);
}

TEST(DecayFunctionTest, GetNameMethod) {
    ExponentialDecay exp_decay;
    PowerLawDecay power_decay;
    StepDecay step_decay;

    EXPECT_STREQ("ExponentialDecay", exp_decay.GetName());
    EXPECT_STREQ("PowerLawDecay", power_decay.GetName());
    EXPECT_STREQ("StepDecay", step_decay.GetName());
}

// ============================================================================
// Edge Cases and Boundary Tests (5 tests)
// ============================================================================

TEST(DecayFunctionTest, VeryLargeDecayConstant) {
    ExponentialDecay decay(1000.0f);  // Very fast decay

    float initial = 1.0f;
    auto one_hour = std::chrono::hours(1);

    float result = decay.ApplyDecay(initial, one_hour);

    // Should decay almost to zero very quickly
    EXPECT_LT(result, 0.001f);
    EXPECT_GE(result, 0.0f);
}

TEST(DecayFunctionTest, VerySmallStepSize) {
    StepDecay decay(0.99f, std::chrono::minutes(1));  // Steps every minute

    float initial = 1.0f;
    auto one_hour = std::chrono::minutes(60);

    float result = decay.ApplyDecay(initial, one_hour);

    // After 60 steps: s = 1.0 * 0.99^60 ≈ 0.547
    EXPECT_NEAR(0.547f, result, 0.01f);
}

TEST(DecayFunctionTest, ZeroElapsedTime) {
    ExponentialDecay exp_decay(0.05f);
    PowerLawDecay power_decay(1.0f, 0.5f);
    StepDecay step_decay(0.9f, std::chrono::hours(24));

    float initial = 0.8f;
    auto zero_time = std::chrono::microseconds(0);

    // No time elapsed = no decay
    EXPECT_FLOAT_EQ(initial, exp_decay.ApplyDecay(initial, zero_time));
    EXPECT_FLOAT_EQ(initial, power_decay.ApplyDecay(initial, zero_time));
    EXPECT_FLOAT_EQ(initial, step_decay.ApplyDecay(initial, zero_time));
}

TEST(DecayFunctionTest, NegativeStrengthHandling) {
    ExponentialDecay decay(0.05f);

    float negative_strength = -0.5f;
    auto elapsed = std::chrono::hours(5);

    // Should handle gracefully (likely return 0 or clamp)
    float result = decay.ApplyDecay(negative_strength, elapsed);
    EXPECT_LE(result, 0.0f);  // Result should not be positive
}

TEST(DecayFunctionTest, StrengthAboveOne) {
    ExponentialDecay decay(0.05f);

    float high_strength = 1.5f;
    auto elapsed = std::chrono::hours(10);

    float result = decay.ApplyDecay(high_strength, elapsed);

    // Should decay from 1.5, but stay below initial
    EXPECT_LE(result, high_strength);
    EXPECT_GE(result, 0.0f);
}

// ============================================================================
// Performance/Validation Tests (2 tests)
// ============================================================================

TEST(DecayFunctionTest, PerformanceBaseline) {
    ExponentialDecay decay(0.05f);

    float initial = 1.0f;
    auto elapsed = std::chrono::hours(5);

    // Measure time for many decay calculations
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int iterations = 100000;
    volatile float result = 0.0f;  // Prevent optimization

    for (int i = 0; i < iterations; ++i) {
        result = decay.ApplyDecay(initial, elapsed);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double ns_per_call = static_cast<double>(duration.count()) / iterations;

    // Should be very fast (target: <100ns per call)
    EXPECT_LT(ns_per_call, 500.0);  // Conservative upper bound

    // Prevent compiler from optimizing away the loop
    EXPECT_GT(result, 0.0f);
}

TEST(DecayFunctionTest, ConsistencyAcrossCalls) {
    ExponentialDecay decay(0.05f);

    float initial = 1.0f;
    auto elapsed = std::chrono::hours(7);

    // Multiple calls with same parameters should give same result
    float result1 = decay.ApplyDecay(initial, elapsed);
    float result2 = decay.ApplyDecay(initial, elapsed);
    float result3 = decay.ApplyDecay(initial, elapsed);

    EXPECT_FLOAT_EQ(result1, result2);
    EXPECT_FLOAT_EQ(result2, result3);
}
