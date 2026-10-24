// File: tests/agent/performance_test.cpp
//
// Phase 6 (docs/AGENT_PLAN.md): a training step of a network with a
// 512-wide trunk is at least 3 times faster with BLAS. Runs only in builds
// with DPAN_WITH_BLAS; ctest runs it alone (timing).

#include "agent/network.hpp"
#include "nn/compute.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

namespace dpan {
namespace agent {
namespace {

/// Median time of a training step in milliseconds
double StepMilliseconds(const AgentNetwork& prototype, const nn::Matrix& inputs,
                        const std::vector<const HeadTarget*>& targets) {
    std::vector<double> runs;
    for (int run = 0; run < 3; ++run) {
        AgentNetwork network(prototype);
        for (int i = 0; i < 3; ++i) network.TrainBatch(inputs, {targets}, {1.0f}, 5.0f);  // warm up
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) network.TrainBatch(inputs, {targets}, {1.0f}, 5.0f);
        runs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 20);
    }
    std::sort(runs.begin(), runs.end());
    return runs[1];
}

TEST(PerformanceTest, BlasMakesWideTrainingStepsAtLeast3TimesFaster) {
    if (!nn::BlasAvailable()) GTEST_SKIP() << "built without BLAS (DPAN_WITH_BLAS=OFF)";
    const bool use_blas = nn::UsingBlas();
    const size_t threads = nn::ThreadCount();

    // 256 numeric inputs (+ missing indicator): trunk width 2 x 257, capped at 512
    AgentNetwork prototype({{"x", InputType::NUMERIC, 0, 257, 0}}, {{"kind", HeadKind::CLASSIFIER, 10}}, 1e-3f, 1, 32, 512);
    ASSERT_EQ(512u, prototype.TrunkWidth());
    std::mt19937 rng(1);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    nn::Matrix inputs(64, 257);  // the agent's batch size
    for (size_t i = 0; i < inputs.Size(); ++i) inputs.Data()[i] = normal(rng);
    std::vector<HeadTarget> labels(64);
    std::vector<const HeadTarget*> targets(64);
    for (size_t i = 0; i < 64; ++i) {
        labels[i].label = static_cast<int>(i % 10);
        targets[i] = &labels[i];
    }

    nn::SetUseBlas(false);
    nn::SetThreadCount(1);
    double plain = StepMilliseconds(prototype, inputs, targets);
    nn::SetThreadCount(threads);
    double threaded = StepMilliseconds(prototype, inputs, targets);
    nn::SetUseBlas(true);
    double blas = StepMilliseconds(prototype, inputs, targets);
    nn::SetUseBlas(use_blas);

    std::cout << "512-wide training step: " << plain << " ms without BLAS (1 thread), " << threaded
              << " ms without BLAS (" << threads << " threads), " << blas << " ms with BLAS: " << plain / blas
              << "x and " << threaded / blas << "x faster\n";
    EXPECT_GE(plain / blas, 3.0);
}

}  // namespace
}  // namespace agent
}  // namespace dpan
