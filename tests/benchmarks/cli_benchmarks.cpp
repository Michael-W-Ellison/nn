// File: tests/benchmarks/cli_benchmarks.cpp
//
// Performance benchmarks for DPAN CLI
// Tests command processing, attention mechanism, and conversation workflows

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include "cli/dpan_cli.hpp"
#include "core/types.hpp"

using namespace dpan;
using namespace std::chrono;

// ============================================================================
// Benchmark Helper Functions
// ============================================================================

struct BenchmarkTimer {
    using TimePoint = high_resolution_clock::time_point;

    TimePoint start;

    BenchmarkTimer() : start(high_resolution_clock::now()) {}

    double ElapsedMs() const {
        auto end = high_resolution_clock::now();
        return duration_cast<duration<double, std::milli>>(end - start).count();
    }

    double ElapsedUs() const {
        auto end = high_resolution_clock::now();
        return duration_cast<duration<double, std::micro>>(end - start).count();
    }

    static double MeasureOps(size_t iterations, std::function<void()> fn) {
        BenchmarkTimer timer;
        for (size_t i = 0; i < iterations; ++i) {
            fn();
        }
        double elapsed = timer.ElapsedMs();
        return (iterations / elapsed) * 1000.0; // ops per second
    }
};

std::vector<std::string> GenerateTestInputs(size_t count) {
    std::vector<std::string> inputs;
    inputs.reserve(count);

    std::vector<std::string> templates = {
        "machine learning", "neural networks", "deep learning",
        "artificial intelligence", "data science", "pattern recognition",
        "natural language processing", "computer vision", "reinforcement learning",
        "supervised learning", "unsupervised learning", "transfer learning"
    };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, templates.size() - 1);

    for (size_t i = 0; i < count; ++i) {
        inputs.push_back(templates[dis(gen)]);
    }

    return inputs;
}

// ============================================================================
// Test Fixture
// ============================================================================

class CliBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        cli_ = std::make_unique<DPANCli>();
        cli_->InitializeClean();
    }

    void TearDown() override {
        cli_.reset();
    }

    std::unique_ptr<DPANCli> cli_;
};

// ============================================================================
// Command Processing Benchmarks
// ============================================================================

TEST_F(CliBenchmark, CommandProcessing_EmptyCommand_1000) {
    BenchmarkTimer timer;
    for (size_t i = 0; i < 1000; ++i) {
        cli_->ProcessCommand("");
    }
    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (1000.0 / elapsed) * 1000.0;

    std::cout << "Empty command (1000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 50.0); // Should be very fast
}

TEST_F(CliBenchmark, CommandProcessing_SimpleCommand_1000) {
    BenchmarkTimer timer;
    for (size_t i = 0; i < 1000; ++i) {
        cli_->ProcessCommand("/help");
    }
    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (1000.0 / elapsed) * 1000.0;

    std::cout << "Simple command /help (1000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 500.0); // Should complete in < 500ms
}

TEST_F(CliBenchmark, CommandProcessing_StatsCommand_100) {
    // First, add some data
    for (size_t i = 0; i < 10; ++i) {
        cli_->ProcessCommand("test input " + std::to_string(i));
    }

    BenchmarkTimer timer;
    for (size_t i = 0; i < 100; ++i) {
        cli_->ProcessCommand("/stats");
    }
    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (100.0 / elapsed) * 1000.0;

    std::cout << "Stats command (100): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 500.0); // Should complete in < 500ms
}

// ============================================================================
// Conversation Processing Benchmarks
// ============================================================================

TEST_F(CliBenchmark, ConversationProcessing_BasicInput_100) {
    auto inputs = GenerateTestInputs(100);

    BenchmarkTimer timer;
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }
    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (100.0 / elapsed) * 1000.0;

    std::cout << "Basic conversation input (100): " << elapsed << "ms, "
              << ops_per_sec << " inputs/sec" << std::endl;

    EXPECT_LT(elapsed, 2000.0); // Should complete in < 2s
}

TEST_F(CliBenchmark, ConversationProcessing_BasicInput_1000) {
    auto inputs = GenerateTestInputs(1000);

    BenchmarkTimer timer;
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }
    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (1000.0 / elapsed) * 1000.0;

    std::cout << "Basic conversation input (1000): " << elapsed << "ms, "
              << ops_per_sec << " inputs/sec" << std::endl;

    EXPECT_LT(elapsed, 20000.0); // Should complete in < 20s
}

// ============================================================================
// Attention Mechanism Benchmarks
// ============================================================================

TEST_F(CliBenchmark, AttentionPrediction_vs_BasicPrediction) {
    // Build up some conversation history
    auto inputs = GenerateTestInputs(50);
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }

    // Benchmark basic prediction
    BenchmarkTimer timer_basic;
    for (size_t i = 0; i < 100; ++i) {
        cli_->ProcessCommand("/predict machine");
    }
    double elapsed_basic = timer_basic.ElapsedMs();

    // Enable attention
    cli_->ProcessCommand("/attention");

    // Benchmark attention-enhanced prediction
    BenchmarkTimer timer_attention;
    for (size_t i = 0; i < 100; ++i) {
        cli_->ProcessCommand("/predict machine");
    }
    double elapsed_attention = timer_attention.ElapsedMs();

    double basic_ops_per_sec = (100.0 / elapsed_basic) * 1000.0;
    double attention_ops_per_sec = (100.0 / elapsed_attention) * 1000.0;

    std::cout << "Basic prediction (100): " << elapsed_basic << "ms, "
              << basic_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "Attention prediction (100): " << elapsed_attention << "ms, "
              << attention_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "Overhead: " << (elapsed_attention - elapsed_basic) << "ms ("
              << ((elapsed_attention / elapsed_basic - 1.0) * 100.0) << "%)" << std::endl;

    // Attention should not be more than 3x slower
    EXPECT_LT(elapsed_attention, elapsed_basic * 3.0);
}

TEST_F(CliBenchmark, DetailedPrediction_Overhead) {
    // Build up some conversation history
    auto inputs = GenerateTestInputs(50);
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }

    // Benchmark basic prediction
    BenchmarkTimer timer_basic;
    for (size_t i = 0; i < 100; ++i) {
        cli_->ProcessCommand("/predict machine");
    }
    double elapsed_basic = timer_basic.ElapsedMs();

    // Benchmark detailed prediction
    BenchmarkTimer timer_detailed;
    for (size_t i = 0; i < 100; ++i) {
        cli_->ProcessCommand("/predict-detailed machine");
    }
    double elapsed_detailed = timer_detailed.ElapsedMs();

    std::cout << "Basic predict (100): " << elapsed_basic << "ms" << std::endl;
    std::cout << "Detailed predict (100): " << elapsed_detailed << "ms" << std::endl;
    std::cout << "Overhead: " << (elapsed_detailed - elapsed_basic) << "ms" << std::endl;

    // Detailed should not be significantly slower (mostly just extra output)
    EXPECT_LT(elapsed_detailed, elapsed_basic * 2.0);
}

TEST_F(CliBenchmark, CompareMode_Performance) {
    // Build up some conversation history
    auto inputs = GenerateTestInputs(50);
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }

    // Enable attention
    cli_->ProcessCommand("/attention");

    // Benchmark compare mode (runs both basic and attention)
    BenchmarkTimer timer;
    for (size_t i = 0; i < 50; ++i) {
        cli_->ProcessCommand("/compare machine");
    }
    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (50.0 / elapsed) * 1000.0;

    std::cout << "Compare mode (50): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 5000.0); // Should complete in < 5s
}

// ============================================================================
// Context Tracking Benchmarks
// ============================================================================

TEST_F(CliBenchmark, ContextUpdate_Performance_1000) {
    BenchmarkTimer timer;
    for (size_t i = 0; i < 1000; ++i) {
        cli_->ProcessCommand("test input " + std::to_string(i % 100));
    }
    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (1000.0 / elapsed) * 1000.0;

    std::cout << "Context updates (1000 inputs): " << elapsed << "ms, "
              << ops_per_sec << " updates/sec" << std::endl;

    // Context tracking should be efficient
    EXPECT_LT(elapsed, 20000.0); // < 20s for 1000 updates
}

// ============================================================================
// Workflow Benchmarks
// ============================================================================

TEST_F(CliBenchmark, Workflow_CompleteConversation) {
    BenchmarkTimer timer;

    // Simulate a complete conversation workflow
    auto inputs = GenerateTestInputs(20);
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }

    // Query stats
    cli_->ProcessCommand("/stats");

    // Make some predictions
    cli_->ProcessCommand("/predict machine");
    cli_->ProcessCommand("/predict neural");

    // Enable attention
    cli_->ProcessCommand("/attention");

    // More predictions with attention
    cli_->ProcessCommand("/predict machine");
    cli_->ProcessCommand("/predict-detailed neural");
    cli_->ProcessCommand("/compare deep");

    // Check attention info
    cli_->ProcessCommand("/attention-info");

    double elapsed = timer.ElapsedMs();

    std::cout << "Complete conversation workflow: " << elapsed << "ms" << std::endl;

    EXPECT_LT(elapsed, 3000.0); // Full workflow should complete in < 3s
}

TEST_F(CliBenchmark, Workflow_AttentionIntensive) {
    BenchmarkTimer timer;

    // Build conversation history
    auto inputs = GenerateTestInputs(100);
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }

    // Enable attention
    cli_->ProcessCommand("/attention");

    // Run attention-intensive operations
    for (size_t i = 0; i < 20; ++i) {
        cli_->ProcessCommand("/predict machine");
        cli_->ProcessCommand("/predict-detailed neural");
        cli_->ProcessCommand("/compare deep");
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (60.0 / elapsed) * 1000.0; // 60 operations total

    std::cout << "Attention-intensive workflow (60 ops): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 10000.0); // Should complete in < 10s
}

TEST_F(CliBenchmark, Workflow_MixedMode) {
    BenchmarkTimer timer;

    // Build initial history
    auto inputs = GenerateTestInputs(50);
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }

    // Toggle attention on/off and make predictions
    for (size_t i = 0; i < 10; ++i) {
        cli_->ProcessCommand("/attention"); // Toggle on
        cli_->ProcessCommand("/predict machine");
        cli_->ProcessCommand("/attention"); // Toggle off
        cli_->ProcessCommand("/predict machine");
    }

    double elapsed = timer.ElapsedMs();

    std::cout << "Mixed mode workflow (20 predictions + 20 toggles): "
              << elapsed << "ms" << std::endl;

    EXPECT_LT(elapsed, 5000.0); // Should complete in < 5s
}

// ============================================================================
// Scalability Benchmarks
// ============================================================================

TEST_F(CliBenchmark, Scalability_SmallVocabulary) {
    auto inputs = GenerateTestInputs(10);

    BenchmarkTimer timer;
    // Process each input 10 times (100 total, small vocabulary)
    for (size_t i = 0; i < 10; ++i) {
        for (const auto& input : inputs) {
            cli_->ProcessCommand(input);
        }
    }
    double elapsed = timer.ElapsedMs();

    std::cout << "Small vocabulary (10 patterns, 100 inputs): "
              << elapsed << "ms" << std::endl;

    EXPECT_LT(elapsed, 2000.0);
}

TEST_F(CliBenchmark, Scalability_LargeVocabulary) {
    // Generate many unique inputs
    std::vector<std::string> inputs;
    for (size_t i = 0; i < 100; ++i) {
        inputs.push_back("unique_pattern_" + std::to_string(i));
    }

    BenchmarkTimer timer;
    for (const auto& input : inputs) {
        cli_->ProcessCommand(input);
    }
    double elapsed = timer.ElapsedMs();

    std::cout << "Large vocabulary (100 unique patterns): "
              << elapsed << "ms" << std::endl;

    EXPECT_LT(elapsed, 5000.0);
}

// ============================================================================
// Memory and Resource Benchmarks
// ============================================================================

TEST_F(CliBenchmark, MemoryGrowth_LongConversation) {
    BenchmarkTimer timer;

    // Process a very long conversation
    for (size_t i = 0; i < 500; ++i) {
        cli_->ProcessCommand("test input " + std::to_string(i % 50));
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (500.0 / elapsed) * 1000.0;

    std::cout << "Long conversation (500 inputs): " << elapsed << "ms, "
              << ops_per_sec << " inputs/sec" << std::endl;

    // Verify conversation length
    EXPECT_EQ(500u, cli_->GetConversationLength());

    // Should still be reasonably fast
    EXPECT_LT(elapsed, 30000.0); // < 30s
}
