// File: tests/association/association_matrix_test.cpp
#include "association/association_matrix.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

namespace dpan {
namespace {

// ============================================================================
// Basic Add/Retrieve Tests
// ============================================================================

TEST(AssociationMatrixTest, DefaultConstruction) {
    AssociationMatrix matrix;
    EXPECT_EQ(0u, matrix.GetAssociationCount());
    EXPECT_EQ(0u, matrix.GetPatternCount());
}

TEST(AssociationMatrixTest, AddAndRetrieveSingle) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.8f);

    EXPECT_TRUE(matrix.AddAssociation(edge));
    EXPECT_EQ(1u, matrix.GetAssociationCount());
    EXPECT_EQ(2u, matrix.GetPatternCount());

    auto retrieved = matrix.GetAssociation(p1, p2);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(p1, retrieved->GetSource());
    EXPECT_EQ(p2, retrieved->GetTarget());
    EXPECT_FLOAT_EQ(0.8f, retrieved->GetStrength());
}

TEST(AssociationMatrixTest, CannotAddDuplicate) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge1(p1, p2, AssociationType::SPATIAL, 0.5f);
    AssociationEdge edge2(p1, p2, AssociationType::SPATIAL, 0.7f);

    EXPECT_TRUE(matrix.AddAssociation(edge1));
    EXPECT_FALSE(matrix.AddAssociation(edge2));  // Duplicate
    EXPECT_EQ(1u, matrix.GetAssociationCount());
}

TEST(AssociationMatrixTest, HasAssociation) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    AssociationEdge edge(p1, p2, AssociationType::CATEGORICAL, 0.6f);
    matrix.AddAssociation(edge);

    EXPECT_TRUE(matrix.HasAssociation(p1, p2));
    EXPECT_FALSE(matrix.HasAssociation(p2, p1));  // Directed
    EXPECT_FALSE(matrix.HasAssociation(p1, p3));
}

// ============================================================================
// Update/Remove Tests
// ============================================================================

TEST(AssociationMatrixTest, UpdateExisting) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge1(p1, p2, AssociationType::CATEGORICAL, 0.5f);
    matrix.AddAssociation(edge1);

    AssociationEdge edge2(p1, p2, AssociationType::CATEGORICAL, 0.9f);
    EXPECT_TRUE(matrix.UpdateAssociation(p1, p2, edge2));

    auto retrieved = matrix.GetAssociation(p1, p2);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_FLOAT_EQ(0.9f, retrieved->GetStrength());
}

TEST(AssociationMatrixTest, UpdateNonExistent) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge(p1, p2, AssociationType::FUNCTIONAL, 0.7f);
    EXPECT_FALSE(matrix.UpdateAssociation(p1, p2, edge));
}

TEST(AssociationMatrixTest, RemoveAssociation) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge(p1, p2, AssociationType::FUNCTIONAL, 0.6f);
    matrix.AddAssociation(edge);

    EXPECT_TRUE(matrix.HasAssociation(p1, p2));

    EXPECT_TRUE(matrix.RemoveAssociation(p1, p2));
    EXPECT_FALSE(matrix.HasAssociation(p1, p2));
    EXPECT_EQ(0u, matrix.GetAssociationCount());
}

TEST(AssociationMatrixTest, RemoveNonExistent) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    EXPECT_FALSE(matrix.RemoveAssociation(p1, p2));
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

TEST(AssociationMatrixTest, AddMultiple) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // Add edges one by one
    EXPECT_TRUE(matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f)));
    EXPECT_TRUE(matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CATEGORICAL, 0.6f)));
    EXPECT_TRUE(matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::SPATIAL, 0.7f)));

    EXPECT_EQ(3u, matrix.GetAssociationCount());
}

TEST(AssociationMatrixTest, RemoveMultiple) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CATEGORICAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::SPATIAL, 0.7f));

    // Remove multiple one by one
    EXPECT_TRUE(matrix.RemoveAssociation(p1, p2));
    EXPECT_TRUE(matrix.RemoveAssociation(p2, p3));

    EXPECT_EQ(1u, matrix.GetAssociationCount());
    EXPECT_TRUE(matrix.HasAssociation(p1, p3));
}

// ============================================================================
// Lookup Tests
// ============================================================================

TEST(AssociationMatrixTest, GetOutgoingAssociations) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // p1 -> p2, p3, p4
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::CAUSAL, 0.7f));

    auto outgoing = matrix.GetOutgoingAssociations(p1);
    EXPECT_EQ(3u, outgoing.size());
}

TEST(AssociationMatrixTest, GetIncomingAssociations) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // p2, p3, p4 -> p1
    matrix.AddAssociation(AssociationEdge(p2, p1, AssociationType::CATEGORICAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p3, p1, AssociationType::CATEGORICAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p4, p1, AssociationType::CATEGORICAL, 0.7f));

    auto incoming = matrix.GetIncomingAssociations(p1);
    EXPECT_EQ(3u, incoming.size());
}

TEST(AssociationMatrixTest, GetAssociationsByType) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::SPATIAL, 0.7f));

    auto causal = matrix.GetAssociationsByType(AssociationType::CAUSAL);
    EXPECT_EQ(2u, causal.size());

    auto spatial = matrix.GetAssociationsByType(AssociationType::SPATIAL);
    EXPECT_EQ(1u, spatial.size());
}

TEST(AssociationMatrixTest, GetNeighbors) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p2, p1, AssociationType::CAUSAL, 0.7f));

    auto outgoing = matrix.GetNeighbors(p1, true);
    EXPECT_EQ(2u, outgoing.size());

    auto incoming = matrix.GetNeighbors(p1, false);
    EXPECT_EQ(1u, incoming.size());
}

TEST(AssociationMatrixTest, GetMutualNeighbors) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // p1 <-> p2 (mutual)
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p1, AssociationType::CAUSAL, 0.6f));

    // p1 -> p3 (not mutual)
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.7f));

    auto mutual = matrix.GetMutualNeighbors(p1);
    EXPECT_EQ(1u, mutual.size());
    if (!mutual.empty()) {
        EXPECT_EQ(p2, mutual[0]);
    }
}

// ============================================================================
// Strength Operations Tests
// ============================================================================

TEST(AssociationMatrixTest, StrengthenAssociation) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));

    EXPECT_TRUE(matrix.StrengthenAssociation(p1, p2, 0.2f));

    auto edge = matrix.GetAssociation(p1, p2);
    ASSERT_NE(edge, nullptr);
    EXPECT_FLOAT_EQ(0.7f, edge->GetStrength());
}

TEST(AssociationMatrixTest, WeakenAssociation) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.8f));

    EXPECT_TRUE(matrix.WeakenAssociation(p1, p2, 0.3f));

    auto edge = matrix.GetAssociation(p1, p2);
    ASSERT_NE(edge, nullptr);
    EXPECT_FLOAT_EQ(0.5f, edge->GetStrength());
}

TEST(AssociationMatrixTest, ApplyDecayAll) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 1.0f);
    edge1.SetDecayRate(0.01f);
    matrix.AddAssociation(edge1);

    AssociationEdge edge2(p2, p3, AssociationType::CATEGORICAL, 1.0f);
    edge2.SetDecayRate(0.01f);
    matrix.AddAssociation(edge2);

    // Apply 100 seconds of decay
    auto elapsed = std::chrono::seconds(100);
    matrix.ApplyDecayAll(elapsed);

    // s(t) = 1.0 * exp(-0.01 * 100) ≈ 0.368
    auto retrieved1 = matrix.GetAssociation(p1, p2);
    ASSERT_NE(retrieved1, nullptr);
    EXPECT_NEAR(0.368f, retrieved1->GetStrength(), 0.01f);

    auto retrieved2 = matrix.GetAssociation(p2, p3);
    ASSERT_NE(retrieved2, nullptr);
    EXPECT_NEAR(0.368f, retrieved2->GetStrength(), 0.01f);
}

TEST(AssociationMatrixTest, ApplyDecayPattern) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 1.0f);
    edge1.SetDecayRate(0.01f);
    matrix.AddAssociation(edge1);

    AssociationEdge edge2(p2, p3, AssociationType::CATEGORICAL, 1.0f);
    edge2.SetDecayRate(0.01f);
    matrix.AddAssociation(edge2);

    // Apply decay only to p1's edges
    auto elapsed = std::chrono::seconds(100);
    matrix.ApplyDecayPattern(p1, elapsed);

    // p1 -> p2 should decay
    auto retrieved1 = matrix.GetAssociation(p1, p2);
    ASSERT_NE(retrieved1, nullptr);
    EXPECT_NEAR(0.368f, retrieved1->GetStrength(), 0.01f);

    // p2 -> p3 should not decay
    auto retrieved2 = matrix.GetAssociation(p2, p3);
    ASSERT_NE(retrieved2, nullptr);
    EXPECT_FLOAT_EQ(1.0f, retrieved2->GetStrength());
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(AssociationMatrixTest, GetAverageDegree) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // p1 -> p2, p3 (degree 2)
    // p2 -> p3 (degree 1)
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.7f));

    // Average: (2 + 1) / 2 = 1.5
    EXPECT_FLOAT_EQ(1.5f, matrix.GetAverageDegree());
}

TEST(AssociationMatrixTest, GetAverageStrength) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.4f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.8f));

    // Average: (0.4 + 0.6 + 0.8) / 3 = 0.6
    EXPECT_FLOAT_EQ(0.6f, matrix.GetAverageStrength());
}

TEST(AssociationMatrixTest, GetDensity) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // 3 patterns, 3 edges
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.7f));

    // Possible edges: 3 * (3-1) = 6
    // Density: 3/6 = 0.5
    EXPECT_FLOAT_EQ(0.5f, matrix.GetDensity());
}

TEST(AssociationMatrixTest, GetDegree) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p2, p1, AssociationType::CAUSAL, 0.7f));

    EXPECT_EQ(2u, matrix.GetDegree(p1, true));   // Outgoing
    EXPECT_EQ(1u, matrix.GetDegree(p1, false));  // Incoming
}

// ============================================================================
// Activation Propagation Tests
// ============================================================================

TEST(AssociationMatrixTest, PropagateActivation) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // p1 -> p2 (0.8), p2 -> p3 (0.6)
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.8f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.6f));

    auto results = matrix.PropagateActivation(p1, 1.0f, 3, 0.01f);

    // Should activate p2 (1.0 * 0.8 = 0.8) and p3 (0.8 * 0.6 = 0.48)
    EXPECT_EQ(2u, results.size());

    if (results.size() >= 2) {
        EXPECT_EQ(p2, results[0].pattern);
        EXPECT_FLOAT_EQ(0.8f, results[0].activation);

        EXPECT_EQ(p3, results[1].pattern);
        EXPECT_FLOAT_EQ(0.48f, results[1].activation);
    }
}

TEST(AssociationMatrixTest, PropagateActivationMaxHops) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // p1 -> p2 -> p3 -> p4 (chain)
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.9f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.9f));
    matrix.AddAssociation(AssociationEdge(p3, p4, AssociationType::CAUSAL, 0.9f));

    // With max_hops=1, should only reach p2
    auto results = matrix.PropagateActivation(p1, 1.0f, 1, 0.01f);
    EXPECT_EQ(1u, results.size());

    // With max_hops=3, should reach all
    results = matrix.PropagateActivation(p1, 1.0f, 3, 0.01f);
    EXPECT_EQ(3u, results.size());
}

TEST(AssociationMatrixTest, PropagateActivationMinThreshold) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // Weak chain
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.05f));

    // p3 activation: 1.0 * 0.5 * 0.05 = 0.025
    // With min_activation=0.1, p3 should not be included
    auto results = matrix.PropagateActivation(p1, 1.0f, 3, 0.1f);
    EXPECT_EQ(1u, results.size());
    EXPECT_EQ(p2, results[0].pattern);
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST(AssociationMatrixTest, SerializationRoundTrip) {
    AssociationMatrix original;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    original.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    original.AddAssociation(AssociationEdge(p2, p3, AssociationType::CATEGORICAL, 0.6f));
    original.AddAssociation(AssociationEdge(p1, p3, AssociationType::SPATIAL, 0.7f));

    // Serialize
    std::stringstream ss;
    original.Serialize(ss);

    // Deserialize
    auto deserialized = AssociationMatrix::Deserialize(ss);
    ASSERT_NE(deserialized, nullptr);

    // Verify
    EXPECT_EQ(original.GetAssociationCount(), deserialized->GetAssociationCount());
    EXPECT_EQ(original.GetPatternCount(), deserialized->GetPatternCount());

    EXPECT_TRUE(deserialized->HasAssociation(p1, p2));
    EXPECT_TRUE(deserialized->HasAssociation(p2, p3));
    EXPECT_TRUE(deserialized->HasAssociation(p1, p3));
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST(AssociationMatrixTest, CompactRemovesDeleted) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // Add 4 associations
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CATEGORICAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p3, p4, AssociationType::SPATIAL, 0.7f));
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::FUNCTIONAL, 0.8f));

    // Remove 2 associations
    matrix.RemoveAssociation(p2, p3);
    matrix.RemoveAssociation(p3, p4);

    // Compact
    matrix.Compact();

    // Verify remaining associations still work
    EXPECT_EQ(2u, matrix.GetAssociationCount());
    EXPECT_TRUE(matrix.HasAssociation(p1, p2));
    EXPECT_TRUE(matrix.HasAssociation(p1, p4));
    EXPECT_FALSE(matrix.HasAssociation(p2, p3));
    EXPECT_FALSE(matrix.HasAssociation(p3, p4));
}

TEST(AssociationMatrixTest, Clear) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    EXPECT_EQ(1u, matrix.GetAssociationCount());

    matrix.Clear();
    EXPECT_EQ(0u, matrix.GetAssociationCount());
    EXPECT_FALSE(matrix.HasAssociation(p1, p2));
}

TEST(AssociationMatrixTest, EstimateMemoryUsage) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    size_t empty_size = matrix.EstimateMemoryUsage();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));

    size_t with_one = matrix.EstimateMemoryUsage();
    EXPECT_GT(with_one, empty_size);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(AssociationMatrixTest, ThreadSafeConcurrentReads) {
    AssociationMatrix matrix;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 10; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Add associations
    for (size_t i = 0; i < patterns.size() - 1; ++i) {
        matrix.AddAssociation(AssociationEdge(
            patterns[i], patterns[i + 1],
            AssociationType::CAUSAL, 0.5f
        ));
    }

    // Concurrent reads
    constexpr int kNumThreads = 10;
    constexpr int kReadsPerThread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&matrix, &patterns]() {
            for (int i = 0; i < kReadsPerThread; ++i) {
                size_t idx = i % (patterns.size() - 1);
                auto edge = matrix.GetAssociation(patterns[idx], patterns[idx + 1]);
                EXPECT_NE(edge, nullptr);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(9u, matrix.GetAssociationCount());
}

TEST(AssociationMatrixTest, ThreadSafeConcurrentWrites) {
    AssociationMatrix matrix;

    constexpr int kNumThreads = 10;
    constexpr int kWritesPerThread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&matrix, t]() {
            for (int i = 0; i < kWritesPerThread; ++i) {
                PatternID p1 = PatternID::Generate();
                PatternID p2 = PatternID::Generate();
                matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(kNumThreads * kWritesPerThread, matrix.GetAssociationCount());
}

// ============================================================================
// Debugging Tests
// ============================================================================

TEST(AssociationMatrixTest, PrintStatistics) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));

    std::ostringstream oss;
    matrix.PrintStatistics(oss);

    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(std::string::npos, output.find("Association Count"));
}

TEST(AssociationMatrixTest, ToString) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));

    std::string str = matrix.ToString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(std::string::npos, str.find("AssociationMatrix"));
}

} // namespace
} // namespace dpan
