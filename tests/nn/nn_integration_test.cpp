// File: tests/nn/nn_integration_test.cpp
//
// Tests for the neural network integration with DPAN: learned encoders,
// embedding similarity in PatternEngine, and neural prediction scoring in
// AssociationLearningSystem.
#include "association/association_learning_system.hpp"
#include "core/pattern_engine.hpp"
#include "nn/neural_encoder.hpp"
#include "nn/neural_prediction_scorer.hpp"
#include "storage/memory_backend.hpp"

#include <gtest/gtest.h>
#include "../test_temp_path.hpp"

#include <atomic>
#include <filesystem>
#include <random>
#include <thread>

namespace dpan {
namespace nn {
namespace {

/// 8-dimensional points generated from 2 latent values (a 2-D manifold)
std::vector<FeatureVector> ManifoldData(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> latent(-1.0f, 1.0f);
    std::vector<FeatureVector> data;
    for (size_t i = 0; i < count; ++i) {
        float u = latent(rng);
        float v = latent(rng);
        data.emplace_back(std::vector<float>{u, v, u + v, u - v, 0.5f * u, -v, u * 0.8f + 0.2f * v, 0.3f * v});
    }
    return data;
}

/// Two well-separated clusters in 8 dimensions
std::vector<FeatureVector> ClusterData(size_t per_cluster, uint32_t seed, std::vector<int>& labels) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::vector<FeatureVector> data;
    for (int cluster = 0; cluster < 2; ++cluster) {
        float sign = cluster == 0 ? 1.0f : -1.0f;
        for (size_t i = 0; i < per_cluster; ++i) {
            std::vector<float> values(8);
            for (size_t d = 0; d < 8; ++d) {
                float center = (d % 2 == 0 ? sign : -sign) * 0.8f;
                values[d] = center + noise(rng);
            }
            data.emplace_back(values);
            labels.push_back(cluster);
        }
    }
    return data;
}

PatternID StorePattern(PatternDatabase& db, const std::vector<float>& features) {
    PatternNode node(PatternID::Generate(),
                     PatternData::FromFeatures(FeatureVector(features), DataModality::NUMERIC),
                     PatternType::ATOMIC);
    PatternID id = node.GetID();
    db.Store(node);
    return id;
}

// ============================================================================
// NeuralEncoder / Autoencoder
// ============================================================================

TEST(NeuralEncoderTest, EncodesFeaturesAndPatterns) {
    Sequential model;
    model.Emplace<Dense>(4, 6, Initialization::XAVIER, 1).Emplace<Tanh>().Emplace<Dense>(6, 3, Initialization::XAVIER, 2);
    NeuralEncoder encoder(model);

    EXPECT_EQ(4u, encoder.InputDimension());
    EXPECT_EQ(3u, encoder.OutputDimension());

    FeatureVector features(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f});
    FeatureVector encoded = encoder.Encode(features);
    EXPECT_EQ(3u, encoded.Dimension());
    EXPECT_EQ(model.Predict(features.Data()), encoded.Data());

    PatternData data = PatternData::FromFeatures(features, DataModality::AUDIO);
    PatternData encoded_data = encoder.Encode(data);
    EXPECT_EQ(DataModality::AUDIO, encoded_data.GetModality());
    EXPECT_EQ(3u, encoded_data.GetFeatures().Dimension());

    EXPECT_THROW(encoder.Encode(FeatureVector(std::vector<float>{1.0f})), std::invalid_argument);

    Sequential no_dense;
    no_dense.Emplace<Tanh>();
    EXPECT_THROW(NeuralEncoder{no_dense}, std::invalid_argument);
}

TEST(NeuralEncoderTest, SaveAndLoad) {
    Sequential model;
    model.Emplace<Dense>(3, 2, Initialization::XAVIER, 9);
    NeuralEncoder encoder(model);

    std::string path = test::UniqueTempPath("dpan_encoder") + ".bin";
    ASSERT_TRUE(encoder.Save(path));
    NeuralEncoder loaded = NeuralEncoder::Load(path);
    std::filesystem::remove(path);

    FeatureVector features(std::vector<float>{1.0f, -1.0f, 0.5f});
    EXPECT_EQ(encoder.Encode(features).Data(), loaded.Encode(features).Data());
}

TEST(AutoencoderTest, LearnsLowDimensionalStructure) {
    auto data = ManifoldData(400, 1);
    Matrix matrix = FeatureMatrix(data);

    Autoencoder::Config config;
    config.hidden_sizes = {16};
    config.embedding_size = 2;
    config.epochs = 150;
    config.learning_rate = 0.01f;
    Autoencoder autoencoder(8, config);

    float before = autoencoder.ReconstructionError(matrix);
    auto history = autoencoder.Train(matrix);
    float after = autoencoder.ReconstructionError(matrix);

    EXPECT_EQ(150u, history.epochs_run);
    EXPECT_LT(after, before * 0.05f);  // At least 20x better than untrained

    NeuralEncoder encoder = autoencoder.Encoder();
    EXPECT_EQ(8u, encoder.InputDimension());
    EXPECT_EQ(2u, encoder.OutputDimension());
    EXPECT_EQ(2u, encoder.Encode(data[0]).Dimension());

    // Generalizes to unseen points from the same manifold
    EXPECT_LT(autoencoder.ReconstructionError(FeatureMatrix(ManifoldData(100, 2))), before * 0.05f);
}

// ============================================================================
// EmbeddingSimilarity
// ============================================================================

TEST(EmbeddingSimilarityTest, ComparesInEmbeddingSpace) {
    std::vector<int> labels;
    auto data = ClusterData(100, 3, labels);
    Autoencoder::Config config;
    config.hidden_sizes = {8};
    config.embedding_size = 2;
    config.epochs = 100;
    Autoencoder autoencoder(8, config);
    autoencoder.Train(data);

    EmbeddingSimilarity similarity(autoencoder.Encoder());
    EXPECT_EQ("Embedding", similarity.GetName());

    // Same cluster: more similar than across clusters
    float same = similarity.ComputeFromFeatures(data[0], data[1]);
    float different = similarity.ComputeFromFeatures(data[0], data[150]);
    EXPECT_GT(same, different);
    EXPECT_NEAR(1.0f, similarity.ComputeFromFeatures(data[0], data[0]), 1e-5f);

    // Batch matches individual computation; wrong dimensions give 0
    std::vector<FeatureVector> candidates{data[1], data[150], FeatureVector(std::vector<float>{1.0f})};
    auto batch = similarity.ComputeBatchFromFeatures(data[0], candidates);
    ASSERT_EQ(3u, batch.size());
    EXPECT_NEAR(same, batch[0], 1e-5f);
    EXPECT_NEAR(different, batch[1], 1e-5f);
    EXPECT_EQ(0.0f, batch[2]);
}

TEST(EmbeddingSimilarityTest, PatternEngineSearchUsesCustomMetric) {
    std::vector<int> labels;
    auto data = ClusterData(30, 4, labels);
    Autoencoder::Config config;
    config.hidden_sizes = {8};
    config.embedding_size = 2;
    config.epochs = 150;
    Autoencoder autoencoder(8, config);
    autoencoder.Train(data);

    PatternEngine::Config engine_config;
    engine_config.similarity_metric_instance =
        std::make_shared<EmbeddingSimilarity>(autoencoder.Encoder());
    PatternEngine engine(engine_config);

    std::unordered_map<PatternID, int> cluster_of;
    for (size_t i = 0; i < data.size(); ++i) {
        PatternID id = engine.CreatePattern(PatternData::FromFeatures(data[i], DataModality::NUMERIC));
        cluster_of[id] = labels[i];
    }

    auto results = engine.FindSimilarPatterns(PatternData::FromFeatures(data[0], DataModality::NUMERIC), 10);
    ASSERT_EQ(10u, results.size());
    for (const auto& result : results) {
        EXPECT_EQ(0, cluster_of[result.pattern_id]);
    }
}

// ============================================================================
// NeuralPredictionScorer
// ============================================================================

class NeuralPredictionScorerTest : public ::testing::Test {
protected:
    void SetUp() override {
        db_ = std::make_unique<MemoryBackend>(MemoryBackend::Config{});
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> value(-1.0f, 1.0f);
        for (int i = 0; i < 40; ++i) {
            ids_.push_back(StorePattern(*db_, {value(rng), value(rng), value(rng)}));
        }
    }

    /// Ground truth: a candidate follows a source when the candidate's first
    /// feature has the same sign as the source's first feature
    bool Follows(PatternID source, PatternID candidate) const {
        float s = db_->Retrieve(source)->GetData().GetFeatures()[0];
        float c = db_->Retrieve(candidate)->GetData().GetFeatures()[0];
        return (s > 0.0f) == (c > 0.0f);
    }

    void AddAllPairs(NeuralPredictionScorer& scorer) {
        for (size_t i = 0; i < ids_.size(); ++i) {
            for (size_t j = 0; j < ids_.size(); ++j) {
                if (i != j) {
                    ASSERT_TRUE(scorer.AddExample(ids_[i], ids_[j], 0.5f, Follows(ids_[i], ids_[j])));
                }
            }
        }
    }

    std::unique_ptr<PatternDatabase> db_;
    std::vector<PatternID> ids_;
};

TEST_F(NeuralPredictionScorerTest, UntrainedScorerReturnsStrength) {
    NeuralPredictionScorer scorer(db_.get(), 3);
    EXPECT_FALSE(scorer.IsTrained());
    EXPECT_FLOAT_EQ(0.37f, scorer.Score(ids_[0], ids_[1], 0.37f, nullptr));
    EXPECT_THROW(scorer.Train(), std::logic_error);

    // Unknown patterns cannot be used as examples
    EXPECT_FALSE(scorer.AddExample(ids_[0], PatternID::Generate(), 0.5f, true));
    EXPECT_EQ(0u, scorer.ExampleCount());

    EXPECT_THROW(NeuralPredictionScorer(nullptr, 3), std::invalid_argument);
    EXPECT_THROW(NeuralPredictionScorer(db_.get(), 0), std::invalid_argument);
}

TEST_F(NeuralPredictionScorerTest, LearnsWhichCandidatesFollow) {
    NeuralPredictionScorer::Config config;
    config.epochs = 60;
    NeuralPredictionScorer scorer(db_.get(), 3, config);
    AddAllPairs(scorer);
    EXPECT_EQ(40u * 39u, scorer.ExampleCount());

    auto history = scorer.Train();
    EXPECT_LT(history.train_loss.back(), history.train_loss.front());
    EXPECT_TRUE(scorer.IsTrained());

    size_t correct = 0, total = 0;
    for (size_t i = 0; i < ids_.size(); ++i) {
        for (size_t j = 0; j < ids_.size(); ++j) {
            if (i == j) {
                continue;
            }
            bool predicted = scorer.Score(ids_[i], ids_[j], 0.5f, nullptr) >= 0.5f;
            correct += predicted == Follows(ids_[i], ids_[j]) ? 1 : 0;
            total++;
        }
    }
    EXPECT_GT(static_cast<float>(correct) / total, 0.9f);
}

TEST_F(NeuralPredictionScorerTest, ReranksAssociationPredictions) {
    NeuralPredictionScorer::Config config;
    config.epochs = 60;
    auto scorer = std::make_shared<NeuralPredictionScorer>(db_.get(), 3, config);
    AddAllPairs(*scorer);
    scorer->Train();

    // Source with one true follower and one false follower, equal strengths
    PatternID source = ids_[0];
    PatternID good, bad;
    bool have_good = false, have_bad = false;
    for (size_t j = 1; j < ids_.size() && !(have_good && have_bad); ++j) {
        if (Follows(source, ids_[j]) && !have_good) {
            good = ids_[j];
            have_good = true;
        } else if (!Follows(source, ids_[j]) && !have_bad) {
            bad = ids_[j];
            have_bad = true;
        }
    }
    ASSERT_TRUE(have_good && have_bad);

    AssociationLearningSystem system;
    auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());
    matrix.AddAssociation(AssociationEdge(source, bad, AssociationType::CAUSAL, 0.55f));
    matrix.AddAssociation(AssociationEdge(source, good, AssociationType::CAUSAL, 0.5f));

    // Without the scorer the stronger (wrong) association wins
    EXPECT_EQ(bad, system.Predict(source, 1)[0]);

    system.SetPredictionScorer(scorer, 0.7f);
    EXPECT_EQ(scorer, system.GetPredictionScorer());
    EXPECT_EQ(good, system.Predict(source, 1)[0]);
    auto ranked = system.PredictWithConfidence(source, 2);
    ASSERT_EQ(2u, ranked.size());
    EXPECT_EQ(good, ranked[0].first);
    EXPECT_GT(ranked[0].second, ranked[1].second);

    EXPECT_THROW(system.SetPredictionScorer(scorer, 1.5f), std::invalid_argument);
    system.SetPredictionScorer(nullptr);
    EXPECT_EQ(bad, system.Predict(source, 1)[0]);
}

TEST_F(NeuralPredictionScorerTest, EncoderAndModelPersistence) {
    Sequential encoder_model;
    encoder_model.Emplace<Dense>(3, 2, Initialization::XAVIER, 3).Emplace<Tanh>();
    auto encoder = std::make_shared<NeuralEncoder>(encoder_model);

    NeuralPredictionScorer::Config config;
    config.epochs = 5;
    NeuralPredictionScorer scorer(db_.get(), 3, config, encoder);
    EXPECT_EQ(2u * 2u + 1u, scorer.InputDimension());
    AddAllPairs(scorer);
    scorer.Train();

    std::string path = test::UniqueTempPath("dpan_scorer") + ".bin";
    ASSERT_TRUE(scorer.SaveModel(path));

    NeuralPredictionScorer restored(db_.get(), 3, config, encoder);
    restored.LoadModel(path);
    EXPECT_FLOAT_EQ(scorer.Score(ids_[0], ids_[1], 0.5f, nullptr),
                    restored.Score(ids_[0], ids_[1], 0.5f, nullptr));

    // A model with a different input size is rejected
    NeuralPredictionScorer without_encoder(db_.get(), 3, config);
    EXPECT_THROW(without_encoder.LoadModel(path), std::runtime_error);
    std::filesystem::remove(path);

    Sequential wrong_encoder;
    wrong_encoder.Emplace<Dense>(5, 2);
    EXPECT_THROW(NeuralPredictionScorer(db_.get(), 3, config,
                                        std::make_shared<NeuralEncoder>(wrong_encoder)),
                 std::invalid_argument);
}

TEST_F(NeuralPredictionScorerTest, ScoringDuringRetrainingIsSafe) {
    NeuralPredictionScorer::Config config;
    config.epochs = 20;
    NeuralPredictionScorer scorer(db_.get(), 3, config);
    AddAllPairs(scorer);
    scorer.Train();

    std::atomic<bool> done{false};
    std::thread trainer([&]() {
        for (int i = 0; i < 3; ++i) {
            scorer.Train();
        }
        done = true;
    });
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t]() {
            while (!done) {
                float score = scorer.Score(ids_[t], ids_[t + 1], 0.5f, nullptr);
                EXPECT_GE(score, 0.0f);
                EXPECT_LE(score, 1.0f);
            }
        });
    }
    trainer.join();
    for (auto& reader : readers) {
        reader.join();
    }
}

}  // namespace
}  // namespace nn
}  // namespace dpan
