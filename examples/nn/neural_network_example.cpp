// File: examples/nn/neural_network_example.cpp
//
// Demonstrates the neural network module on its own and combined with DPAN:
//   1. Train a classifier with backpropagation, save and reload it
//   2. Learn an embedding with an autoencoder and search patterns in
//      embedding space through PatternEngine
//   3. Rank association predictions with a network trained on outcomes

#include "association/association_learning_system.hpp"
#include "core/pattern_engine.hpp"
#include "nn/neural_encoder.hpp"
#include "nn/neural_prediction_scorer.hpp"
#include "storage/memory_backend.hpp"

#include <cstdio>
#include <iostream>
#include <random>

using namespace dpan;
using namespace dpan::nn;

namespace {

void TrainClassifier() {
    std::cout << "1. Classifier (XOR)\n";

    Matrix x = Matrix::FromRows({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    Matrix y = Matrix::FromRows({{0}, {1}, {1}, {0}});

    Sequential model;
    model.Emplace<Dense>(2, 8).Emplace<Tanh>().Emplace<Dense>(8, 1).Emplace<Sigmoid>();

    Adam optimizer(0.05f);
    Sequential::FitOptions options;
    options.epochs = 1000;
    options.batch_size = 4;
    auto history = model.Fit(x, y, BinaryCrossEntropy(), optimizer, options);

    std::cout << model.Summary();
    std::cout << "   loss " << history.train_loss.front() << " -> " << history.train_loss.back()
              << ", accuracy " << BinaryAccuracy(model.Predict(x), y) << "\n";

    const char* path = "xor_model.dpnn";
    model.Save(path);
    Sequential restored = Sequential::Load(path);
    std::remove(path);
    std::cout << "   reloaded model predicts [1, 0] -> " << restored.Predict({1.0f, 0.0f})[0] << "\n\n";
}

void LearnedSimilaritySearch() {
    std::cout << "2. Autoencoder embeddings in PatternEngine\n";

    // Two groups of 6-dimensional readings
    std::mt19937 rng(1);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::vector<FeatureVector> readings;
    for (int i = 0; i < 100; ++i) {
        float s = i < 50 ? 1.0f : -1.0f;
        readings.emplace_back(std::vector<float>{s + noise(rng), -s + noise(rng), s + noise(rng),
                                                 noise(rng), s * 0.5f + noise(rng), -s + noise(rng)});
    }

    Autoencoder::Config config;
    config.hidden_sizes = {8};
    config.embedding_size = 2;
    config.epochs = 150;
    Autoencoder autoencoder(6, config);
    autoencoder.Train(readings);
    std::cout << "   reconstruction error " << autoencoder.ReconstructionError(FeatureMatrix(readings)) << "\n";

    PatternEngine::Config engine_config;
    engine_config.similarity_metric_instance = std::make_shared<EmbeddingSimilarity>(autoencoder.Encoder());
    PatternEngine engine(engine_config);
    for (const auto& reading : readings) {
        engine.CreatePattern(PatternData::FromFeatures(reading, DataModality::NUMERIC));
    }

    auto results = engine.FindSimilarPatterns(PatternData::FromFeatures(readings[0], DataModality::NUMERIC), 5);
    std::cout << "   top matches for reading 0:";
    for (const auto& result : results) {
        std::cout << " " << result.pattern_id.value() << " (" << result.similarity << ")";
    }
    std::cout << "\n\n";
}

void LearnedPredictionScoring() {
    std::cout << "3. Neural prediction scoring\n";

    MemoryBackend db(MemoryBackend::Config{});
    auto store = [&db](std::vector<float> features) {
        PatternNode node(PatternID::Generate(),
                         PatternData::FromFeatures(FeatureVector(std::move(features)), DataModality::NUMERIC),
                         PatternType::ATOMIC);
        PatternID id = node.GetID();
        db.Store(node);
        return id;
    };

    // "Warm" patterns are followed by warm ones, "cold" by cold ones
    std::vector<PatternID> warm, cold;
    for (int i = 0; i < 10; ++i) {
        warm.push_back(store({1.0f, 0.1f * i}));
        cold.push_back(store({-1.0f, 0.1f * i}));
    }

    auto scorer = std::make_shared<NeuralPredictionScorer>(&db, 2);
    for (size_t i = 0; i < warm.size(); ++i) {
        for (size_t j = 0; j < warm.size(); ++j) {
            scorer->AddExample(warm[i], warm[j], 0.5f, true);
            scorer->AddExample(warm[i], cold[j], 0.5f, false);
            scorer->AddExample(cold[i], cold[j], 0.5f, true);
            scorer->AddExample(cold[i], warm[j], 0.5f, false);
        }
    }
    scorer->Train();

    AssociationLearningSystem system;
    auto& matrix = system.GetAssociationMatrix();
    matrix.AddAssociation(AssociationEdge(warm[0], cold[1], AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(warm[0], warm[1], AssociationType::CAUSAL, 0.5f));

    std::cout << "   by strength only:  " << (system.Predict(warm[0], 1)[0] == warm[1] ? "warm" : "cold") << "\n";
    system.SetPredictionScorer(scorer, 0.7f);
    std::cout << "   with neural scorer: " << (system.Predict(warm[0], 1)[0] == warm[1] ? "warm" : "cold") << "\n";
}

}  // namespace

int main() {
    TrainClassifier();
    LearnedSimilaritySearch();
    LearnedPredictionScoring();
    return 0;
}
