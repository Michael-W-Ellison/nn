// File: src/nn/neural_prediction_scorer.hpp
//
// PredictionScorer backed by a neural network trained on prediction
// outcomes. Input per (source, candidate) pair: source features, candidate
// features (optionally encoded by a NeuralEncoder) and the association
// strength; output: probability that the candidate follows the source.
//
// Usage:
//   auto scorer = std::make_shared<NeuralPredictionScorer>(&db, feature_dim);
//   scorer->AddExample(source, actual_next, strength, true);   // per outcome
//   scorer->AddExample(source, wrong_guess, strength, false);
//   scorer->Train();
//   learning_system.SetPredictionScorer(scorer, 0.5f);
//
// Thread safety: all methods are thread-safe. Train() builds a new model and
// swaps it in, so Score() keeps working (with the previous model) meanwhile.
#pragma once

#include "association/prediction_scorer.hpp"
#include "nn/neural_encoder.hpp"
#include "storage/pattern_database.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

namespace dpan {
namespace nn {

class NeuralPredictionScorer : public PredictionScorer {
public:
    struct Config {
        std::vector<size_t> hidden_sizes{32, 16};
        float learning_rate{0.005f};
        size_t epochs{200};
        size_t batch_size{32};
        uint32_t seed{42};

        /// Feed the association strength to the network as an extra input
        bool include_strength{true};

        /// Oldest examples are dropped beyond this count
        size_t max_examples{100000};
    };

    /// @param db Pattern database used to look up pattern features (not
    ///        owned; must outlive the scorer)
    /// @param feature_dimension Feature count of stored patterns (input
    ///        dimension of `encoder` when one is given)
    /// @param encoder Optional encoder applied to pattern features
    NeuralPredictionScorer(PatternDatabase* db, size_t feature_dimension);
    NeuralPredictionScorer(PatternDatabase* db, size_t feature_dimension, const Config& config,
                           std::shared_ptr<const NeuralEncoder> encoder = nullptr);

    /// Record an outcome: whether `candidate` actually followed `source`.
    /// @return false if either pattern is missing or has the wrong dimension
    bool AddExample(PatternID source, PatternID candidate, float association_strength,
                    bool followed);

    size_t ExampleCount() const;
    void ClearExamples();

    /// Train a new model on all recorded examples (binary cross-entropy,
    /// Adam) and start using it
    /// @throws std::logic_error if there are no examples
    Sequential::History Train();

    bool IsTrained() const;

    /// Learned probability; before training, or when features are not
    /// available, returns association_strength (no effect on ranking)
    float Score(PatternID source, PatternID candidate, float association_strength,
                const ContextVector* context) const override;

    /// Persist / restore the trained network
    bool SaveModel(const std::string& path) const;
    void LoadModel(const std::string& path);

    size_t InputDimension() const;

private:
    PatternDatabase* db_;
    size_t feature_dimension_;
    Config config_;
    std::shared_ptr<const NeuralEncoder> encoder_;

    mutable std::mutex examples_mutex_;
    std::vector<std::vector<float>> example_inputs_;
    std::vector<float> example_labels_;

    mutable std::shared_mutex model_mutex_;
    std::shared_ptr<const Sequential> model_;

    std::optional<std::vector<float>> PatternFeatures(PatternID id) const;
    std::optional<std::vector<float>> BuildInput(PatternID source, PatternID candidate,
                                                 float association_strength) const;
    Sequential BuildModel() const;
};

}  // namespace nn
}  // namespace dpan
