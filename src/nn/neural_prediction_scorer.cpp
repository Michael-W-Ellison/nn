// File: src/nn/neural_prediction_scorer.cpp
#include "nn/neural_prediction_scorer.hpp"

#include <algorithm>
#include <stdexcept>

namespace dpan {
namespace nn {

NeuralPredictionScorer::NeuralPredictionScorer(PatternDatabase* db, size_t feature_dimension)
    : NeuralPredictionScorer(db, feature_dimension, Config{}) {}

NeuralPredictionScorer::NeuralPredictionScorer(PatternDatabase* db, size_t feature_dimension,
                                               const Config& config,
                                               std::shared_ptr<const NeuralEncoder> encoder)
    : db_(db), feature_dimension_(feature_dimension), config_(config), encoder_(std::move(encoder)) {
    if (!db_) {
        throw std::invalid_argument("NeuralPredictionScorer: database is null");
    }
    if (feature_dimension_ == 0) {
        throw std::invalid_argument("NeuralPredictionScorer: feature_dimension must be positive");
    }
    if (encoder_ && encoder_->InputDimension() != feature_dimension_) {
        throw std::invalid_argument("NeuralPredictionScorer: encoder input dimension does not "
                                    "match feature_dimension");
    }
}

size_t NeuralPredictionScorer::InputDimension() const {
    size_t per_pattern = encoder_ ? encoder_->OutputDimension() : feature_dimension_;
    return 2 * per_pattern + (config_.include_strength ? 1 : 0);
}

std::optional<std::vector<float>> NeuralPredictionScorer::PatternFeatures(PatternID id) const {
    auto node = db_->Retrieve(id);
    if (!node) {
        return std::nullopt;
    }
    FeatureVector features = node->GetData().GetFeatures();
    if (features.Dimension() != feature_dimension_) {
        return std::nullopt;
    }
    if (encoder_) {
        return encoder_->Encode(features).Data();
    }
    return features.Data();
}

std::optional<std::vector<float>> NeuralPredictionScorer::BuildInput(
    PatternID source, PatternID candidate, float association_strength) const {

    auto source_features = PatternFeatures(source);
    auto candidate_features = PatternFeatures(candidate);
    if (!source_features || !candidate_features) {
        return std::nullopt;
    }
    std::vector<float> input = std::move(*source_features);
    input.insert(input.end(), candidate_features->begin(), candidate_features->end());
    if (config_.include_strength) {
        input.push_back(association_strength);
    }
    return input;
}

bool NeuralPredictionScorer::AddExample(PatternID source, PatternID candidate,
                                        float association_strength, bool followed) {
    auto input = BuildInput(source, candidate, association_strength);
    if (!input) {
        return false;
    }
    std::lock_guard<std::mutex> lock(examples_mutex_);
    example_inputs_.push_back(std::move(*input));
    example_labels_.push_back(followed ? 1.0f : 0.0f);
    if (example_inputs_.size() > config_.max_examples) {
        size_t excess = example_inputs_.size() - config_.max_examples;
        example_inputs_.erase(example_inputs_.begin(), example_inputs_.begin() + excess);
        example_labels_.erase(example_labels_.begin(), example_labels_.begin() + excess);
    }
    return true;
}

size_t NeuralPredictionScorer::ExampleCount() const {
    std::lock_guard<std::mutex> lock(examples_mutex_);
    return example_inputs_.size();
}

void NeuralPredictionScorer::ClearExamples() {
    std::lock_guard<std::mutex> lock(examples_mutex_);
    example_inputs_.clear();
    example_labels_.clear();
}

Sequential NeuralPredictionScorer::BuildModel() const {
    Sequential model;
    uint32_t seed = config_.seed;
    size_t previous = InputDimension();
    for (size_t hidden : config_.hidden_sizes) {
        model.Emplace<Dense>(previous, hidden, Initialization::HE, seed++);
        model.Emplace<ReLU>();
        previous = hidden;
    }
    model.Emplace<Dense>(previous, 1, Initialization::XAVIER, seed++);
    model.Emplace<Sigmoid>();
    return model;
}

Sequential::History NeuralPredictionScorer::Train() {
    Matrix inputs;
    Matrix labels;
    {
        std::lock_guard<std::mutex> lock(examples_mutex_);
        if (example_inputs_.empty()) {
            throw std::logic_error("NeuralPredictionScorer::Train: no examples recorded");
        }
        inputs = Matrix::FromRows(example_inputs_);
        labels = Matrix(example_labels_.size(), 1, example_labels_);
    }

    Sequential model = BuildModel();
    Adam optimizer(config_.learning_rate);
    Sequential::FitOptions options;
    options.epochs = config_.epochs;
    options.batch_size = config_.batch_size;
    options.seed = config_.seed;
    auto history = model.Fit(inputs, labels, BinaryCrossEntropy(), optimizer, options);

    std::unique_lock<std::shared_mutex> lock(model_mutex_);
    model_ = std::make_shared<const Sequential>(std::move(model));
    return history;
}

bool NeuralPredictionScorer::IsTrained() const {
    std::shared_lock<std::shared_mutex> lock(model_mutex_);
    return model_ != nullptr;
}

float NeuralPredictionScorer::Score(PatternID source, PatternID candidate,
                                    float association_strength, const ContextVector*) const {
    std::shared_ptr<const Sequential> model;
    {
        std::shared_lock<std::shared_mutex> lock(model_mutex_);
        model = model_;
    }
    if (!model) {
        return association_strength;
    }
    auto input = BuildInput(source, candidate, association_strength);
    if (!input) {
        return association_strength;
    }
    return std::clamp(model->Predict(*input)[0], 0.0f, 1.0f);
}

bool NeuralPredictionScorer::SaveModel(const std::string& path) const {
    std::shared_lock<std::shared_mutex> lock(model_mutex_);
    return model_ && model_->Save(path);
}

void NeuralPredictionScorer::LoadModel(const std::string& path) {
    Sequential model = Sequential::Load(path);
    // The first Dense layer must accept this scorer's input
    for (size_t i = 0; i < model.LayerCount(); ++i) {
        if (const auto* dense = dynamic_cast<const Dense*>(&model.GetLayer(i))) {
            if (dense->InputSize() != InputDimension()) {
                throw std::runtime_error("NeuralPredictionScorer::LoadModel: model expects " +
                                         std::to_string(dense->InputSize()) + " inputs, scorer has " +
                                         std::to_string(InputDimension()));
            }
            break;
        }
    }
    std::unique_lock<std::shared_mutex> lock(model_mutex_);
    model_ = std::make_shared<const Sequential>(std::move(model));
}

}  // namespace nn
}  // namespace dpan
