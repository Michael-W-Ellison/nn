// File: src/nn/neural_encoder.cpp
#include "nn/neural_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dpan {
namespace nn {

namespace {

std::unique_ptr<Layer> MakeActivation(Autoencoder::Activation activation) {
    switch (activation) {
        case Autoencoder::Activation::RELU:
            return std::make_unique<ReLU>();
        case Autoencoder::Activation::SIGMOID:
            return std::make_unique<Sigmoid>();
        case Autoencoder::Activation::TANH:
            break;
    }
    return std::make_unique<Tanh>();
}

Initialization InitFor(Autoencoder::Activation activation) {
    return activation == Autoencoder::Activation::RELU ? Initialization::HE : Initialization::XAVIER;
}

float CosineOfEmbeddings(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0f;
    }
    return std::clamp(static_cast<float>(dot / std::sqrt(norm_a * norm_b)), 0.0f, 1.0f);
}

}  // namespace

// ============================================================================
// NeuralEncoder
// ============================================================================

NeuralEncoder::NeuralEncoder(Sequential model)
    : model_(std::make_shared<const Sequential>(std::move(model))) {
    for (size_t i = 0; i < model_->LayerCount(); ++i) {
        if (const auto* dense = dynamic_cast<const Dense*>(&model_->GetLayer(i))) {
            if (input_dimension_ == 0) {
                input_dimension_ = dense->InputSize();
            }
            output_dimension_ = dense->OutputSize();
        }
    }
    if (input_dimension_ == 0) {
        throw std::invalid_argument("NeuralEncoder: model has no Dense layer");
    }
}

FeatureVector NeuralEncoder::Encode(const FeatureVector& features) const {
    if (features.Dimension() != input_dimension_) {
        throw std::invalid_argument("NeuralEncoder: expected " + std::to_string(input_dimension_) +
                                    " features, got " + std::to_string(features.Dimension()));
    }
    return FeatureVector(model_->Predict(features.Data()));
}

PatternData NeuralEncoder::Encode(const PatternData& data) const {
    return PatternData::FromFeatures(Encode(data.GetFeatures()), data.GetModality());
}

Matrix NeuralEncoder::EncodeBatch(const Matrix& features) const {
    return model_->Predict(features);
}

NeuralEncoder NeuralEncoder::Load(const std::string& path) {
    return NeuralEncoder(Sequential::Load(path));
}

Matrix FeatureMatrix(const std::vector<FeatureVector>& features) {
    std::vector<std::vector<float>> rows;
    rows.reserve(features.size());
    for (const auto& f : features) {
        rows.push_back(f.Data());
    }
    return Matrix::FromRows(rows);
}

// ============================================================================
// Autoencoder
// ============================================================================

Autoencoder::Autoencoder(size_t input_size) : Autoencoder(input_size, Config{}) {}

Autoencoder::Autoencoder(size_t input_size, const Config& config)
    : config_(config), optimizer_(config.learning_rate) {

    if (input_size == 0 || config.embedding_size == 0) {
        throw std::invalid_argument("Autoencoder: sizes must be positive");
    }

    uint32_t seed = config.seed;
    Initialization init = InitFor(config.activation);

    // Encoder: hidden layers with activation, linear embedding
    size_t previous = input_size;
    for (size_t hidden : config.hidden_sizes) {
        model_.Emplace<Dense>(previous, hidden, init, seed++);
        model_.Add(MakeActivation(config.activation));
        previous = hidden;
    }
    model_.Emplace<Dense>(previous, config.embedding_size, Initialization::XAVIER, seed++);
    encoder_layer_count_ = model_.LayerCount();

    // Decoder mirrors the encoder, linear reconstruction
    previous = config.embedding_size;
    for (auto it = config.hidden_sizes.rbegin(); it != config.hidden_sizes.rend(); ++it) {
        model_.Emplace<Dense>(previous, *it, init, seed++);
        model_.Add(MakeActivation(config.activation));
        previous = *it;
    }
    model_.Emplace<Dense>(previous, input_size, Initialization::XAVIER, seed++);
}

Sequential::History Autoencoder::Train(const Matrix& data) {
    Sequential::FitOptions options;
    options.epochs = config_.epochs;
    options.batch_size = config_.batch_size;
    options.validation_split = config_.validation_split;
    options.early_stopping_patience = config_.early_stopping_patience;
    options.seed = config_.seed;
    return model_.Fit(data, data, MeanSquaredError(), optimizer_, options);
}

Sequential::History Autoencoder::Train(const std::vector<FeatureVector>& data) {
    return Train(FeatureMatrix(data));
}

float Autoencoder::ReconstructionError(const Matrix& data) const {
    return model_.Evaluate(data, data, MeanSquaredError());
}

Matrix Autoencoder::Reconstruct(const Matrix& data) const {
    return model_.Predict(data);
}

NeuralEncoder Autoencoder::Encoder() const {
    return NeuralEncoder(model_.Slice(0, encoder_layer_count_));
}

// ============================================================================
// EmbeddingSimilarity
// ============================================================================

EmbeddingSimilarity::EmbeddingSimilarity(NeuralEncoder encoder) : encoder_(std::move(encoder)) {}

float EmbeddingSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float EmbeddingSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() != encoder_.InputDimension() || b.Dimension() != encoder_.InputDimension()) {
        return 0.0f;
    }
    Matrix encoded = encoder_.EncodeBatch(Matrix::FromRows({a.Data(), b.Data()}));
    return CosineOfEmbeddings(encoded.Row(0), encoded.Row(1));
}

std::vector<float> EmbeddingSimilarity::ComputeBatchFromFeatures(
    const FeatureVector& query,
    const std::vector<FeatureVector>& candidates) const {

    std::vector<float> scores(candidates.size(), 0.0f);
    if (query.Dimension() != encoder_.InputDimension()) {
        return scores;
    }

    // Encode the query once and all matching candidates in one batch
    std::vector<std::vector<float>> rows{query.Data()};
    std::vector<size_t> positions;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].Dimension() == encoder_.InputDimension()) {
            rows.push_back(candidates[i].Data());
            positions.push_back(i);
        }
    }
    Matrix encoded = encoder_.EncodeBatch(Matrix::FromRows(rows));
    std::vector<float> query_embedding = encoded.Row(0);
    for (size_t j = 0; j < positions.size(); ++j) {
        scores[positions[j]] = CosineOfEmbeddings(query_embedding, encoded.Row(j + 1));
    }
    return scores;
}

}  // namespace nn
}  // namespace dpan
