// File: src/nn/neural_encoder.hpp
//
// Learned feature extraction for DPAN patterns.
//
// NeuralEncoder maps feature vectors to embeddings with a trained network.
// Encode patterns before storing them (PatternEngine::CreatePattern) to keep
// learned representations, or compare raw patterns in embedding space with
// EmbeddingSimilarity. Autoencoder trains an encoder without labels.
#pragma once

#include "core/pattern_data.hpp"
#include "nn/sequential.hpp"
#include "similarity/similarity_metric.hpp"

#include <memory>
#include <string>
#include <vector>

namespace dpan {
namespace nn {

/// Wraps a trained network as a feature transform (thread-safe, read-only)
class NeuralEncoder {
public:
    /// @throws std::invalid_argument if the model contains no Dense layer
    explicit NeuralEncoder(Sequential model);

    /// Number of input features (first Dense layer)
    size_t InputDimension() const { return input_dimension_; }

    /// Embedding size (last Dense layer)
    size_t OutputDimension() const { return output_dimension_; }

    /// @throws std::invalid_argument if the feature count != InputDimension()
    FeatureVector Encode(const FeatureVector& features) const;

    /// Encodes the pattern's features; keeps its modality
    PatternData Encode(const PatternData& data) const;

    /// One sample per row
    Matrix EncodeBatch(const Matrix& features) const;

    const Sequential& Model() const { return *model_; }

    bool Save(const std::string& path) const { return model_->Save(path); }
    static NeuralEncoder Load(const std::string& path);

private:
    std::shared_ptr<const Sequential> model_;
    size_t input_dimension_{0};
    size_t output_dimension_{0};
};

/// Matrix with one feature vector per row (all must have the same dimension)
Matrix FeatureMatrix(const std::vector<FeatureVector>& features);

/// Dense autoencoder: input -> hidden... -> embedding -> hidden... -> input,
/// trained to reconstruct its input (mean squared error, Adam)
class Autoencoder {
public:
    enum class Activation { TANH, RELU, SIGMOID };

    struct Config {
        /// Encoder hidden layer sizes (the decoder mirrors them)
        std::vector<size_t> hidden_sizes{32};
        size_t embedding_size{8};
        Activation activation{Activation::TANH};
        float learning_rate{0.005f};
        size_t epochs{200};
        size_t batch_size{32};
        float validation_split{0.0f};
        size_t early_stopping_patience{0};
        uint32_t seed{42};
    };

    explicit Autoencoder(size_t input_size);
    Autoencoder(size_t input_size, const Config& config);

    /// Train on one sample per row; continues from the current weights
    Sequential::History Train(const Matrix& data);
    Sequential::History Train(const std::vector<FeatureVector>& data);

    /// Mean reconstruction loss (MeanSquaredError::Value)
    float ReconstructionError(const Matrix& data) const;

    Matrix Reconstruct(const Matrix& data) const;

    /// Copy of the trained encoder half
    NeuralEncoder Encoder() const;

    const Sequential& Model() const { return model_; }
    const Config& GetConfig() const { return config_; }

private:
    Config config_;
    Sequential model_;
    size_t encoder_layer_count_{0};
    Adam optimizer_;
};

/// Similarity of raw feature vectors compared in embedding space:
/// max(0, cosine(encode(a), encode(b))). Inputs whose dimension differs from
/// the encoder's input have similarity 0.
class EmbeddingSimilarity : public SimilarityMetric {
public:
    explicit EmbeddingSimilarity(NeuralEncoder encoder);

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::vector<float> ComputeBatchFromFeatures(
        const FeatureVector& query,
        const std::vector<FeatureVector>& candidates) const override;
    std::string GetName() const override { return "Embedding"; }
    bool IsSymmetric() const override { return true; }

    const NeuralEncoder& Encoder() const { return encoder_; }

private:
    NeuralEncoder encoder_;
};

}  // namespace nn
}  // namespace dpan
