// File: src/agent/network.hpp
//
// The agent's neural network (docs/AGENT_PLAN.md, sections 2-3):
//
//   fused vector -> channel embeddings -> trunk (2 x Dense+ReLU) -> one head per output
//
// Channels with an embedding size (hashed categories, text) are projected by
// their own learned Dense layer; other channels pass through. Each head is a
// Dense layer on the trunk output: classifiers produce logits (softmax
// cross-entropy), regressors produce values (squared error). A head is
// trained only on the samples that have a target for it.
#pragma once

#include "agent/encoders.hpp"
#include "nn/layers.hpp"
#include "nn/optimizer.hpp"
#include "nn/sequential.hpp"

#include <memory>
#include <string>
#include <vector>

namespace dpan {
namespace agent {

/// What one sample teaches one head
struct HeadTarget {
    /// Classifier: class to train toward (-1 = none)
    int label{-1};

    /// Classifier: class to train away from (-1 = none); from an incorrect
    /// OUTCOME
    int negative{-1};

    /// Regressor: target values in the head's (standardized) units; NaN
    /// elements have no target
    std::vector<float> values;

    /// Q-values (stored samples only): action taken and the reward it
    /// received; the agent turns them into a value target when it trains
    int action{-1};
    float reward{0.0f};
    bool exploratory{false};  ///< The action was chosen at random

    bool Empty() const { return label < 0 && negative < 0 && values.empty() && action < 0; }
};

/// CLASSIFIER: softmax cross-entropy. REGRESSOR: squared error.
/// Q_VALUES: one value per action, Huber loss (quadratic within 1 of the
/// target, linear beyond), trained on the taken action only.
enum class HeadKind { CLASSIFIER, REGRESSOR, Q_VALUES };

struct HeadSpec {
    std::string name;
    HeadKind kind{HeadKind::REGRESSOR};

    /// Classes (classifier) or values (regressor); a classifier may start
    /// with 0 classes and grow
    size_t size{0};
};

class AgentNetwork {
public:
    /// @param layout Channel layout of the fused vector (ObservationEncoder::Layout)
    /// @param min_width, max_width Trunk width = 2 x embedded size, clamped
    AgentNetwork(const std::vector<ChannelLayout>& layout, std::vector<HeadSpec> heads,
                 float learning_rate, uint32_t seed, size_t min_width = 32, size_t max_width = 512);

    /// Deep copy (weights and optimizer state), e.g. for a target network
    AgentNetwork(const AgentNetwork& other);
    AgentNetwork& operator=(const AgentNetwork&) = delete;

    size_t InputSize() const { return input_size_; }
    size_t EmbeddedSize() const { return embedded_size_; }
    size_t TrunkWidth() const { return trunk_width_; }
    size_t HeadCount() const { return heads_.size(); }

    /// Number of trainable values (weights and biases)
    size_t ParameterCount() const;

    /// All trainable parameters (embeddings, trunk, heads)
    std::vector<nn::Parameter> AllParameters();

    /// False if any weight or bias is NaN or infinite
    bool AllFinite();

    /// Widen both trunk layers to `width` units without changing any output
    /// (Net2WiderNet: new units copy existing ones and the copies share
    /// their outgoing weights). Optimizer state is reset.
    void Widen(size_t width);
    const HeadSpec& Head(size_t index) const { return head_specs_.at(index); }

    /// Outputs of one head for a batch (logits for classifiers); a
    /// classifier with no classes returns a matrix with 0 columns
    nn::Matrix Infer(const nn::Matrix& inputs, size_t head) const;

    /// Add classes to a classifier head (existing weights are kept, the
    /// head's optimizer state is reset)
    void GrowClasses(size_t head, size_t class_count);

    /// One optimization step. targets[h][row] (nullptr = no target);
    /// loss_scale[h] multiplies head h's loss. Gradients are clipped to
    /// clip_norm (0 = off) before the update.
    /// @return Mean loss per head over its targets (NaN for heads without any)
    std::vector<float> TrainBatch(const nn::Matrix& inputs,
                                  const std::vector<std::vector<const HeadTarget*>>& targets,
                                  const std::vector<float>& loss_scale, float clip_norm);

    float LearningRate() const { return learning_rate_; }
    void SetLearningRate(float learning_rate);

private:
    struct Embedding {
        size_t offset;
        size_t size;       // channel size including the missing indicator
        size_t output_offset;
        std::unique_ptr<nn::Dense> dense;  // nullptr = pass through
    };

    nn::Matrix Embed(const nn::Matrix& inputs, bool training) const;
    nn::Matrix TrunkInfer(const nn::Matrix& embedded) const;
    std::vector<nn::Parameter> BodyParameters();

    size_t input_size_{0};
    size_t embedded_size_{0};
    size_t trunk_width_{0};
    float learning_rate_;
    uint32_t seed_;
    std::vector<Embedding> embeddings_;
    std::vector<std::unique_ptr<nn::Layer>> trunk_;
    std::vector<HeadSpec> head_specs_;
    std::vector<std::unique_ptr<nn::Dense>> heads_;
    nn::Adam body_optimizer_;
    std::vector<nn::Adam> head_optimizers_;
};

/// Bottleneck autoencoder on the fused vector, for ANOMALY outputs
class AnomalyAutoencoder {
public:
    /// Bottleneck = input_size / 4, clamped to [1, 64]
    AnomalyAutoencoder(size_t input_size, float learning_rate, uint32_t seed);

    /// Mean squared reconstruction error per row
    std::vector<float> Errors(const nn::Matrix& inputs) const;

    /// One optimization step; returns the batch loss
    float TrainBatch(const nn::Matrix& inputs, float clip_norm);

    void SetLearningRate(float learning_rate) { optimizer_.SetLearningRate(learning_rate); }

    size_t ParameterCount() const { return parameter_count_; }

    /// False if any weight or bias is NaN or infinite
    bool AllFinite();

    std::vector<nn::Parameter> Parameters() { return model_.Parameters(); }

private:
    nn::Sequential model_;
    nn::Adam optimizer_;
    size_t parameter_count_{0};
};

}  // namespace agent
}  // namespace dpan
