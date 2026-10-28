// File: src/agent/network.hpp
//
// The agent's neural network (docs/AGENT_PLAN.md, sections 2-3):
//
//   fused vector -> channel embeddings -> trunk (2 x Dense+ReLU) -> one head per output
//
// With memory (memory_units > 0) a GRU after the trunk carries a state from
// step to step, h_t = GRU(trunk(x_t), h_{t-1}), and a Dense+ReLU layer on
// [trunk output, h_t] feeds the heads (so they can combine the current
// observation with the remembered ones non-linearly). The caller keeps the
// state between steps (Step) and passes it to Infer; training runs whole
// sequences (TrainSequences).
//
// Channels with an embedding size are mapped by their own learned layers:
// hashed categories and text by a Dense projection, images by convolution
// and pooling layers and a projection; other channels pass through. Each head is a
// Dense layer on the trunk output: classifiers produce logits (softmax
// cross-entropy), regressors produce values (squared error). A head is
// trained only on the samples that have a target for it.
#pragma once

#include "agent/encoders.hpp"
#include "agent/word_sequence.hpp"
#include "nn/layers.hpp"
#include "nn/optimizer.hpp"
#include "nn/recurrent.hpp"
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

/// Sequences for TrainSequences: inputs[t] holds step t of every sequence
/// (rows x inputs), initial the state before step 0 (rows x memory units),
/// resets[t][row] starts a sequence over at step t (state 0). Shorter
/// sequences are padded at the end with rows that have no targets.
struct SequenceBatch {
    std::vector<nn::Matrix> inputs;
    nn::Matrix initial;
    std::vector<std::vector<bool>> resets;
};

class AgentNetwork {
public:
    /// @param layout Channel layout of the fused vector (ObservationEncoder::Layout)
    /// @param min_width, max_width Trunk width = 2 x embedded size, clamped
    /// @param memory_units GRU state size (0 = no memory)
    /// @param memory_timescale Longest time scale of the GRU's chrono
    ///        initialization (0 = update-gate bias 1)
    AgentNetwork(const std::vector<ChannelLayout>& layout, std::vector<HeadSpec> heads,
                 float learning_rate, uint32_t seed, size_t min_width = 32, size_t max_width = 512,
                 size_t memory_units = 0, float memory_timescale = 0.0f);

    /// Deep copy (weights and optimizer state), e.g. for a target network
    AgentNetwork(const AgentNetwork& other);

    /// Write / read everything: layout, weights, heads, optimizer state
    /// @throws std::runtime_error on malformed data (Deserialize)
    void Serialize(std::ostream& out) const;
    static std::unique_ptr<AgentNetwork> Deserialize(std::istream& in);
    AgentNetwork& operator=(const AgentNetwork&) = delete;

    size_t InputSize() const { return input_size_; }
    size_t EmbeddedSize() const { return embedded_size_; }
    size_t TrunkWidth() const { return trunk_width_; }
    size_t HeadCount() const { return heads_.size(); }
    size_t MemoryUnits() const { return memory_units_; }

    /// Memory: the state after the given inputs, from the state before them
    /// (rows x memory units each)
    nn::Matrix Step(const nn::Matrix& inputs, const nn::Matrix& state) const;

    /// Number of trainable values (weights and biases)
    size_t ParameterCount() const;

    /// After Deserialize: give the TEXT channels their fixed word vectors
    /// (ChannelLayout::word_vectors, from the encoder)
    /// @throws std::invalid_argument if they do not match
    void AttachWordVectors(const std::vector<ChannelLayout>& layout);

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
    /// classifier with no classes returns a matrix with 0 columns. With
    /// memory, `memory` holds each row's state after its step (nullptr = 0).
    nn::Matrix Infer(const nn::Matrix& inputs, size_t head, const nn::Matrix* memory = nullptr) const;

    /// Add classes to a classifier head (existing weights are kept, the
    /// head's optimizer state is reset)
    void GrowClasses(size_t head, size_t class_count);

    /// One optimization step. targets[h][row] (nullptr = no target);
    /// loss_scale[h] multiplies head h's loss. Gradients are clipped to
    /// clip_norm (0 = off) before the update.
    /// @return Mean loss per head over its targets (NaN for heads without any)
    /// @throws std::logic_error for a network with memory
    std::vector<float> TrainBatch(const nn::Matrix& inputs,
                                  const std::vector<std::vector<const HeadTarget*>>& targets,
                                  const std::vector<float>& loss_scale, float clip_norm);

    /// Memory: the state after each step of the sequences (no training),
    /// one rows x memory units matrix per step
    std::vector<nn::Matrix> States(const SequenceBatch& batch) const;

    /// One optimization step on sequences, with backpropagation through time
    /// (networks with memory). targets[h] has one entry per step and row,
    /// time-major (index t x rows + row).
    std::vector<float> TrainSequences(const SequenceBatch& batch,
                                      const std::vector<std::vector<const HeadTarget*>>& targets,
                                      const std::vector<float>& loss_scale, float clip_norm);

    /// Mean loss per head over its targets, as TrainBatch computes it, without
    /// training (`memory` as for Infer)
    std::vector<float> Losses(const nn::Matrix& inputs, const std::vector<std::vector<const HeadTarget*>>& targets,
                              const nn::Matrix* memory = nullptr) const;

    float LearningRate() const { return learning_rate_; }
    void SetLearningRate(float learning_rate);

private:
    /// A channel's part of the network input: passed through, or its
    /// features (without the missing indicator) mapped by `model` to
    /// `width` values (a Dense projection, or convolution, pooling and a
    /// projection for images); the missing indicator follows
    struct Embedding {
        size_t offset;
        size_t size;       // channel size including the missing indicator
        size_t output_offset;
        size_t width{0};
        std::unique_ptr<nn::Sequential> model;  // nullptr = pass through
        // TEXT word sequence: the channel's word ids follow the features
        // the model reads; the reader's output follows the model's
        std::unique_ptr<WordSequenceModel> words;
        size_t Features() const { return size - 1 - (words ? words->Length() : 0); }
    };

    static std::unique_ptr<nn::Sequential> ImageModel(const ChannelLayout& channel, uint32_t& seed);

    AgentNetwork() = default;
    nn::Matrix Embed(const nn::Matrix& inputs, bool training) const;
    nn::Matrix TrunkInfer(const nn::Matrix& embedded) const;
    /// Trunk output, followed by the memory state when there is one
    nn::Matrix HeadInput(const nn::Matrix& inputs, const nn::Matrix* memory) const;
    std::vector<nn::Parameter> BodyParameters();

    // Training pieces: embeddings and trunk forward (caching), the heads'
    // losses and backward pass (returns the gradient of their input, empty
    // if no head had a target), the trunk and embeddings backward pass, the update
    nn::Matrix BodyForward(const nn::Matrix& inputs);
    nn::Matrix HeadsStep(const nn::Matrix& head_input, const std::vector<std::vector<const HeadTarget*>>& targets,
                         const std::vector<float>& loss_scale, std::vector<float>& losses, std::vector<bool>& trained);
    void BodyBackward(const nn::Matrix& grad_hidden);
    void Update(const std::vector<bool>& trained, float clip_norm);

    size_t input_size_{0};
    size_t embedded_size_{0};
    size_t trunk_width_{0};
    size_t memory_units_{0};
    std::unique_ptr<nn::GRU> memory_;
    std::unique_ptr<nn::Dense> post_;  // memory: [trunk, state] -> head input (then ReLU)
    nn::ReLU post_relu_;
    size_t HeadInputWidth() const { return post_ ? post_->OutputSize() : trunk_width_; }
    float learning_rate_{0.0f};
    uint32_t seed_{0};
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

    void Serialize(std::ostream& out) const;
    static std::unique_ptr<AnomalyAutoencoder> Deserialize(std::istream& in);

    void SetLearningRate(float learning_rate) { optimizer_.SetLearningRate(learning_rate); }

    size_t ParameterCount() const { return parameter_count_; }

    /// False if any weight or bias is NaN or infinite
    bool AllFinite();

    std::vector<nn::Parameter> Parameters() { return model_.Parameters(); }

private:
    AnomalyAutoencoder() = default;

    nn::Sequential model_;
    nn::Adam optimizer_;
    size_t parameter_count_{0};
};

}  // namespace agent
}  // namespace dpan
