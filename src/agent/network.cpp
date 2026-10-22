// File: src/agent/network.cpp
#include "agent/network.hpp"

#include "nn/loss.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dpan {
namespace agent {

using nn::Matrix;

AgentNetwork::AgentNetwork(const std::vector<ChannelLayout>& layout, std::vector<HeadSpec> heads,
                           float learning_rate, uint32_t seed, size_t min_width, size_t max_width)
    : learning_rate_(learning_rate),
      seed_(seed),
      head_specs_(std::move(heads)),
      body_optimizer_(learning_rate) {
    uint32_t layer_seed = seed;
    for (const auto& channel : layout) {
        if (channel.size == 0) {
            throw std::invalid_argument("AgentNetwork: channel '" + channel.name + "' has no size");
        }
        Embedding embedding{channel.offset, channel.size, embedded_size_, nullptr};
        if (channel.embedding_size > 0 && channel.size > 1) {
            // Project the features; the missing indicator passes through
            embedding.dense = std::make_unique<nn::Dense>(channel.size - 1, channel.embedding_size,
                                                          nn::Initialization::XAVIER, ++layer_seed);
            embedded_size_ += channel.embedding_size + 1;
        } else {
            embedded_size_ += channel.size;
        }
        input_size_ = std::max(input_size_, channel.offset + channel.size);
        embeddings_.push_back(std::move(embedding));
    }

    trunk_width_ = std::max(min_width, std::min(max_width, 2 * embedded_size_));
    trunk_.push_back(std::make_unique<nn::Dense>(embedded_size_, trunk_width_, nn::Initialization::HE,
                                                 ++layer_seed));
    trunk_.push_back(std::make_unique<nn::ReLU>());
    trunk_.push_back(std::make_unique<nn::Dense>(trunk_width_, trunk_width_, nn::Initialization::HE,
                                                 ++layer_seed));
    trunk_.push_back(std::make_unique<nn::ReLU>());

    for (const auto& spec : head_specs_) {
        heads_.push_back(spec.size == 0 ? nullptr
                                        : std::make_unique<nn::Dense>(trunk_width_, spec.size,
                                                                      nn::Initialization::XAVIER,
                                                                      ++layer_seed));
        head_optimizers_.emplace_back(learning_rate);
    }
    seed_ = layer_seed;
}

namespace {

std::unique_ptr<nn::Dense> CloneDense(const nn::Dense& dense) {
    return std::unique_ptr<nn::Dense>(static_cast<nn::Dense*>(dense.Clone().release()));
}

}  // namespace

AgentNetwork::AgentNetwork(const AgentNetwork& other)
    : input_size_(other.input_size_),
      embedded_size_(other.embedded_size_),
      trunk_width_(other.trunk_width_),
      learning_rate_(other.learning_rate_),
      seed_(other.seed_),
      head_specs_(other.head_specs_),
      body_optimizer_(other.body_optimizer_),
      head_optimizers_(other.head_optimizers_) {
    for (const auto& embedding : other.embeddings_) {
        embeddings_.push_back({embedding.offset, embedding.size, embedding.output_offset,
                               embedding.dense ? CloneDense(*embedding.dense) : nullptr});
    }
    for (const auto& layer : other.trunk_) {
        trunk_.push_back(layer->Clone());
    }
    for (const auto& head : other.heads_) {
        heads_.push_back(head ? CloneDense(*head) : nullptr);
    }
}

Matrix AgentNetwork::Embed(const Matrix& inputs, bool training) const {
    if (inputs.Cols() != input_size_) {
        throw std::invalid_argument("AgentNetwork: expected " + std::to_string(input_size_) +
                                    " inputs, got " + std::to_string(inputs.Cols()));
    }
    Matrix embedded(inputs.Rows(), embedded_size_);
    for (const auto& embedding : embeddings_) {
        if (!embedding.dense) {
            for (size_t r = 0; r < inputs.Rows(); ++r) {
                std::copy(inputs.Data() + r * input_size_ + embedding.offset,
                          inputs.Data() + r * input_size_ + embedding.offset + embedding.size,
                          embedded.Data() + r * embedded_size_ + embedding.output_offset);
            }
            continue;
        }
        Matrix slice = inputs.SliceCols(embedding.offset, embedding.offset + embedding.size - 1);
        Matrix projected = training ? embedding.dense->Forward(slice, true) : embedding.dense->Infer(slice);
        size_t width = projected.Cols();
        for (size_t r = 0; r < inputs.Rows(); ++r) {
            std::copy(projected.Data() + r * width, projected.Data() + (r + 1) * width,
                      embedded.Data() + r * embedded_size_ + embedding.output_offset);
            embedded(r, embedding.output_offset + width) =
                inputs(r, embedding.offset + embedding.size - 1);
        }
    }
    return embedded;
}

Matrix AgentNetwork::TrunkInfer(const Matrix& embedded) const {
    Matrix hidden = embedded;
    for (const auto& layer : trunk_) {
        hidden = layer->Infer(hidden);
    }
    return hidden;
}

Matrix AgentNetwork::Infer(const Matrix& inputs, size_t head) const {
    if (!heads_.at(head)) {
        return Matrix(inputs.Rows(), 0);
    }
    return heads_[head]->Infer(TrunkInfer(Embed(inputs, false)));
}

void AgentNetwork::GrowClasses(size_t head, size_t class_count) {
    if (head_specs_.at(head).kind != HeadKind::CLASSIFIER) {
        throw std::logic_error("AgentNetwork::GrowClasses: head is not a classifier");
    }
    size_t old_count = head_specs_[head].size;
    if (class_count <= old_count) {
        return;
    }
    auto grown = std::make_unique<nn::Dense>(trunk_width_, class_count, nn::Initialization::XAVIER,
                                             ++seed_);
    if (heads_[head]) {
        for (size_t i = 0; i < trunk_width_; ++i) {
            for (size_t c = 0; c < old_count; ++c) {
                grown->Weights()(i, c) = heads_[head]->Weights()(i, c);
            }
        }
        for (size_t c = 0; c < old_count; ++c) {
            grown->Bias()(0, c) = heads_[head]->Bias()(0, c);
        }
    }
    heads_[head] = std::move(grown);
    head_specs_[head].size = class_count;
    head_optimizers_[head].Reset();
}

std::vector<nn::Parameter> AgentNetwork::BodyParameters() {
    std::vector<nn::Parameter> parameters;
    for (auto& embedding : embeddings_) {
        if (embedding.dense) {
            auto p = embedding.dense->Parameters();
            parameters.insert(parameters.end(), p.begin(), p.end());
        }
    }
    for (auto& layer : trunk_) {
        auto p = layer->Parameters();
        parameters.insert(parameters.end(), p.begin(), p.end());
    }
    return parameters;
}

void AgentNetwork::SetLearningRate(float learning_rate) {
    learning_rate_ = learning_rate;
    body_optimizer_.SetLearningRate(learning_rate);
    for (auto& optimizer : head_optimizers_) {
        optimizer.SetLearningRate(learning_rate);
    }
}

std::vector<float> AgentNetwork::TrainBatch(const Matrix& inputs,
                                            const std::vector<std::vector<const HeadTarget*>>& targets,
                                            const std::vector<float>& loss_scale, float clip_norm) {
    const size_t rows = inputs.Rows();
    if (targets.size() != heads_.size() || loss_scale.size() != heads_.size()) {
        throw std::invalid_argument("AgentNetwork::TrainBatch: one target list and scale per head");
    }
    std::vector<float> losses(heads_.size(), std::numeric_limits<float>::quiet_NaN());

    // Forward through embeddings and trunk, caching for Backward()
    for (auto& embedding : embeddings_) {
        if (embedding.dense) embedding.dense->ZeroGradients();
    }
    for (auto& layer : trunk_) {
        layer->ZeroGradients();
    }
    Matrix embedded = Embed(inputs, true);
    Matrix hidden = embedded;
    for (auto& layer : trunk_) {
        hidden = layer->Forward(hidden, true);
    }

    Matrix grad_hidden(rows, trunk_width_);
    bool any = false;
    std::vector<bool> trained(heads_.size(), false);
    for (size_t h = 0; h < heads_.size(); ++h) {
        auto& head = heads_[h];
        if (!head || targets[h].size() != rows) {
            continue;
        }
        size_t count = 0;
        for (const auto* target : targets[h]) {
            count += target && !target->Empty() ? 1 : 0;
        }
        if (count == 0) {
            continue;
        }

        head->ZeroGradients();
        Matrix output = head->Forward(hidden, true);
        const size_t width = output.Cols();
        Matrix grad(rows, width);
        double loss = 0.0;
        size_t terms = 0;

        if (head_specs_[h].kind == HeadKind::CLASSIFIER) {
            Matrix probabilities = nn::Softmax::Compute(output);
            for (size_t r = 0; r < rows; ++r) {
                const HeadTarget* target = targets[h][r];
                if (!target || target->Empty()) continue;
                ++terms;
                if (target->label >= 0 && static_cast<size_t>(target->label) < width) {
                    auto c = static_cast<size_t>(target->label);
                    loss -= std::log(std::max(probabilities(r, c), 1e-7f));
                    for (size_t j = 0; j < width; ++j) {
                        grad(r, j) += probabilities(r, j) - (j == c ? 1.0f : 0.0f);
                    }
                }
                if (target->negative >= 0 && static_cast<size_t>(target->negative) < width) {
                    // Loss -log(1 - p_c) pushes the predicted class down
                    auto c = static_cast<size_t>(target->negative);
                    float p = probabilities(r, c);
                    float rest = std::max(1.0f - p, 1e-4f);
                    loss -= std::log(rest);
                    for (size_t j = 0; j < width; ++j) {
                        grad(r, j) += p * ((j == c ? 1.0f : 0.0f) - probabilities(r, j)) / rest;
                    }
                }
            }
            grad *= 1.0f / static_cast<float>(terms);
            loss /= static_cast<double>(terms);
        } else {
            for (size_t r = 0; r < rows; ++r) {
                const HeadTarget* target = targets[h][r];
                if (!target || target->values.size() != width) continue;
                for (size_t j = 0; j < width; ++j) {
                    float t = target->values[j];
                    if (std::isnan(t)) continue;
                    float diff = output(r, j) - t;
                    if (head_specs_[h].kind == HeadKind::Q_VALUES && std::fabs(diff) > 1.0f) {
                        loss += std::fabs(diff) - 0.5;  // Huber, linear part
                        grad(r, j) = diff > 0.0f ? 1.0f : -1.0f;
                    } else if (head_specs_[h].kind == HeadKind::Q_VALUES) {
                        loss += 0.5 * diff * diff;
                        grad(r, j) = diff;
                    } else {
                        loss += static_cast<double>(diff) * diff;
                        grad(r, j) = 2.0f * diff;
                    }
                    ++terms;
                }
            }
            if (terms == 0) continue;
            grad *= 1.0f / static_cast<float>(terms);
            loss /= static_cast<double>(terms);
        }

        losses[h] = static_cast<float>(loss);
        grad *= loss_scale[h];
        grad_hidden += head->Backward(grad);
        trained[h] = true;
        any = true;
    }
    if (!any) {
        return losses;
    }

    Matrix grad = grad_hidden;
    for (auto it = trunk_.rbegin(); it != trunk_.rend(); ++it) {
        grad = (*it)->Backward(grad);
    }
    for (auto& embedding : embeddings_) {
        if (!embedding.dense) continue;
        size_t width = embedding.dense->OutputSize();
        Matrix slice(rows, width);
        for (size_t r = 0; r < rows; ++r) {
            std::copy(grad.Data() + r * embedded_size_ + embedding.output_offset,
                      grad.Data() + r * embedded_size_ + embedding.output_offset + width,
                      slice.Data() + r * width);
        }
        embedding.dense->Backward(slice);
    }

    // Clip all gradients together, then update
    std::vector<nn::Parameter> body = BodyParameters();
    std::vector<nn::Parameter> all = body;
    for (size_t h = 0; h < heads_.size(); ++h) {
        if (trained[h]) {
            auto p = heads_[h]->Parameters();
            all.insert(all.end(), p.begin(), p.end());
        }
    }
    if (clip_norm > 0.0f) {
        nn::ClipGradientNorm(all, clip_norm);
    }
    body_optimizer_.Step(body);
    for (size_t h = 0; h < heads_.size(); ++h) {
        if (trained[h]) {
            head_optimizers_[h].Step(heads_[h]->Parameters());
        }
    }
    return losses;
}

// ============================================================================
// AnomalyAutoencoder
// ============================================================================

AnomalyAutoencoder::AnomalyAutoencoder(size_t input_size, float learning_rate, uint32_t seed)
    : optimizer_(learning_rate) {
    // Narrower than the input, or the autoencoder can learn to copy anything
    size_t bottleneck = std::max<size_t>(1, std::min<size_t>(64, input_size / 4));
    model_.Emplace<nn::Dense>(input_size, bottleneck, nn::Initialization::HE, seed)
        .Emplace<nn::ReLU>()
        .Emplace<nn::Dense>(bottleneck, input_size, nn::Initialization::XAVIER, seed + 1);
}

std::vector<float> AnomalyAutoencoder::Errors(const Matrix& inputs) const {
    Matrix reconstruction = model_.Predict(inputs);
    std::vector<float> errors(inputs.Rows(), 0.0f);
    for (size_t r = 0; r < inputs.Rows(); ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < inputs.Cols(); ++c) {
            double diff = reconstruction(r, c) - inputs(r, c);
            sum += diff * diff;
        }
        errors[r] = static_cast<float>(sum / static_cast<double>(inputs.Cols()));
    }
    return errors;
}

float AnomalyAutoencoder::TrainBatch(const Matrix& inputs, float clip_norm) {
    return model_.TrainBatch(inputs, inputs, nn::MeanSquaredError(), optimizer_, clip_norm);
}

}  // namespace agent
}  // namespace dpan
