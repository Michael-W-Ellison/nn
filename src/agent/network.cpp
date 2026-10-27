// File: src/agent/network.cpp
#include "agent/network.hpp"

#include "core/binary_io.hpp"
#include "nn/convolution.hpp"
#include "nn/loss.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace dpan {
namespace agent {

using nn::Matrix;

AgentNetwork::AgentNetwork(const std::vector<ChannelLayout>& layout, std::vector<HeadSpec> heads,
                           float learning_rate, uint32_t seed, size_t min_width, size_t max_width,
                           size_t memory_units, float memory_timescale)
    : memory_units_(memory_units),
      learning_rate_(learning_rate),
      seed_(seed),
      head_specs_(std::move(heads)),
      body_optimizer_(learning_rate) {
    uint32_t layer_seed = seed;
    for (const auto& channel : layout) {
        if (channel.size == 0) {
            throw std::invalid_argument("AgentNetwork: channel '" + channel.name + "' has no size");
        }
        Embedding embedding{channel.offset, channel.size, embedded_size_, 0, nullptr};
        if (channel.embedding_size > 0 && channel.size > 1) {
            // Map the features; the missing indicator passes through
            if (channel.image_height > 0) {
                embedding.model = ImageModel(channel, layer_seed);
            } else {
                embedding.model = std::make_unique<nn::Sequential>();
                embedding.model->Emplace<nn::Dense>(channel.size - 1, channel.embedding_size,
                                                    nn::Initialization::XAVIER, ++layer_seed);
            }
            embedding.width = channel.embedding_size;
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
    if (memory_units_ > 0) {
        memory_ = std::make_unique<nn::GRU>(trunk_width_, memory_units_, ++layer_seed, memory_timescale);
        post_ = std::make_unique<nn::Dense>(trunk_width_ + memory_units_, trunk_width_, nn::Initialization::HE,
                                            ++layer_seed);
    }

    for (const auto& spec : head_specs_) {
        heads_.push_back(spec.size == 0 ? nullptr
                                        : std::make_unique<nn::Dense>(HeadInputWidth(), spec.size,
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
      memory_units_(other.memory_units_),
      memory_(other.memory_ ? std::make_unique<nn::GRU>(*other.memory_) : nullptr),
      post_(other.post_ ? CloneDense(*other.post_) : nullptr),
      learning_rate_(other.learning_rate_),
      seed_(other.seed_),
      head_specs_(other.head_specs_),
      body_optimizer_(other.body_optimizer_),
      head_optimizers_(other.head_optimizers_) {
    for (const auto& embedding : other.embeddings_) {
        embeddings_.push_back({embedding.offset, embedding.size, embedding.output_offset, embedding.width,
                               embedding.model ? std::make_unique<nn::Sequential>(*embedding.model) : nullptr});
    }
    for (const auto& layer : other.trunk_) {
        trunk_.push_back(layer->Clone());
    }
    for (const auto& head : other.heads_) {
        heads_.push_back(head ? CloneDense(*head) : nullptr);
    }
}

namespace {

std::unique_ptr<nn::Dense> ReadDense(std::istream& in) {
    auto layer = nn::Layer::Deserialize(in);
    if (layer->Type() != "Dense") throw std::runtime_error("AgentNetwork: expected a Dense layer");
    return std::unique_ptr<nn::Dense>(static_cast<nn::Dense*>(layer.release()));
}

}  // namespace

void AgentNetwork::Serialize(std::ostream& out) const {
    io::WriteSize(out, input_size_);
    io::WriteSize(out, embedded_size_);
    io::WriteSize(out, trunk_width_);
    io::Write<float>(out, learning_rate_);
    io::Write<uint32_t>(out, seed_);
    io::WriteSize(out, embeddings_.size());
    for (const auto& embedding : embeddings_) {
        io::WriteSize(out, embedding.offset);
        io::WriteSize(out, embedding.size);
        io::WriteSize(out, embedding.output_offset);
        io::WriteSize(out, embedding.width);
        io::Write<uint8_t>(out, embedding.model ? 1 : 0);
        if (embedding.model) embedding.model->Serialize(out);
    }
    io::WriteSize(out, trunk_.size());
    for (const auto& layer : trunk_) layer->Serialize(out);
    io::WriteSize(out, memory_units_);
    if (memory_) {
        memory_->Serialize(out);
        post_->Serialize(out);
    }
    io::WriteSize(out, heads_.size());
    for (size_t h = 0; h < heads_.size(); ++h) {
        io::WriteString(out, head_specs_[h].name);
        io::Write<uint8_t>(out, static_cast<uint8_t>(head_specs_[h].kind));
        io::WriteSize(out, head_specs_[h].size);
        io::Write<uint8_t>(out, heads_[h] ? 1 : 0);
        if (heads_[h]) heads_[h]->Serialize(out);
        head_optimizers_[h].SaveState(out);
    }
    body_optimizer_.SaveState(out);
}

std::unique_ptr<AgentNetwork> AgentNetwork::Deserialize(std::istream& in) {
    std::unique_ptr<AgentNetwork> network(new AgentNetwork());
    network->input_size_ = io::ReadSize(in, 1u << 30);
    network->embedded_size_ = io::ReadSize(in, 1u << 30);
    network->trunk_width_ = io::ReadSize(in, 1u << 20);
    network->learning_rate_ = io::Read<float>(in);
    network->seed_ = io::Read<uint32_t>(in);
    size_t embeddings = io::ReadSize(in, 1u << 20);
    for (size_t i = 0; i < embeddings; ++i) {
        Embedding embedding{};
        embedding.offset = io::ReadSize(in, 1u << 30);
        embedding.size = io::ReadSize(in, 1u << 30);
        embedding.output_offset = io::ReadSize(in, 1u << 30);
        embedding.width = io::ReadSize(in, 1u << 20);
        if (io::Read<uint8_t>(in)) embedding.model = std::make_unique<nn::Sequential>(nn::Sequential::Deserialize(in));
        network->embeddings_.push_back(std::move(embedding));
    }
    size_t layers = io::ReadSize(in, 64);
    for (size_t i = 0; i < layers; ++i) network->trunk_.push_back(nn::Layer::Deserialize(in));
    network->memory_units_ = io::ReadSize(in, 1u << 16);
    if (network->memory_units_ > 0) {
        network->memory_ = std::make_unique<nn::GRU>(nn::GRU::Deserialize(in));
        if (network->memory_->HiddenSize() != network->memory_units_ ||
            network->memory_->InputSize() != network->trunk_width_) {
            throw std::runtime_error("AgentNetwork: memory does not match the trunk");
        }
        network->post_ = ReadDense(in);
        if (network->post_->InputSize() != network->trunk_width_ + network->memory_units_) {
            throw std::runtime_error("AgentNetwork: memory layer does not match the trunk");
        }
    }
    size_t heads = io::ReadSize(in, 1u << 16);
    for (size_t h = 0; h < heads; ++h) {
        HeadSpec spec;
        spec.name = io::ReadString(in, 1u << 16);
        auto kind = io::Read<uint8_t>(in);
        if (kind > static_cast<uint8_t>(HeadKind::Q_VALUES)) throw std::runtime_error("AgentNetwork: bad head kind");
        spec.kind = static_cast<HeadKind>(kind);
        spec.size = io::ReadSize(in, 1u << 24);
        network->head_specs_.push_back(spec);
        network->heads_.push_back(io::Read<uint8_t>(in) ? ReadDense(in) : nullptr);
        network->head_optimizers_.emplace_back();
        network->head_optimizers_.back().LoadState(in);
    }
    network->body_optimizer_.LoadState(in);
    return network;
}

std::unique_ptr<nn::Sequential> AgentNetwork::ImageModel(const ChannelLayout& channel, uint32_t& seed) {
    // Up to three blocks of 3x3 convolution, ReLU and 2x2 max-pooling while
    // the image is at least 8 pixels on each side, then a projection
    constexpr size_t kFilters[] = {8, 16, 32};
    auto model = std::make_unique<nn::Sequential>();
    nn::ImageShape shape{channel.image_height, channel.image_width, channel.image_channels};
    for (size_t block = 0; block < 3 && std::min(shape.height, shape.width) >= 8; ++block) {
        model->Emplace<nn::Conv2D>(shape, kFilters[block], 3, 1, 1, nn::Initialization::HE, ++seed);
        shape.channels = kFilters[block];
        model->Emplace<nn::ReLU>();
        model->Emplace<nn::MaxPool2D>(shape, 2);
        shape = {shape.height / 2, shape.width / 2, shape.channels};
    }
    model->Emplace<nn::Flatten>(shape);
    model->Emplace<nn::Dense>(shape.Size(), channel.embedding_size, nn::Initialization::XAVIER, ++seed);
    return model;
}

Matrix AgentNetwork::Embed(const Matrix& inputs, bool training) const {
    if (inputs.Cols() != input_size_) {
        throw std::invalid_argument("AgentNetwork: expected " + std::to_string(input_size_) +
                                    " inputs, got " + std::to_string(inputs.Cols()));
    }
    Matrix embedded(inputs.Rows(), embedded_size_);
    for (const auto& embedding : embeddings_) {
        if (!embedding.model) {
            for (size_t r = 0; r < inputs.Rows(); ++r) {
                std::copy(inputs.Data() + r * input_size_ + embedding.offset,
                          inputs.Data() + r * input_size_ + embedding.offset + embedding.size,
                          embedded.Data() + r * embedded_size_ + embedding.output_offset);
            }
            continue;
        }
        Matrix slice = inputs.SliceCols(embedding.offset, embedding.offset + embedding.size - 1);
        Matrix projected = training ? embedding.model->Forward(slice, true) : embedding.model->Predict(slice);
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

std::vector<float> AgentNetwork::Losses(const Matrix& inputs,
                                        const std::vector<std::vector<const HeadTarget*>>& targets,
                                        const Matrix* memory) const {
    std::vector<float> losses(heads_.size(), std::numeric_limits<float>::quiet_NaN());
    Matrix hidden = HeadInput(inputs, memory);
    for (size_t h = 0; h < heads_.size() && h < targets.size(); ++h) {
        if (!heads_[h] || targets[h].size() != inputs.Rows() || head_specs_[h].kind == HeadKind::Q_VALUES) continue;
        Matrix output = heads_[h]->Infer(hidden);
        double loss = 0.0;
        size_t terms = 0;
        if (head_specs_[h].kind == HeadKind::CLASSIFIER) {
            Matrix probabilities = nn::Softmax::Compute(output);
            for (size_t r = 0; r < output.Rows(); ++r) {
                const HeadTarget* t = targets[h][r];
                if (!t || t->Empty()) continue;
                ++terms;
                if (t->label >= 0 && static_cast<size_t>(t->label) < output.Cols()) {
                    loss -= std::log(std::max(probabilities(r, static_cast<size_t>(t->label)), 1e-7f));
                }
                if (t->negative >= 0 && static_cast<size_t>(t->negative) < output.Cols()) {
                    loss -= std::log(std::max(1.0f - probabilities(r, static_cast<size_t>(t->negative)), 1e-4f));
                }
            }
        } else {
            for (size_t r = 0; r < output.Rows(); ++r) {
                const HeadTarget* t = targets[h][r];
                if (!t || t->values.size() != output.Cols()) continue;
                for (size_t j = 0; j < output.Cols(); ++j) {
                    if (std::isnan(t->values[j])) continue;
                    double diff = output(r, j) - t->values[j];
                    loss += diff * diff;
                    ++terms;
                }
            }
        }
        if (terms > 0) losses[h] = static_cast<float>(loss / static_cast<double>(terms));
    }
    return losses;
}

Matrix AgentNetwork::HeadInput(const Matrix& inputs, const Matrix* memory) const {
    Matrix hidden = TrunkInfer(Embed(inputs, false));
    if (!memory_) return hidden;
    if (memory && (memory->Rows() != inputs.Rows() || memory->Cols() != memory_units_)) {
        throw std::invalid_argument("AgentNetwork: expected a memory state of " + std::to_string(memory_units_) +
                                    " values per row, got " + memory->ShapeString());
    }
    Matrix joined = nn::ConcatCols(hidden, memory ? *memory : Matrix(inputs.Rows(), memory_units_));
    return post_relu_.Infer(post_->Infer(joined));
}

Matrix AgentNetwork::Step(const Matrix& inputs, const Matrix& state) const {
    if (!memory_) throw std::logic_error("AgentNetwork::Step: the network has no memory");
    return memory_->Step(TrunkInfer(Embed(inputs, false)), state);
}

Matrix AgentNetwork::Infer(const Matrix& inputs, size_t head, const Matrix* memory) const {
    if (!heads_.at(head)) {
        return Matrix(inputs.Rows(), 0);
    }
    return heads_[head]->Infer(HeadInput(inputs, memory));
}

void AgentNetwork::GrowClasses(size_t head, size_t class_count) {
    if (head_specs_.at(head).kind != HeadKind::CLASSIFIER) {
        throw std::logic_error("AgentNetwork::GrowClasses: head is not a classifier");
    }
    size_t old_count = head_specs_[head].size;
    if (class_count <= old_count) {
        return;
    }
    const size_t input_width = HeadInputWidth();
    auto grown = std::make_unique<nn::Dense>(input_width, class_count, nn::Initialization::XAVIER, ++seed_);
    if (heads_[head]) {
        for (size_t i = 0; i < input_width; ++i) {
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

size_t AgentNetwork::ParameterCount() const {
    size_t count = 0;
    for (const auto& embedding : embeddings_) {
        if (!embedding.model) continue;
        for (const auto& p : const_cast<nn::Sequential&>(*embedding.model).Parameters()) count += p.value->Size();
    }
    for (const auto& layer : trunk_) {
        if (const auto* dense = dynamic_cast<const nn::Dense*>(layer.get())) {
            count += dense->Weights().Size() + dense->Bias().Size();
        }
    }
    for (const auto& head : heads_) {
        if (head) count += head->Weights().Size() + head->Bias().Size();
    }
    if (memory_) count += memory_->ParameterCount() + post_->Weights().Size() + post_->Bias().Size();
    return count;
}

std::vector<nn::Parameter> AgentNetwork::AllParameters() {
    std::vector<nn::Parameter> parameters = BodyParameters();
    for (auto& head : heads_) {
        if (!head) continue;
        auto p = head->Parameters();
        parameters.insert(parameters.end(), p.begin(), p.end());
    }
    return parameters;
}

bool AgentNetwork::AllFinite() {
    for (const auto& parameter : AllParameters()) {
        const float* v = parameter.value->Data();
        for (size_t i = 0; i < parameter.value->Size(); ++i) {
            if (!std::isfinite(v[i])) return false;
        }
    }
    return true;
}

void AgentNetwork::Widen(size_t width) {
    const size_t old_width = trunk_width_;
    if (width <= old_width) return;
    std::mt19937 rng(++seed_);

    // Unit u of the widened layer copies unit source[u] of the old one; the
    // copies of a unit split its outgoing weights in shares summing to 1
    // (slightly unequal, so the copies do not stay identical)
    auto plan = [&](std::vector<size_t>& source, std::vector<float>& share) {
        source.resize(width);
        std::uniform_int_distribution<size_t> pick(0, old_width - 1);
        for (size_t u = 0; u < width; ++u) source[u] = u < old_width ? u : pick(rng);
        std::vector<std::vector<size_t>> copies(old_width);
        for (size_t u = 0; u < width; ++u) copies[source[u]].push_back(u);
        std::uniform_real_distribution<float> jitter(0.9f, 1.1f);
        share.assign(width, 1.0f);
        for (const auto& group : copies) {
            if (group.size() < 2) continue;
            float total = 0.0f;
            for (size_t u : group) total += (share[u] = jitter(rng));
            for (size_t u : group) share[u] /= total;
        }
    };
    std::vector<size_t> source1, source2;
    std::vector<float> share1, share2;
    plan(source1, share1);
    plan(source2, share2);

    auto* first = static_cast<nn::Dense*>(trunk_[0].get());
    auto* second = static_cast<nn::Dense*>(trunk_[2].get());
    auto wide_first = std::make_unique<nn::Dense>(embedded_size_, width, nn::Initialization::ZEROS);
    auto wide_second = std::make_unique<nn::Dense>(width, width, nn::Initialization::ZEROS);
    for (size_t u = 0; u < width; ++u) {
        for (size_t i = 0; i < embedded_size_; ++i) wide_first->Weights()(i, u) = first->Weights()(i, source1[u]);
        wide_first->Bias()(0, u) = first->Bias()(0, source1[u]);
    }
    for (size_t v = 0; v < width; ++v) {
        for (size_t u = 0; u < width; ++u) {
            wide_second->Weights()(u, v) = second->Weights()(source1[u], source2[v]) * share1[u];
        }
        wide_second->Bias()(0, v) = second->Bias()(0, source2[v]);
    }
    trunk_[0] = std::move(wide_first);
    trunk_[2] = std::move(wide_second);

    if (memory_) {
        // The GRU reads the trunk output like a head
        std::vector<Matrix> wide_input;
        for (size_t g = 0; g < 3; ++g) {
            const Matrix& old = memory_->InputWeights(g);
            Matrix wide(width, memory_units_);
            for (size_t u = 0; u < width; ++u) {
                for (size_t c = 0; c < memory_units_; ++c) wide(u, c) = old(source2[u], c) * share2[u];
            }
            wide_input.push_back(std::move(wide));
        }
        memory_->SetInputWeights(std::move(wide_input));

        // So does the layer on [trunk, state]; the heads read its output,
        // whose width does not change
        auto wide_post = std::make_unique<nn::Dense>(width + memory_units_, post_->OutputSize(),
                                                     nn::Initialization::ZEROS);
        for (size_t c = 0; c < post_->OutputSize(); ++c) {
            for (size_t u = 0; u < width; ++u) wide_post->Weights()(u, c) = post_->Weights()(source2[u], c) * share2[u];
            for (size_t m = 0; m < memory_units_; ++m) {
                wide_post->Weights()(width + m, c) = post_->Weights()(old_width + m, c);
            }
        }
        wide_post->Bias() = post_->Bias();
        post_ = std::move(wide_post);
    }

    for (auto& head : heads_) {
        if (!head || post_) continue;
        auto wide_head = std::make_unique<nn::Dense>(width, head->OutputSize(), nn::Initialization::ZEROS);
        for (size_t u = 0; u < width; ++u) {
            for (size_t c = 0; c < head->OutputSize(); ++c) {
                wide_head->Weights()(u, c) = head->Weights()(source2[u], c) * share2[u];
            }
        }
        wide_head->Bias() = head->Bias();
        head = std::move(wide_head);
    }
    trunk_width_ = width;
    body_optimizer_.Reset();
    for (auto& optimizer : head_optimizers_) optimizer.Reset();
}

std::vector<nn::Parameter> AgentNetwork::BodyParameters() {
    std::vector<nn::Parameter> parameters;
    for (auto& embedding : embeddings_) {
        if (embedding.model) {
            auto p = embedding.model->Parameters();
            parameters.insert(parameters.end(), p.begin(), p.end());
        }
    }
    for (auto& layer : trunk_) {
        auto p = layer->Parameters();
        parameters.insert(parameters.end(), p.begin(), p.end());
    }
    if (memory_) {
        auto p = memory_->Parameters();
        parameters.insert(parameters.end(), p.begin(), p.end());
        p = post_->Parameters();
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

Matrix AgentNetwork::BodyForward(const Matrix& inputs) {
    for (auto& embedding : embeddings_) {
        if (embedding.model) embedding.model->ZeroGradients();
    }
    for (auto& layer : trunk_) {
        layer->ZeroGradients();
    }
    if (memory_) {
        memory_->ZeroGradients();
        post_->ZeroGradients();
    }
    Matrix hidden = Embed(inputs, true);
    for (auto& layer : trunk_) {
        hidden = layer->Forward(hidden, true);
    }
    return hidden;
}

Matrix AgentNetwork::HeadsStep(const Matrix& head_input, const std::vector<std::vector<const HeadTarget*>>& targets,
                               const std::vector<float>& loss_scale, std::vector<float>& losses,
                               std::vector<bool>& trained) {
    const size_t rows = head_input.Rows();
    losses.assign(heads_.size(), std::numeric_limits<float>::quiet_NaN());
    trained.assign(heads_.size(), false);
    Matrix grad_input(rows, head_input.Cols());
    bool any = false;
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
        Matrix output = head->Forward(head_input, true);
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
        grad_input += head->Backward(grad);
        trained[h] = true;
        any = true;
    }
    return any ? grad_input : Matrix();
}

void AgentNetwork::BodyBackward(const Matrix& grad_hidden) {
    const size_t rows = grad_hidden.Rows();
    Matrix grad = grad_hidden;
    for (auto it = trunk_.rbegin(); it != trunk_.rend(); ++it) {
        grad = (*it)->Backward(grad);
    }
    for (auto& embedding : embeddings_) {
        if (!embedding.model) continue;
        size_t width = embedding.width;
        Matrix slice(rows, width);
        for (size_t r = 0; r < rows; ++r) {
            std::copy(grad.Data() + r * embedded_size_ + embedding.output_offset,
                      grad.Data() + r * embedded_size_ + embedding.output_offset + width,
                      slice.Data() + r * width);
        }
        embedding.model->Backward(slice);
    }
}

void AgentNetwork::Update(const std::vector<bool>& trained, float clip_norm) {
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
}

std::vector<float> AgentNetwork::TrainBatch(const Matrix& inputs,
                                            const std::vector<std::vector<const HeadTarget*>>& targets,
                                            const std::vector<float>& loss_scale, float clip_norm) {
    if (memory_) throw std::logic_error("AgentNetwork::TrainBatch: a network with memory trains on sequences");
    if (targets.size() != heads_.size() || loss_scale.size() != heads_.size()) {
        throw std::invalid_argument("AgentNetwork::TrainBatch: one target list and scale per head");
    }
    Matrix hidden = BodyForward(inputs);
    std::vector<float> losses;
    std::vector<bool> trained;
    Matrix grad_hidden = HeadsStep(hidden, targets, loss_scale, losses, trained);
    if (grad_hidden.Empty()) return losses;
    BodyBackward(grad_hidden);
    Update(trained, clip_norm);
    return losses;
}

std::vector<Matrix> AgentNetwork::States(const SequenceBatch& batch) const {
    if (!memory_) throw std::logic_error("AgentNetwork::States: the network has no memory");
    std::vector<Matrix> states;
    Matrix state = batch.initial;
    for (size_t t = 0; t < batch.inputs.size(); ++t) {
        if (!batch.resets.empty()) {
            for (size_t r = 0; r < batch.resets[t].size() && r < state.Rows(); ++r) {
                if (!batch.resets[t][r]) continue;
                for (size_t c = 0; c < memory_units_; ++c) state(r, c) = 0.0f;
            }
        }
        state = memory_->Step(TrunkInfer(Embed(batch.inputs[t], false)), state);
        states.push_back(state);
    }
    return states;
}

std::vector<float> AgentNetwork::TrainSequences(const SequenceBatch& batch,
                                                const std::vector<std::vector<const HeadTarget*>>& targets,
                                                const std::vector<float>& loss_scale, float clip_norm) {
    if (!memory_) throw std::logic_error("AgentNetwork::TrainSequences: the network has no memory");
    const size_t steps = batch.inputs.size();
    const size_t rows = steps == 0 ? 0 : batch.inputs[0].Rows();
    if (targets.size() != heads_.size() || loss_scale.size() != heads_.size()) {
        throw std::invalid_argument("AgentNetwork::TrainSequences: one target list and scale per head");
    }
    if (batch.initial.Rows() != rows || batch.initial.Cols() != memory_units_) {
        throw std::invalid_argument("AgentNetwork::TrainSequences: initial state must be rows x memory units");
    }
    std::vector<float> losses(heads_.size(), std::numeric_limits<float>::quiet_NaN());
    if (steps == 0 || rows == 0) return losses;

    // All steps through embeddings and trunk at once (time-major rows)
    Matrix stacked(steps * rows, input_size_);
    for (size_t t = 0; t < steps; ++t) {
        if (batch.inputs[t].Rows() != rows || batch.inputs[t].Cols() != input_size_) {
            throw std::invalid_argument("AgentNetwork::TrainSequences: every step needs rows x inputs");
        }
        std::copy(batch.inputs[t].Data(), batch.inputs[t].Data() + rows * input_size_,
                  stacked.Data() + t * rows * input_size_);
    }
    Matrix hidden = BodyForward(stacked);
    std::vector<Matrix> trunk_steps(steps, Matrix(rows, trunk_width_));
    for (size_t t = 0; t < steps; ++t) {
        std::copy(hidden.Data() + t * rows * trunk_width_, hidden.Data() + (t + 1) * rows * trunk_width_,
                  trunk_steps[t].Data());
    }
    std::vector<Matrix> states = memory_->Forward(trunk_steps, batch.initial, batch.resets);

    const size_t head_width = trunk_width_ + memory_units_;
    Matrix head_input(steps * rows, head_width);
    for (size_t t = 0; t < steps; ++t) {
        for (size_t r = 0; r < rows; ++r) {
            float* row = head_input.Data() + (t * rows + r) * head_width;
            std::copy(trunk_steps[t].Data() + r * trunk_width_, trunk_steps[t].Data() + (r + 1) * trunk_width_, row);
            std::copy(states[t].Data() + r * memory_units_, states[t].Data() + (r + 1) * memory_units_,
                      row + trunk_width_);
        }
    }
    Matrix post = post_relu_.Forward(post_->Forward(head_input, true), true);
    std::vector<bool> trained;
    Matrix grad_post = HeadsStep(post, targets, loss_scale, losses, trained);
    if (grad_post.Empty()) return losses;
    Matrix grad_input = post_->Backward(post_relu_.Backward(grad_post));

    // Split the gradient: the trunk part goes straight back, the memory part
    // back through time, then into the trunk outputs of each step
    std::vector<Matrix> grad_states(steps, Matrix(rows, memory_units_));
    Matrix grad_hidden(steps * rows, trunk_width_);
    for (size_t t = 0; t < steps; ++t) {
        for (size_t r = 0; r < rows; ++r) {
            const float* row = grad_input.Data() + (t * rows + r) * head_width;
            std::copy(row, row + trunk_width_, grad_hidden.Data() + (t * rows + r) * trunk_width_);
            std::copy(row + trunk_width_, row + head_width, grad_states[t].Data() + r * memory_units_);
        }
    }
    std::vector<Matrix> grad_trunk = memory_->Backward(grad_states);
    for (size_t t = 0; t < steps; ++t) {
        float* out = grad_hidden.Data() + t * rows * trunk_width_;
        const float* in = grad_trunk[t].Data();
        for (size_t i = 0; i < rows * trunk_width_; ++i) out[i] += in[i];
    }
    BodyBackward(grad_hidden);
    Update(trained, clip_norm);
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
    parameter_count_ = (input_size + 1) * bottleneck + (bottleneck + 1) * input_size;
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

void AnomalyAutoencoder::Serialize(std::ostream& out) const {
    model_.Serialize(out);
    optimizer_.SaveState(out);
    io::WriteSize(out, parameter_count_);
}

std::unique_ptr<AnomalyAutoencoder> AnomalyAutoencoder::Deserialize(std::istream& in) {
    std::unique_ptr<AnomalyAutoencoder> autoencoder(new AnomalyAutoencoder());
    autoencoder->model_ = nn::Sequential::Deserialize(in);
    autoencoder->optimizer_.LoadState(in);
    autoencoder->parameter_count_ = io::ReadSize(in);
    return autoencoder;
}

bool AnomalyAutoencoder::AllFinite() {
    for (const auto& parameter : model_.Parameters()) {
        const float* v = parameter.value->Data();
        for (size_t i = 0; i < parameter.value->Size(); ++i) {
            if (!std::isfinite(v[i])) return false;
        }
    }
    return true;
}

float AnomalyAutoencoder::TrainBatch(const Matrix& inputs, float clip_norm) {
    return model_.TrainBatch(inputs, inputs, nn::MeanSquaredError(), optimizer_, clip_norm);
}

}  // namespace agent
}  // namespace dpan
