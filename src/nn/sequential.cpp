// File: src/nn/sequential.cpp
#include "nn/sequential.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace dpan {
namespace nn {

namespace {

constexpr char kModelMagic[4] = {'D', 'P', 'N', 'N'};
constexpr uint32_t kModelVersion = 1;

}  // namespace

Sequential::Sequential(const Sequential& other) {
    layers_.reserve(other.layers_.size());
    for (const auto& layer : other.layers_) {
        layers_.push_back(layer->Clone());
    }
}

Sequential& Sequential::operator=(const Sequential& other) {
    if (this != &other) {
        Sequential copy(other);
        layers_ = std::move(copy.layers_);
    }
    return *this;
}

Sequential& Sequential::Add(std::unique_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("Sequential::Add: null layer");
    }
    layers_.push_back(std::move(layer));
    return *this;
}

Sequential Sequential::Slice(size_t begin, size_t end) const {
    if (begin > end || end > layers_.size()) {
        throw std::out_of_range("Sequential::Slice: invalid layer range");
    }
    Sequential slice;
    for (size_t i = begin; i < end; ++i) {
        slice.layers_.push_back(layers_[i]->Clone());
    }
    return slice;
}

Matrix Sequential::Predict(const Matrix& input) const {
    Matrix output = input;
    for (const auto& layer : layers_) {
        output = layer->Infer(output);
    }
    return output;
}

std::vector<float> Sequential::Predict(const std::vector<float>& input) const {
    return Predict(Matrix::RowVector(input)).Row(0);
}

float Sequential::Evaluate(const Matrix& input, const Matrix& target, const Loss& loss) const {
    return loss.Value(Predict(input), target);
}

Matrix Sequential::Forward(const Matrix& input, bool training) {
    Matrix output = input;
    for (auto& layer : layers_) {
        output = layer->Forward(output, training);
    }
    return output;
}

void Sequential::Backward(const Matrix& grad_output) {
    Matrix grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = (*it)->Backward(grad);
    }
}

void Sequential::ZeroGradients() {
    for (auto& layer : layers_) {
        layer->ZeroGradients();
    }
}

std::vector<Parameter> Sequential::Parameters() {
    std::vector<Parameter> params;
    for (auto& layer : layers_) {
        for (const auto& param : layer->Parameters()) {
            params.push_back(param);
        }
    }
    return params;
}

size_t Sequential::ParameterCount() const {
    size_t count = 0;
    for (const auto& layer : layers_) {
        for (const auto& param : const_cast<Layer&>(*layer).Parameters()) {
            count += param.value->Size();
        }
    }
    return count;
}

float Sequential::TrainBatch(const Matrix& input, const Matrix& target, const Loss& loss,
                             Optimizer& optimizer) {
    if (layers_.empty()) {
        throw std::logic_error("Sequential::TrainBatch: model has no layers");
    }
    ZeroGradients();
    Matrix prediction = Forward(input, true);
    float value = loss.Value(prediction, target);
    Backward(loss.Gradient(prediction, target));
    optimizer.Step(Parameters());
    return value;
}

Sequential::History Sequential::Fit(const Matrix& input, const Matrix& target, const Loss& loss,
                                    Optimizer& optimizer) {
    return Fit(input, target, loss, optimizer, FitOptions{});
}

Sequential::History Sequential::Fit(const Matrix& input, const Matrix& target, const Loss& loss,
                                    Optimizer& optimizer, const FitOptions& options) {
    if (input.Rows() != target.Rows()) {
        throw std::invalid_argument("Sequential::Fit: input and target sample counts differ");
    }
    if (input.Rows() == 0) {
        throw std::invalid_argument("Sequential::Fit: empty dataset");
    }
    if (options.batch_size == 0) {
        throw std::invalid_argument("Sequential::Fit: batch_size must be positive");
    }
    if (options.validation_split < 0.0f || options.validation_split >= 1.0f) {
        throw std::invalid_argument("Sequential::Fit: validation_split must be in [0, 1)");
    }

    size_t validation_count = static_cast<size_t>(
        std::floor(static_cast<float>(input.Rows()) * options.validation_split));
    size_t train_count = input.Rows() - validation_count;
    if (train_count == 0) {
        throw std::invalid_argument("Sequential::Fit: validation_split leaves no training data");
    }

    std::vector<size_t> train_indices(train_count);
    std::iota(train_indices.begin(), train_indices.end(), 0);
    std::vector<size_t> validation_indices(validation_count);
    std::iota(validation_indices.begin(), validation_indices.end(), train_count);
    Matrix validation_input = input.SelectRows(validation_indices);
    Matrix validation_target = target.SelectRows(validation_indices);

    std::mt19937 rng(options.seed);
    History history;
    float best_loss = std::numeric_limits<float>::infinity();
    size_t epochs_without_improvement = 0;

    for (size_t epoch = 0; epoch < options.epochs; ++epoch) {
        if (options.shuffle) {
            std::shuffle(train_indices.begin(), train_indices.end(), rng);
        }

        double epoch_loss = 0.0;
        for (size_t start = 0; start < train_count; start += options.batch_size) {
            size_t end = std::min(start + options.batch_size, train_count);
            std::vector<size_t> batch(train_indices.begin() + start, train_indices.begin() + end);
            float batch_loss = TrainBatch(input.SelectRows(batch), target.SelectRows(batch), loss,
                                          optimizer);
            epoch_loss += static_cast<double>(batch_loss) * static_cast<double>(end - start);
        }

        float train_loss = static_cast<float>(epoch_loss / static_cast<double>(train_count));
        float validation_loss = validation_count > 0
            ? Evaluate(validation_input, validation_target, loss)
            : std::numeric_limits<float>::quiet_NaN();

        history.train_loss.push_back(train_loss);
        if (validation_count > 0) {
            history.validation_loss.push_back(validation_loss);
        }
        history.epochs_run = epoch + 1;

        if (options.on_epoch_end && !options.on_epoch_end(epoch, train_loss, validation_loss)) {
            history.stopped_early = true;
            break;
        }

        if (options.early_stopping_patience > 0) {
            float monitored = validation_count > 0 ? validation_loss : train_loss;
            if (monitored < best_loss - options.min_delta) {
                best_loss = monitored;
                epochs_without_improvement = 0;
            } else if (++epochs_without_improvement >= options.early_stopping_patience) {
                history.stopped_early = true;
                break;
            }
        }
    }

    return history;
}

void Sequential::Serialize(std::ostream& out) const {
    out.write(kModelMagic, sizeof(kModelMagic));
    out.write(reinterpret_cast<const char*>(&kModelVersion), sizeof(kModelVersion));
    uint32_t count = static_cast<uint32_t>(layers_.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& layer : layers_) {
        layer->Serialize(out);
    }
}

Sequential Sequential::Deserialize(std::istream& in) {
    char magic[4] = {};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kModelMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("Sequential::Deserialize: not a DPAN model");
    }
    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!in || version != kModelVersion) {
        throw std::runtime_error("Sequential::Deserialize: unsupported model version");
    }
    uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || count > 100000) {
        throw std::runtime_error("Sequential::Deserialize: invalid layer count");
    }
    Sequential model;
    for (uint32_t i = 0; i < count; ++i) {
        model.Add(Layer::Deserialize(in));
    }
    return model;
}

bool Sequential::Save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    Serialize(out);
    return out.good();
}

Sequential Sequential::Load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Sequential::Load: cannot open " + path);
    }
    return Deserialize(in);
}

std::string Sequential::Summary() const {
    std::ostringstream out;
    size_t total = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        size_t params = 0;
        for (const auto& param : const_cast<Layer&>(*layers_[i]).Parameters()) {
            params += param.value->Size();
        }
        total += params;
        out << i << ": " << layers_[i]->Type();
        if (const auto* dense = dynamic_cast<const Dense*>(layers_[i].get())) {
            out << " (" << dense->InputSize() << " -> " << dense->OutputSize() << ")";
        }
        out << ", " << params << " parameters\n";
    }
    out << "Total parameters: " << total << "\n";
    return out.str();
}

}  // namespace nn
}  // namespace dpan
