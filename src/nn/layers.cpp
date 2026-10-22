// File: src/nn/layers.cpp
#include "nn/layers.hpp"
#include "core/binary_io.hpp"

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace dpan {
namespace nn {

namespace {

void WriteString(std::ostream& out, const std::string& s) {
    io::Write<uint32_t>(out, static_cast<uint32_t>(s.size()));
    io::WriteBytes(out, s.data(), s.size());
}

std::string ReadString(std::istream& in) {
    uint32_t length = io::Read<uint32_t>(in);
    if (length > 256) {
        throw std::runtime_error("Layer::Deserialize: invalid layer type");
    }
    std::string s(length, '\0');
    io::ReadBytes(in, s.data(), length);
    return s;
}

void WriteFloat(std::ostream& out, float value) {
    io::Write<float>(out, value);
}

float ReadFloat(std::istream& in) {
    return io::Read<float>(in);
}

}  // namespace

// ============================================================================
// Layer
// ============================================================================

void Layer::ZeroGradients() {
    for (auto& param : Parameters()) {
        param.gradient->Fill(0.0f);
    }
}

void Layer::SerializeParameters(std::ostream&) const {}

void Layer::Serialize(std::ostream& out) const {
    WriteString(out, Type());
    SerializeParameters(out);
}

std::unique_ptr<Layer> Layer::Deserialize(std::istream& in) {
    std::string type = ReadString(in);
    if (type == "Dense") {
        return Dense::DeserializeParameters(in);
    }
    if (type == "ReLU") {
        return std::make_unique<ReLU>();
    }
    if (type == "LeakyReLU") {
        return std::make_unique<LeakyReLU>(ReadFloat(in));
    }
    if (type == "Sigmoid") {
        return std::make_unique<Sigmoid>();
    }
    if (type == "Tanh") {
        return std::make_unique<Tanh>();
    }
    if (type == "Softmax") {
        return std::make_unique<Softmax>();
    }
    if (type == "Dropout") {
        return std::make_unique<Dropout>(ReadFloat(in));
    }
    throw std::runtime_error("Layer::Deserialize: unknown layer type '" + type + "'");
}

// ============================================================================
// Dense
// ============================================================================

Dense::Dense(size_t input_size, size_t output_size, Initialization init, uint32_t seed)
    : weights_(input_size, output_size),
      bias_(1, output_size),
      weight_grad_(input_size, output_size),
      bias_grad_(1, output_size) {

    if (input_size == 0 || output_size == 0) {
        throw std::invalid_argument("Dense: layer sizes must be positive");
    }

    std::mt19937 rng(seed);
    switch (init) {
        case Initialization::XAVIER: {
            float limit = std::sqrt(6.0f / static_cast<float>(input_size + output_size));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (size_t i = 0; i < weights_.Size(); ++i) {
                weights_.Data()[i] = dist(rng);
            }
            break;
        }
        case Initialization::HE: {
            std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / static_cast<float>(input_size)));
            for (size_t i = 0; i < weights_.Size(); ++i) {
                weights_.Data()[i] = dist(rng);
            }
            break;
        }
        case Initialization::ZEROS:
            break;
    }
}

Matrix Dense::Forward(const Matrix& input, bool) {
    input_cache_ = input;
    return Infer(input);
}

Matrix Dense::Infer(const Matrix& input) const {
    if (input.Cols() != weights_.Rows()) {
        throw std::invalid_argument("Dense: expected " + std::to_string(weights_.Rows()) +
                                    " input features, got " + std::to_string(input.Cols()));
    }
    return AddRowVector(MatMul(input, weights_), bias_);
}

Matrix Dense::Backward(const Matrix& grad_output) {
    if (input_cache_.Empty()) {
        throw std::logic_error("Dense::Backward called before Forward");
    }
    // dL/dW = x^T g, dL/db = sum over batch of g, dL/dx = g W^T
    weight_grad_ += MatMulTransposeA(input_cache_, grad_output);
    bias_grad_ += SumRows(grad_output);
    return MatMulTransposeB(grad_output, weights_);
}

std::vector<Parameter> Dense::Parameters() {
    return {{&weights_, &weight_grad_}, {&bias_, &bias_grad_}};
}

void Dense::SerializeParameters(std::ostream& out) const {
    weights_.Serialize(out);
    bias_.Serialize(out);
}

std::unique_ptr<Dense> Dense::DeserializeParameters(std::istream& in) {
    std::unique_ptr<Dense> layer(new Dense());
    layer->weights_ = Matrix::Deserialize(in);
    layer->bias_ = Matrix::Deserialize(in);
    if (layer->bias_.Rows() != 1 || layer->bias_.Cols() != layer->weights_.Cols()) {
        throw std::runtime_error("Dense::Deserialize: bias does not match weights");
    }
    layer->weight_grad_ = Matrix(layer->weights_.Rows(), layer->weights_.Cols());
    layer->bias_grad_ = Matrix(1, layer->bias_.Cols());
    return layer;
}

std::unique_ptr<Layer> Dense::Clone() const {
    std::unique_ptr<Dense> copy(new Dense());
    copy->weights_ = weights_;
    copy->bias_ = bias_;
    copy->weight_grad_ = Matrix(weights_.Rows(), weights_.Cols());
    copy->bias_grad_ = Matrix(1, bias_.Cols());
    return copy;
}

// ============================================================================
// Element-wise activations
// ============================================================================

Matrix ElementwiseActivation::Forward(const Matrix& input, bool) {
    input_cache_ = input;
    output_cache_ = Infer(input);
    return output_cache_;
}

Matrix ElementwiseActivation::Infer(const Matrix& input) const {
    Matrix out(input.Rows(), input.Cols());
    for (size_t i = 0; i < input.Size(); ++i) {
        out.Data()[i] = Apply(input.Data()[i]);
    }
    return out;
}

Matrix ElementwiseActivation::Backward(const Matrix& grad_output) {
    if (!grad_output.SameShape(input_cache_)) {
        throw std::logic_error(Type() + "::Backward: gradient shape does not match last Forward");
    }
    Matrix grad_input(grad_output.Rows(), grad_output.Cols());
    for (size_t i = 0; i < grad_output.Size(); ++i) {
        grad_input.Data()[i] = grad_output.Data()[i] *
                               Derivative(input_cache_.Data()[i], output_cache_.Data()[i]);
    }
    return grad_input;
}

void LeakyReLU::SerializeParameters(std::ostream& out) const {
    WriteFloat(out, alpha_);
}

float Sigmoid::Apply(float x) const {
    // Split by sign so exp() never overflows
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    float e = std::exp(x);
    return e / (1.0f + e);
}

float Tanh::Apply(float x) const {
    return std::tanh(x);
}

// ============================================================================
// Softmax
// ============================================================================

Matrix Softmax::Compute(const Matrix& logits) {
    Matrix out(logits.Rows(), logits.Cols());
    for (size_t r = 0; r < logits.Rows(); ++r) {
        const float* in_row = logits.Data() + r * logits.Cols();
        float* out_row = out.Data() + r * logits.Cols();
        float max_value = *std::max_element(in_row, in_row + logits.Cols());
        float sum = 0.0f;
        for (size_t c = 0; c < logits.Cols(); ++c) {
            out_row[c] = std::exp(in_row[c] - max_value);
            sum += out_row[c];
        }
        for (size_t c = 0; c < logits.Cols(); ++c) {
            out_row[c] /= sum;
        }
    }
    return out;
}

Matrix Softmax::Forward(const Matrix& input, bool) {
    output_cache_ = Compute(input);
    return output_cache_;
}

Matrix Softmax::Infer(const Matrix& input) const {
    return Compute(input);
}

Matrix Softmax::Backward(const Matrix& grad_output) {
    if (!grad_output.SameShape(output_cache_)) {
        throw std::logic_error("Softmax::Backward: gradient shape does not match last Forward");
    }
    // dL/dx_i = y_i (g_i - sum_j g_j y_j)
    Matrix grad_input(grad_output.Rows(), grad_output.Cols());
    for (size_t r = 0; r < grad_output.Rows(); ++r) {
        float dot = 0.0f;
        for (size_t c = 0; c < grad_output.Cols(); ++c) {
            dot += grad_output(r, c) * output_cache_(r, c);
        }
        for (size_t c = 0; c < grad_output.Cols(); ++c) {
            grad_input(r, c) = output_cache_(r, c) * (grad_output(r, c) - dot);
        }
    }
    return grad_input;
}

// ============================================================================
// Dropout
// ============================================================================

Dropout::Dropout(float rate, uint32_t seed) : rate_(rate), rng_(seed) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Dropout: rate must be in [0, 1)");
    }
}

Matrix Dropout::Forward(const Matrix& input, bool training) {
    if (!training || rate_ == 0.0f) {
        mask_ = Matrix(input.Rows(), input.Cols(), 1.0f);
        return input;
    }
    std::bernoulli_distribution keep(1.0 - rate_);
    float scale = 1.0f / (1.0f - rate_);
    mask_ = Matrix(input.Rows(), input.Cols());
    for (size_t i = 0; i < mask_.Size(); ++i) {
        mask_.Data()[i] = keep(rng_) ? scale : 0.0f;
    }
    return Hadamard(input, mask_);
}

Matrix Dropout::Backward(const Matrix& grad_output) {
    return Hadamard(grad_output, mask_);
}

void Dropout::SerializeParameters(std::ostream& out) const {
    WriteFloat(out, rate_);
}

}  // namespace nn
}  // namespace dpan
