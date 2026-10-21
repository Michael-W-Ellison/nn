// File: src/nn/layers.hpp
//
// Neural network layers with forward and backward (backpropagation) passes.
//
// Forward() caches what Backward() needs, so training a layer is not
// thread-safe. Infer() computes the same output as Forward(x, false) without
// caching and is safe to call concurrently.
#pragma once

#include "nn/matrix.hpp"

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dpan {
namespace nn {

/// A trainable tensor and its gradient
struct Parameter {
    Matrix* value;
    Matrix* gradient;
};

class Layer {
public:
    virtual ~Layer() = default;

    /// Compute outputs for a batch (one sample per row) and cache state for
    /// Backward(). `training` enables training-only behaviour (dropout).
    virtual Matrix Forward(const Matrix& input, bool training) = 0;

    /// Given dLoss/dOutput for the last Forward() batch, accumulate parameter
    /// gradients and return dLoss/dInput
    virtual Matrix Backward(const Matrix& grad_output) = 0;

    /// Inference-only forward pass (no caching, thread-safe)
    virtual Matrix Infer(const Matrix& input) const = 0;

    /// Trainable parameters (empty for parameter-free layers)
    virtual std::vector<Parameter> Parameters() { return {}; }

    /// Reset accumulated gradients to zero
    void ZeroGradients();

    /// Layer type name used in serialization (e.g. "Dense")
    virtual std::string Type() const = 0;

    /// Write layer configuration and parameters (not the type name)
    virtual void SerializeParameters(std::ostream& out) const;

    virtual std::unique_ptr<Layer> Clone() const = 0;

    /// Read a layer written by Serialize()
    static std::unique_ptr<Layer> Deserialize(std::istream& in);

    /// Write type name and parameters
    void Serialize(std::ostream& out) const;
};

/// Weight initialization schemes for Dense layers
enum class Initialization {
    XAVIER,  ///< Uniform(±sqrt(6 / (in + out))): sigmoid / tanh
    HE,      ///< Normal(0, sqrt(2 / in)): ReLU family
    ZEROS    ///< All zeros (tests)
};

/// Fully connected layer: y = x W + b, W is in x out, b is 1 x out
class Dense : public Layer {
public:
    Dense(size_t input_size, size_t output_size,
          Initialization init = Initialization::XAVIER, uint32_t seed = 42);

    Matrix Forward(const Matrix& input, bool training) override;
    Matrix Backward(const Matrix& grad_output) override;
    Matrix Infer(const Matrix& input) const override;
    std::vector<Parameter> Parameters() override;

    std::string Type() const override { return "Dense"; }
    void SerializeParameters(std::ostream& out) const override;
    std::unique_ptr<Layer> Clone() const override;
    static std::unique_ptr<Dense> DeserializeParameters(std::istream& in);

    size_t InputSize() const { return weights_.Rows(); }
    size_t OutputSize() const { return weights_.Cols(); }

    Matrix& Weights() { return weights_; }
    const Matrix& Weights() const { return weights_; }
    Matrix& Bias() { return bias_; }
    const Matrix& Bias() const { return bias_; }

private:
    Dense() = default;

    Matrix weights_;
    Matrix bias_;
    Matrix weight_grad_;
    Matrix bias_grad_;
    Matrix input_cache_;
};

/// Base for element-wise activations: y = f(x), dy/dx = f'(x)
class ElementwiseActivation : public Layer {
public:
    Matrix Forward(const Matrix& input, bool training) override;
    Matrix Backward(const Matrix& grad_output) override;
    Matrix Infer(const Matrix& input) const override;

protected:
    virtual float Apply(float x) const = 0;
    /// Derivative given input x and output y = f(x)
    virtual float Derivative(float x, float y) const = 0;

private:
    Matrix input_cache_;
    Matrix output_cache_;
};

class ReLU : public ElementwiseActivation {
public:
    std::string Type() const override { return "ReLU"; }
    std::unique_ptr<Layer> Clone() const override { return std::make_unique<ReLU>(*this); }

protected:
    float Apply(float x) const override { return x > 0.0f ? x : 0.0f; }
    float Derivative(float x, float) const override { return x > 0.0f ? 1.0f : 0.0f; }
};

class LeakyReLU : public ElementwiseActivation {
public:
    explicit LeakyReLU(float alpha = 0.01f) : alpha_(alpha) {}
    std::string Type() const override { return "LeakyReLU"; }
    void SerializeParameters(std::ostream& out) const override;
    std::unique_ptr<Layer> Clone() const override { return std::make_unique<LeakyReLU>(*this); }
    float Alpha() const { return alpha_; }

protected:
    float Apply(float x) const override { return x > 0.0f ? x : alpha_ * x; }
    float Derivative(float x, float) const override { return x > 0.0f ? 1.0f : alpha_; }

private:
    float alpha_;
};

class Sigmoid : public ElementwiseActivation {
public:
    std::string Type() const override { return "Sigmoid"; }
    std::unique_ptr<Layer> Clone() const override { return std::make_unique<Sigmoid>(*this); }

protected:
    float Apply(float x) const override;
    float Derivative(float, float y) const override { return y * (1.0f - y); }
};

class Tanh : public ElementwiseActivation {
public:
    std::string Type() const override { return "Tanh"; }
    std::unique_ptr<Layer> Clone() const override { return std::make_unique<Tanh>(*this); }

protected:
    float Apply(float x) const override;
    float Derivative(float, float y) const override { return 1.0f - y * y; }
};

/// Row-wise softmax (outputs of each sample sum to 1)
class Softmax : public Layer {
public:
    Matrix Forward(const Matrix& input, bool training) override;
    Matrix Backward(const Matrix& grad_output) override;
    Matrix Infer(const Matrix& input) const override;
    std::string Type() const override { return "Softmax"; }
    std::unique_ptr<Layer> Clone() const override { return std::make_unique<Softmax>(*this); }

    /// Numerically stable row-wise softmax
    static Matrix Compute(const Matrix& logits);

private:
    Matrix output_cache_;
};

/// Inverted dropout: during training zeroes each value with probability
/// `rate` and scales the rest by 1 / (1 - rate); identity at inference
class Dropout : public Layer {
public:
    explicit Dropout(float rate = 0.5f, uint32_t seed = 42);

    Matrix Forward(const Matrix& input, bool training) override;
    Matrix Backward(const Matrix& grad_output) override;
    Matrix Infer(const Matrix& input) const override { return input; }
    std::string Type() const override { return "Dropout"; }
    void SerializeParameters(std::ostream& out) const override;
    std::unique_ptr<Layer> Clone() const override { return std::make_unique<Dropout>(*this); }
    float Rate() const { return rate_; }

private:
    float rate_;
    std::mt19937 rng_;
    Matrix mask_;
};

}  // namespace nn
}  // namespace dpan
