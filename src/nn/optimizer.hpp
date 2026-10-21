// File: src/nn/optimizer.hpp
//
// Gradient-based optimizers. Step() updates every parameter from its
// gradient; per-parameter state (momentum, moments) is kept by position, so
// an optimizer instance must always be used with the same parameter list.
#pragma once

#include "nn/layers.hpp"

#include <vector>

namespace dpan {
namespace nn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void Step(const std::vector<Parameter>& parameters) = 0;
    virtual void Reset() = 0;

    float LearningRate() const { return learning_rate_; }
    void SetLearningRate(float learning_rate) { learning_rate_ = learning_rate; }

protected:
    explicit Optimizer(float learning_rate) : learning_rate_(learning_rate) {}
    float learning_rate_;
};

/// Stochastic gradient descent with optional momentum and L2 weight decay:
///   v = momentum * v + (g + weight_decay * w);  w -= learning_rate * v
class SGD : public Optimizer {
public:
    explicit SGD(float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
    void Step(const std::vector<Parameter>& parameters) override;
    void Reset() override { velocity_.clear(); }

private:
    float momentum_;
    float weight_decay_;
    std::vector<Matrix> velocity_;
};

/// Adam (Kingma & Ba, 2015) with bias correction and optional L2 weight decay
class Adam : public Optimizer {
public:
    explicit Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
                  float epsilon = 1e-8f, float weight_decay = 0.0f);
    void Step(const std::vector<Parameter>& parameters) override;
    void Reset() override;

private:
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    size_t step_{0};
    std::vector<Matrix> first_moment_;
    std::vector<Matrix> second_moment_;
};

}  // namespace nn
}  // namespace dpan
