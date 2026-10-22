// File: src/nn/optimizer.hpp
//
// Gradient-based optimizers. Step() updates every parameter from its
// gradient; per-parameter state (momentum, moments) is kept by position, so
// an optimizer instance must always be used with the same parameter list.
#pragma once

#include "nn/layers.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace dpan {
namespace nn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void Step(const std::vector<Parameter>& parameters) = 0;
    virtual void Reset() = 0;

    /// Optimizer name used in saved state ("SGD", "Adam")
    virtual std::string Type() const = 0;

    /// Write hyperparameters and per-parameter state (momentum, moments,
    /// step count) so training can resume exactly
    virtual void SaveState(std::ostream& out) const = 0;

    /// Restore state written by SaveState() of the same optimizer type
    /// @throws std::runtime_error on a type mismatch or malformed data
    virtual void LoadState(std::istream& in) = 0;

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
    std::string Type() const override { return "SGD"; }
    void SaveState(std::ostream& out) const override;
    void LoadState(std::istream& in) override;

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
    std::string Type() const override { return "Adam"; }
    void SaveState(std::ostream& out) const override;
    void LoadState(std::istream& in) override;

private:
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    size_t step_{0};
    std::vector<Matrix> first_moment_;
    std::vector<Matrix> second_moment_;
};

// ============================================================================
// Gradient clipping
// ============================================================================

/// L2 norm of all gradients together
float GradientNorm(const std::vector<Parameter>& parameters);

/// Scale all gradients so their combined L2 norm is at most max_norm
/// (no change if it already is, or if max_norm <= 0)
/// @return The norm before clipping
float ClipGradientNorm(const std::vector<Parameter>& parameters, float max_norm);

// ============================================================================
// Learning-rate schedules
// ============================================================================

/// Learning rate as a function of progress. Sequential::Fit applies a
/// schedule after each epoch; streaming learners can call it every N steps.
class LearningRateSchedule {
public:
    virtual ~LearningRateSchedule() = default;

    /// @param step Number of completed steps (epochs in Fit), >= 1
    /// @param base_rate Learning rate when training started
    /// @param current_rate Learning rate used for the last step
    /// @param loss Loss after the last step (monitored by ReduceOnPlateau)
    /// @return Learning rate for the next step
    virtual float NextRate(size_t step, float base_rate, float current_rate, float loss) = 0;
};

/// base_rate * gamma^(step / step_size)
class StepDecay : public LearningRateSchedule {
public:
    StepDecay(size_t step_size, float gamma);
    float NextRate(size_t step, float base_rate, float current_rate, float loss) override;

private:
    size_t step_size_;
    float gamma_;
};

/// base_rate * gamma^step
class ExponentialDecay : public LearningRateSchedule {
public:
    explicit ExponentialDecay(float gamma);
    float NextRate(size_t step, float base_rate, float current_rate, float loss) override;

private:
    float gamma_;
};

/// Cosine from base_rate down to min_rate over total_steps, then min_rate
class CosineAnnealing : public LearningRateSchedule {
public:
    explicit CosineAnnealing(size_t total_steps, float min_rate = 0.0f);
    float NextRate(size_t step, float base_rate, float current_rate, float loss) override;

private:
    size_t total_steps_;
    float min_rate_;
};

/// Multiply the rate by `factor` when the loss has not improved by more than
/// `threshold` (relative) for `patience` steps; never below min_rate
class ReduceOnPlateau : public LearningRateSchedule {
public:
    ReduceOnPlateau(float factor = 0.5f, size_t patience = 10, float min_rate = 1e-6f,
                    float threshold = 1e-4f);
    float NextRate(size_t step, float base_rate, float current_rate, float loss) override;

private:
    float factor_;
    size_t patience_;
    float min_rate_;
    float threshold_;
    float best_loss_;
    size_t steps_without_improvement_{0};
};

}  // namespace nn
}  // namespace dpan
