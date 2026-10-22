// File: src/nn/optimizer.cpp
#include "nn/optimizer.hpp"

#include "core/binary_io.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dpan {
namespace nn {

namespace {

/// Make per-parameter state match the parameter list (created zeroed)
void EnsureState(std::vector<Matrix>& state, const std::vector<Parameter>& parameters) {
    if (state.empty()) {
        for (const auto& param : parameters) {
            state.emplace_back(param.value->Rows(), param.value->Cols());
        }
        return;
    }
    if (state.size() != parameters.size()) {
        throw std::logic_error("Optimizer used with a different parameter list; call Reset()");
    }
    for (size_t i = 0; i < state.size(); ++i) {
        if (!state[i].SameShape(*parameters[i].value)) {
            throw std::logic_error("Optimizer state does not match parameter shapes; call Reset()");
        }
    }
}

void WriteStateMatrices(std::ostream& out, const std::vector<Matrix>& state) {
    io::WriteSize(out, state.size());
    for (const auto& m : state) {
        m.Serialize(out);
    }
}

std::vector<Matrix> ReadStateMatrices(std::istream& in) {
    std::vector<Matrix> state(io::ReadSize(in, 1u << 20));
    for (auto& m : state) {
        m = Matrix::Deserialize(in);
    }
    return state;
}

void ExpectType(std::istream& in, const std::string& type) {
    std::string stored = io::ReadString(in, 64);
    if (stored != type) {
        throw std::runtime_error("Optimizer::LoadState: state is for " + stored + ", not " + type);
    }
}

}  // namespace

SGD::SGD(float learning_rate, float momentum, float weight_decay)
    : Optimizer(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {
    if (learning_rate <= 0.0f || momentum < 0.0f || momentum >= 1.0f || weight_decay < 0.0f) {
        throw std::invalid_argument("SGD: invalid hyperparameters");
    }
}

void SGD::Step(const std::vector<Parameter>& parameters) {
    EnsureState(velocity_, parameters);
    for (size_t p = 0; p < parameters.size(); ++p) {
        float* w = parameters[p].value->Data();
        const float* g = parameters[p].gradient->Data();
        float* v = velocity_[p].Data();
        for (size_t i = 0; i < parameters[p].value->Size(); ++i) {
            float grad = g[i] + weight_decay_ * w[i];
            v[i] = momentum_ * v[i] + grad;
            w[i] -= learning_rate_ * v[i];
        }
    }
}

void SGD::SaveState(std::ostream& out) const {
    io::WriteString(out, Type());
    io::Write<float>(out, learning_rate_);
    io::Write<float>(out, momentum_);
    io::Write<float>(out, weight_decay_);
    WriteStateMatrices(out, velocity_);
}

void SGD::LoadState(std::istream& in) {
    ExpectType(in, Type());
    learning_rate_ = io::Read<float>(in);
    momentum_ = io::Read<float>(in);
    weight_decay_ = io::Read<float>(in);
    velocity_ = ReadStateMatrices(in);
}

Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      weight_decay_(weight_decay) {
    if (learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f ||
        epsilon <= 0.0f || weight_decay < 0.0f) {
        throw std::invalid_argument("Adam: invalid hyperparameters");
    }
}

void Adam::Reset() {
    step_ = 0;
    first_moment_.clear();
    second_moment_.clear();
}

void Adam::Step(const std::vector<Parameter>& parameters) {
    EnsureState(first_moment_, parameters);
    EnsureState(second_moment_, parameters);
    step_++;

    float correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
    float correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));

    for (size_t p = 0; p < parameters.size(); ++p) {
        float* w = parameters[p].value->Data();
        const float* g = parameters[p].gradient->Data();
        float* m = first_moment_[p].Data();
        float* v = second_moment_[p].Data();
        for (size_t i = 0; i < parameters[p].value->Size(); ++i) {
            float grad = g[i] + weight_decay_ * w[i];
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad;
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * grad * grad;
            float m_hat = m[i] / correction1;
            float v_hat = v[i] / correction2;
            w[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

void Adam::SaveState(std::ostream& out) const {
    io::WriteString(out, Type());
    io::Write<float>(out, learning_rate_);
    io::Write<float>(out, beta1_);
    io::Write<float>(out, beta2_);
    io::Write<float>(out, epsilon_);
    io::Write<float>(out, weight_decay_);
    io::WriteSize(out, step_);
    WriteStateMatrices(out, first_moment_);
    WriteStateMatrices(out, second_moment_);
}

void Adam::LoadState(std::istream& in) {
    ExpectType(in, Type());
    learning_rate_ = io::Read<float>(in);
    beta1_ = io::Read<float>(in);
    beta2_ = io::Read<float>(in);
    epsilon_ = io::Read<float>(in);
    weight_decay_ = io::Read<float>(in);
    step_ = io::ReadSize(in);
    first_moment_ = ReadStateMatrices(in);
    second_moment_ = ReadStateMatrices(in);
    if (first_moment_.size() != second_moment_.size()) {
        throw std::runtime_error("Adam::LoadState: moment counts differ");
    }
}

// ============================================================================
// Gradient clipping
// ============================================================================

float GradientNorm(const std::vector<Parameter>& parameters) {
    double sum = 0.0;
    for (const auto& param : parameters) {
        const float* g = param.gradient->Data();
        for (size_t i = 0; i < param.gradient->Size(); ++i) {
            sum += static_cast<double>(g[i]) * g[i];
        }
    }
    return static_cast<float>(std::sqrt(sum));
}

float ClipGradientNorm(const std::vector<Parameter>& parameters, float max_norm) {
    float norm = GradientNorm(parameters);
    if (max_norm > 0.0f && norm > max_norm) {
        float scale = max_norm / norm;
        for (const auto& param : parameters) {
            *param.gradient *= scale;
        }
    }
    return norm;
}

// ============================================================================
// Learning-rate schedules
// ============================================================================

StepDecay::StepDecay(size_t step_size, float gamma) : step_size_(step_size), gamma_(gamma) {
    if (step_size == 0 || gamma <= 0.0f || gamma > 1.0f) {
        throw std::invalid_argument("StepDecay: step_size must be > 0 and gamma in (0, 1]");
    }
}

float StepDecay::NextRate(size_t step, float base_rate, float, float) {
    return base_rate * std::pow(gamma_, static_cast<float>(step / step_size_));
}

ExponentialDecay::ExponentialDecay(float gamma) : gamma_(gamma) {
    if (gamma <= 0.0f || gamma > 1.0f) {
        throw std::invalid_argument("ExponentialDecay: gamma must be in (0, 1]");
    }
}

float ExponentialDecay::NextRate(size_t step, float base_rate, float, float) {
    return base_rate * std::pow(gamma_, static_cast<float>(step));
}

CosineAnnealing::CosineAnnealing(size_t total_steps, float min_rate)
    : total_steps_(total_steps), min_rate_(min_rate) {
    if (total_steps == 0 || min_rate < 0.0f) {
        throw std::invalid_argument("CosineAnnealing: total_steps must be > 0, min_rate >= 0");
    }
}

float CosineAnnealing::NextRate(size_t step, float base_rate, float, float) {
    if (step >= total_steps_) {
        return min_rate_;
    }
    const float pi = 3.14159265358979f;
    float progress = static_cast<float>(step) / static_cast<float>(total_steps_);
    return min_rate_ + 0.5f * (base_rate - min_rate_) * (1.0f + std::cos(pi * progress));
}

ReduceOnPlateau::ReduceOnPlateau(float factor, size_t patience, float min_rate, float threshold)
    : factor_(factor), patience_(patience), min_rate_(min_rate), threshold_(threshold),
      best_loss_(std::numeric_limits<float>::infinity()) {
    if (factor <= 0.0f || factor >= 1.0f || patience == 0 || min_rate < 0.0f || threshold < 0.0f) {
        throw std::invalid_argument("ReduceOnPlateau: invalid parameters");
    }
}

float ReduceOnPlateau::NextRate(size_t, float, float current_rate, float loss) {
    if (loss < best_loss_ * (1.0f - threshold_)) {
        best_loss_ = loss;
        steps_without_improvement_ = 0;
        return current_rate;
    }
    if (++steps_without_improvement_ >= patience_) {
        steps_without_improvement_ = 0;
        return std::max(min_rate_, current_rate * factor_);
    }
    return current_rate;
}

void ReduceOnPlateau::SaveState(std::ostream& out) const {
    io::Write<float>(out, best_loss_);
    io::WriteSize(out, steps_without_improvement_);
}

void ReduceOnPlateau::LoadState(std::istream& in) {
    best_loss_ = io::Read<float>(in);
    steps_without_improvement_ = io::ReadSize(in);
}

}  // namespace nn
}  // namespace dpan
