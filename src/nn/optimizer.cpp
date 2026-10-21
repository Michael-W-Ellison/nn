// File: src/nn/optimizer.cpp
#include "nn/optimizer.hpp"

#include <cmath>
#include <stdexcept>

namespace dpan {
namespace nn {

namespace {

/// Make per-parameter state match the parameter list (created zeroed)
void EnsureState(std::vector<Matrix>& state, const std::vector<Parameter>& parameters) {
    if (state.size() != parameters.size()) {
        if (!state.empty()) {
            throw std::logic_error("Optimizer used with a different parameter list; call Reset()");
        }
        for (const auto& param : parameters) {
            state.emplace_back(param.value->Rows(), param.value->Cols());
        }
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

}  // namespace nn
}  // namespace dpan
